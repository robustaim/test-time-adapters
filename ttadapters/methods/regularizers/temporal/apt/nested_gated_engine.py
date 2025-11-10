"""
Nested APT (Gated): Multi-timescale + Confidence-Gated Feature Adaptation

This implementation combines:
1. Temporal confidence-gated feature adaptation
2. Multi-timescale parameter updates
3. Adaptive learning rates based on tracking quality

Key idea: Temporal quality controls both feature adaptation momentum
AND learning rates for different parameter levels.
"""
import torch
from torch import nn, optim
import numpy as np
from collections import deque
from typing import Dict, List, Optional

from ....base import AdaptationEngine
from .....models.base import BaseModel, ModelProvider

from .config import APTConfig
from .tracker import TemporalTracker


class ConfidenceGatedFeatureAdapter(nn.Module):
    """
    Feature adapter with temporal confidence-gated momentum

    Key: Adaptation speed is controlled by temporal quality
    - High quality (good tracking) → Fast adaptation
    - Low quality (poor tracking) → Slow adaptation
    """

    def __init__(self, num_channels: int, alpha_base: float = 0.1):
        super().__init__()

        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones(num_channels))

        # Source statistics
        self.register_buffer('source_mean', torch.zeros(num_channels))
        self.register_buffer('source_var', torch.ones(num_channels))

        self.alpha_base = alpha_base
        self.initialized = False

    def initialize_source_stats(self, features: torch.Tensor):
        """Initialize source statistics"""
        if not self.initialized:
            with torch.no_grad():
                self.source_mean.copy_(features.mean(dim=[0, 2, 3]))
                self.source_var.copy_(features.var(dim=[0, 2, 3]))
                self.running_mean.copy_(self.source_mean)
                self.running_var.copy_(self.source_var)
                self.initialized = True

    def forward(self, features: torch.Tensor, temporal_quality: float) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W]
            temporal_quality: scalar in [0, 1]
                - 1.0: Perfect temporal consistency → Fast adaptation
                - 0.0: No temporal consistency → No adaptation
        """
        self.initialize_source_stats(features)

        # Compute batch statistics
        batch_mean = features.mean(dim=[0, 2, 3])
        batch_var = features.var(dim=[0, 2, 3])

        # Temporal quality gates the adaptation momentum
        alpha_dynamic = self.alpha_base * temporal_quality

        # Update running statistics
        with torch.no_grad():
            self.running_mean = (1 - alpha_dynamic) * self.running_mean + alpha_dynamic * batch_mean
            self.running_var = (1 - alpha_dynamic) * self.running_var + alpha_dynamic * batch_var

        # Normalize
        features_norm = (features - self.running_mean.view(1, -1, 1, 1)) / \
                        torch.sqrt(self.running_var.view(1, -1, 1, 1) + 1e-5)

        # Re-scale to source distribution
        features_adapted = features_norm * torch.sqrt(self.source_var.view(1, -1, 1, 1) + 1e-5) + \
                           self.source_mean.view(1, -1, 1, 1)

        return features_adapted


class GatedNestedAPTConfig(APTConfig):
    """Configuration for Gated Nested APT"""
    adaptation_name: str = "GatedNestedAPT"

    # Feature adaptation settings
    use_feature_adaptation: bool = True
    feature_alpha_base: float = 0.1

    # Nested Learning: Multi-timescale update frequencies
    fast_update_freq: int = 1      # Feature adapters
    medium_update_freq: int = 5    # BatchNorm
    slow_update_freq: int = 20     # FPN/Backbone

    # Nested Learning: Base learning rates (modulated by quality)
    fast_lr: float = 1e-4
    medium_lr: float = 1e-5
    slow_lr: float = 1e-6

    # Adaptive LR modulation
    use_adaptive_lr: bool = True
    lr_modulation_strength: float = 0.5  # How much quality affects LR

    # Quality computation
    quality_ema_decay: float = 0.9  # EMA for temporal quality
    quality_method: str = "weighted"  # "iou", "ratio", "weighted"

    # Update strategy
    update_bn: bool = True
    update_fpn_last_layer: bool = False
    update_backbone_last_layer: bool = False


class GatedNestedAPTEngine(AdaptationEngine):
    """
    Gated Nested APT: Adaptive Multi-timescale Updates

    This engine combines:
    1. Confidence-gated feature adaptation (gradient-free)
    2. Multi-timescale parameter updates (gradient-based)
    3. Quality-based learning rate modulation

    Key innovation: Temporal quality controls BOTH:
    - Feature adaptation momentum (alpha)
    - Learning rates for all levels (fast/medium/slow)

    When tracking is good → Adapt aggressively
    When tracking is poor → Adapt conservatively
    """
    model_name = "GatedNestedAPT"

    def __init__(self, base_model: BaseModel, config: GatedNestedAPTConfig):
        super().__init__(base_model, config)
        self.config: GatedNestedAPTConfig = config

        # Initialize temporal tracker
        self.tracker = TemporalTracker(
            max_age=config.max_age,
            min_hits=config.min_hits,
            iou_threshold=config.iou_threshold
        )

        # Frame buffer and counters
        self.frame_buffer = deque(maxlen=config.buffer_size)
        self.frame_count = 0

        # Statistics
        self.adaptation_steps = 0
        self.total_loss = 0.0

        # Temporal quality tracking
        self.temporal_quality = 1.0  # Start optimistic
        self.quality_history = deque(maxlen=100)

        # Nested Learning: Track updates per level
        self.updates_per_level = {
            'fast': 0,
            'medium': 0,
            'slow': 0
        }

        # Feature adapters
        self.feature_adapters = nn.ModuleDict()
        if config.use_feature_adaptation:
            self._setup_feature_adapters()

    def _setup_feature_adapters(self):
        """Setup confidence-gated feature adapters"""
        if self.base_model.model_provider == ModelProvider.Detectron2:
            if hasattr(self.base_model, 'backbone'):
                out_features = self.base_model.backbone._out_features
                out_channels = self.base_model.backbone._out_feature_channels

                for stage_name in out_features:
                    num_channels = out_channels[stage_name]
                    adapter = ConfidenceGatedFeatureAdapter(
                        num_channels=num_channels,
                        alpha_base=self.config.feature_alpha_base
                    )
                    self.feature_adapters[stage_name] = adapter.to(self.device)

    def _get_bn_params(self) -> List[nn.Parameter]:
        """Get BatchNorm parameters (fast update)"""
        params = []
        for module in self.base_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                params.extend(module.parameters())
            if "FrozenBatchNorm2d" in module.__class__.__name__:
                if hasattr(module, 'weight') and not isinstance(module.weight, nn.Parameter):
                    module.weight = nn.Parameter(module.weight)
                if hasattr(module, 'bias') and not isinstance(module.bias, nn.Parameter):
                    module.bias = nn.Parameter(module.bias)
                if hasattr(module, 'weight'):
                    params.append(module.weight)
                if hasattr(module, 'bias'):
                    params.append(module.bias)
        return params

    def _get_fpn_params(self) -> List[nn.Parameter]:
        """Get FPN parameters (medium update)"""
        params = []
        if self.config.update_fpn_last_layer and hasattr(self.base_model, 'backbone'):
            if hasattr(self.base_model.backbone, 'fpn'):
                fpn_layers = list(self.base_model.backbone.fpn.children())
                if len(fpn_layers) > 0:
                    params.extend(fpn_layers[-1].parameters())
        return params

    def _get_backbone_params(self) -> List[nn.Parameter]:
        """Get backbone parameters (slow update)"""
        params = []
        if self.config.update_backbone_last_layer and hasattr(self.base_model, 'backbone'):
            if hasattr(self.base_model.backbone, 'res5'):
                params.extend(self.base_model.backbone.res5.parameters())
        return params

    @property
    def optimizer(self):
        """Create multi-timescale optimizer"""
        if self._optimizer is None:
            param_groups = []

            # Fast: BatchNorm
            if self.config.update_bn:
                bn_params = self._get_bn_params()
                if bn_params:
                    param_groups.append({
                        'params': bn_params,
                        'lr': self.config.fast_lr,
                        'name': 'fast',
                        'update_freq': self.config.fast_update_freq
                    })

            # Medium: FPN
            fpn_params = self._get_fpn_params()
            if fpn_params:
                param_groups.append({
                    'params': fpn_params,
                    'lr': self.config.medium_lr,
                    'name': 'medium',
                    'update_freq': self.config.medium_update_freq
                })

            # Slow: Backbone
            backbone_params = self._get_backbone_params()
            if backbone_params:
                param_groups.append({
                    'params': backbone_params,
                    'lr': self.config.slow_lr,
                    'name': 'slow',
                    'update_freq': self.config.slow_update_freq
                })

            if self.config.optim == "SGD":
                self._optimizer = optim.SGD(param_groups, momentum=0.9)
            elif self.config.optim == "Adam":
                self._optimizer = optim.Adam(param_groups)
            elif self.config.optim == "AdamW":
                self._optimizer = optim.AdamW(param_groups)
            else:
                raise ValueError(f"Unknown optimizer: {self.config.optim}")

        return self._optimizer

    def online_parameters(self):
        """Return all adaptable parameters"""
        params = []
        if self.config.update_bn:
            params.extend(self._get_bn_params())
        params.extend(self._get_fpn_params())
        params.extend(self._get_backbone_params())
        return params

    def compute_temporal_quality(self, matched_ious: List[float], total_tracks: int, matched_tracks: int) -> float:
        """
        Compute temporal quality metric

        Args:
            matched_ious: List of IOU values for matched boxes
            total_tracks: Total number of active tracks
            matched_tracks: Number of successfully matched tracks

        Returns:
            quality: float in [0, 1]
        """
        if self.config.quality_method == "iou":
            # Simple: average IOU of matches
            if len(matched_ious) > 0:
                quality = np.mean(matched_ious)
            else:
                quality = 0.0

        elif self.config.quality_method == "ratio":
            # Match ratio: what fraction of tracks were matched?
            if total_tracks > 0:
                quality = matched_tracks / total_tracks
            else:
                quality = 0.5

        elif self.config.quality_method == "weighted":
            # Weighted combination
            if len(matched_ious) > 0 and total_tracks > 0:
                avg_iou = np.mean(matched_ious)
                match_ratio = matched_tracks / total_tracks
                quality = 0.7 * avg_iou + 0.3 * match_ratio
            else:
                quality = 0.0

        else:
            raise ValueError(f"Unknown quality method: {self.config.quality_method}")

        return float(quality)

    def update_temporal_quality(self, current_quality: float):
        """Update temporal quality with EMA"""
        self.quality_history.append(current_quality)

        # EMA update
        self.temporal_quality = (
                self.config.quality_ema_decay * self.temporal_quality +
                (1 - self.config.quality_ema_decay) * current_quality
        )

    def modulate_learning_rates(self):
        """Modulate learning rates based on temporal quality"""
        if not self.config.use_adaptive_lr:
            return

        # Quality modulation: [0.5, 1.5] range
        # High quality → Higher LR (1.5x)
        # Low quality → Lower LR (0.5x)
        modulation = 1.0 + self.config.lr_modulation_strength * (2 * self.temporal_quality - 1)
        modulation = max(0.5, min(1.5, modulation))  # Clamp

        for group in self.optimizer.param_groups:
            base_lr = group['lr'] / getattr(group, '_last_modulation', 1.0)
            group['lr'] = base_lr * modulation
            group['_last_modulation'] = modulation

    @property
    def loss_function(self):
        """Get loss function"""
        if self._loss_function is None:
            if self.config.loss_type == "smooth_l1":
                self._loss_function = nn.SmoothL1Loss()
            elif self.config.loss_type == "l1":
                self._loss_function = nn.L1Loss()
            elif self.config.loss_type == "l2":
                self._loss_function = nn.MSELoss()
            elif self.config.loss_type == "giou":
                from torchvision.ops import generalized_box_iou_loss
                self._loss_function = lambda p, t: generalized_box_iou_loss(p, t).mean()
        return self._loss_function

    def extract_detections(self, outputs, conf_threshold: Optional[float] = None):
        """Extract detections from model outputs"""
        if conf_threshold is None:
            conf_threshold = self.config.conf_threshold

        if self.base_model.model_provider == ModelProvider.Detectron2:
            instances = outputs['instances']
            scores = instances.scores.detach().cpu().numpy()
            boxes = instances.pred_boxes.tensor.detach().cpu().numpy()
            classes = instances.pred_classes.detach().cpu().numpy()

            mask = scores >= conf_threshold
            return boxes[mask], scores[mask], classes[mask]

        raise NotImplementedError(f"Unsupported model provider")

    def compute_temporal_loss(self, current_boxes, predicted_boxes, current_classes, predicted_classes):
        """Compute temporal loss and quality metrics"""
        if len(predicted_boxes) == 0 or len(current_boxes) == 0:
            return torch.tensor(0.0, device=self.device), []

        if isinstance(predicted_boxes, np.ndarray):
            predicted_boxes = torch.from_numpy(predicted_boxes).float().to(self.device)
        if isinstance(predicted_classes, np.ndarray):
            predicted_classes = torch.from_numpy(predicted_classes).long().to(self.device)

        from torchvision.ops import box_iou

        loss = torch.tensor(0.0, device=self.device)
        n_matched = 0
        matched_ious = []

        for cls_id in torch.unique(predicted_classes):
            pred_mask = predicted_classes == cls_id
            curr_mask = current_classes == cls_id

            if not curr_mask.any() or not pred_mask.any():
                continue

            pred_cls_boxes = predicted_boxes[pred_mask]
            curr_cls_boxes = current_boxes[curr_mask]

            iou_matrix = box_iou(curr_cls_boxes, pred_cls_boxes)

            if iou_matrix.numel() > 0:
                max_ious, max_indices = iou_matrix.max(dim=0)
                valid_matches = max_ious > self.config.iou_threshold

                if valid_matches.any():
                    matched_curr = curr_cls_boxes[max_indices[valid_matches]]
                    matched_pred = pred_cls_boxes[valid_matches]

                    loss += self.loss_function(matched_curr, matched_pred)
                    n_matched += valid_matches.sum().item()

                    # Record IOUs for quality computation
                    matched_ious.extend(max_ious[valid_matches].cpu().numpy().tolist())

        if n_matched > 0:
            loss = loss / n_matched

        return loss * self.config.loss_weight, matched_ious

    def forward(self, *args, **kwargs):
        """Forward with gated multi-timescale adaptation"""
        self.frame_count += 1

        # Patch backbone with gated feature adapters
        original_backbone_forward = None
        if hasattr(self.base_model, 'backbone'):
            original_backbone_forward = self.base_model.backbone.forward

        if self.adapting and self.config.use_feature_adaptation and original_backbone_forward:
            def adapted_backbone_forward(x):
                features = original_backbone_forward(x)

                # Apply gated feature adapters
                # temporal_quality controls adaptation speed
                adapted_features = {}
                for stage_name, feat in features.items():
                    if stage_name in self.feature_adapters:
                        adapted_features[stage_name] = self.feature_adapters[stage_name](
                            feat, self.temporal_quality
                        )
                    else:
                        adapted_features[stage_name] = feat

                return adapted_features

            self.base_model.backbone.forward = adapted_backbone_forward

        # Forward pass
        outputs = self.base_model(*args, **kwargs)

        # Restore
        if original_backbone_forward:
            self.base_model.backbone.forward = original_backbone_forward

        if not self.adapting:
            return outputs

        # Adaptation
        if self.base_model.model_provider == ModelProvider.Detectron2:
            if isinstance(outputs, list):
                losses = []
                all_matched_ious = []
                total_tracks = len(self.tracker.trackers)
                matched_count = 0

                for output in outputs:
                    boxes, scores, classes = self.extract_detections(output)

                    if len(boxes) > 0:
                        predicted_boxes, predicted_classes, _ = self.tracker.update(boxes, classes)
                    else:
                        predicted_boxes, predicted_classes, _ = self.tracker.update(
                            np.empty((0, 4)), np.empty(0)
                        )

                    if len(predicted_boxes) > 0:
                        current_boxes = output['instances'].pred_boxes.tensor
                        current_classes = output['instances'].pred_classes
                        current_scores = output['instances'].scores

                        conf_mask = current_scores >= self.config.conf_threshold

                        if conf_mask.any():
                            loss, matched_ious = self.compute_temporal_loss(
                                current_boxes[conf_mask],
                                predicted_boxes,
                                current_classes[conf_mask],
                                predicted_classes
                            )

                            if loss.item() > 0:
                                losses.append(loss)
                                all_matched_ious.extend(matched_ious)
                                matched_count += len(matched_ious)

                # Update temporal quality
                if all_matched_ious:
                    current_quality = self.compute_temporal_quality(
                        all_matched_ious, total_tracks, matched_count
                    )
                    self.update_temporal_quality(current_quality)

                # Modulate learning rates based on quality
                self.modulate_learning_rates()

                # Backpropagation with multi-timescale updates
                if len(losses) > 0:
                    temporal_loss = torch.stack(losses).mean()

                    self.optimizer.zero_grad()
                    temporal_loss.backward()

                    # Selective updates
                    for group in self.optimizer.param_groups:
                        update_freq = group.get('update_freq', 1)
                        level_name = group.get('name', 'unknown')

                        if self.frame_count % update_freq == 0:
                            self.updates_per_level[level_name] = self.updates_per_level.get(level_name, 0) + 1
                        else:
                            for param in group['params']:
                                if param.grad is not None:
                                    param.grad = None

                    self.optimizer.step()

                    self.adaptation_steps += 1
                    self.total_loss += temporal_loss.item()

        return outputs

    def reset(self):
        """Reset adaptation state"""
        self.tracker.reset()
        self.frame_buffer.clear()
        self.frame_count = 0
        self.adaptation_steps = 0
        self.total_loss = 0.0
        self.temporal_quality = 1.0
        self.quality_history.clear()
        self.updates_per_level = {k: 0 for k in self.updates_per_level}

        for adapter in self.feature_adapters.values():
            adapter.initialized = False

        super().reset()

    def get_adaptation_stats(self) -> Dict:
        """Get detailed adaptation statistics"""
        return {
            'adaptation_steps': self.adaptation_steps,
            'frame_count': self.frame_count,
            'avg_temporal_loss': self.total_loss / max(1, self.adaptation_steps),
            'total_loss': self.total_loss,
            'num_tracks': len(self.tracker.trackers),
            'temporal_quality': self.temporal_quality,
            'avg_quality': np.mean(self.quality_history) if self.quality_history else 0.0,
            'updates_fast': self.updates_per_level.get('fast', 0),
            'updates_medium': self.updates_per_level.get('medium', 0),
            'updates_slow': self.updates_per_level.get('slow', 0),
        }

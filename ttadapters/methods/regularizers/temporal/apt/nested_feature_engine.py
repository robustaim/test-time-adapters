"""
APT with Nested Learning: Multi-timescale Learnable Feature Adaptation

This implementation combines:
1. APT's temporal consistency via Kalman tracking
2. Learnable feature adapters (scale/shift parameters)
3. Nested Learning's multi-timescale parameter updates

Key idea: Different parameter groups update at different frequencies,
mimicking the brain's neuroplasticity with fast and slow learning rates.
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


class LearnableFeatureAdapter(nn.Module):
    """
    Feature adapter with learnable scale/shift parameters

    This adapter normalizes features using running statistics (gradient-free)
    and then applies learnable affine transformation (gradient-based).
    """

    def __init__(self, num_channels: int, alpha_base: float = 0.1):
        super().__init__()

        # Running statistics (gradient-free, NORM-style)
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones(num_channels))

        # Source statistics (preserve source domain knowledge)
        self.register_buffer('source_mean', torch.zeros(num_channels))
        self.register_buffer('source_var', torch.ones(num_channels))

        # Learnable adaptation parameters (gradient-based, trained by APT)
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.shift = nn.Parameter(torch.zeros(num_channels))

        self.alpha = alpha_base
        self.initialized = False

    def initialize_source_stats(self, features: torch.Tensor):
        """Initialize source statistics from first batch"""
        if not self.initialized:
            with torch.no_grad():
                self.source_mean.copy_(features.mean(dim=[0, 2, 3]))
                self.source_var.copy_(features.var(dim=[0, 2, 3]))
                self.running_mean.copy_(self.source_mean)
                self.running_var.copy_(self.source_var)
                self.initialized = True

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W]

        Returns:
            features_adapted: [B, C, H, W] with gradient flow through scale/shift
        """
        self.initialize_source_stats(features)

        # 1. Update running statistics (gradient-free, NORM-style)
        with torch.no_grad():
            batch_mean = features.mean(dim=[0, 2, 3])
            batch_var = features.var(dim=[0, 2, 3])

            self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * batch_mean
            self.running_var = (1 - self.alpha) * self.running_var + self.alpha * batch_var

        # 2. Normalize using running statistics
        features_norm = (features - self.running_mean.view(1, -1, 1, 1)) / \
                        torch.sqrt(self.running_var.view(1, -1, 1, 1) + 1e-5)

        # 3. Learnable transformation (gradient flows here!)
        # This is what APT temporal loss optimizes
        features_adapted = features_norm * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)

        return features_adapted


class NestedAPTConfig(APTConfig):
    """Configuration for Nested APT"""
    adaptation_name: str = "NestedAPT"

    # Feature adaptation settings
    use_feature_adaptation: bool = True
    feature_alpha: float = 0.1

    # Nested Learning: Multi-timescale update frequencies
    fast_update_freq: int = 1      # Feature adapters update every frame
    medium_update_freq: int = 5    # BatchNorm updates every 5 frames
    slow_update_freq: int = 20     # FPN/Backbone updates every 20 frames

    # Nested Learning: Learning rates for each timescale
    fast_lr: float = 1e-4          # Feature adapter scale/shift
    medium_lr: float = 1e-5        # BatchNorm parameters
    slow_lr: float = 1e-6          # FPN/Backbone parameters

    # Update strategy
    update_bn: bool = True
    update_fpn_last_layer: bool = False
    update_backbone_last_layer: bool = False

    # Regularization
    use_l2_regularization: bool = True
    l2_weight: float = 0.01

    # Adaptive learning rate adjustment
    use_adaptive_lr: bool = False   # Meta-learning for LR adjustment
    lr_adaptation_rate: float = 0.1


class NestedAPTEngine(AdaptationEngine):
    """
    Nested APT: Multi-timescale Test-time Adaptation

    Combines:
    - APT's temporal consistency (Kalman filter tracking)
    - Learnable feature adapters (gradient-based)
    - Nested Learning's multi-timescale updates

    Parameter hierarchy:
    - Fast (every frame): Feature adapter scale/shift
    - Medium (every 5 frames): BatchNorm parameters
    - Slow (every 20 frames): FPN/Backbone parameters
    """
    model_name = "NestedAPT"

    def __init__(self, base_model: BaseModel, config: NestedAPTConfig):
        super().__init__(base_model, config)
        self.config: NestedAPTConfig = config

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
        self.total_reg_loss = 0.0

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

        # Adaptive LR controller (optional meta-learning)
        if config.use_adaptive_lr:
            self.lr_controller = nn.Linear(3, 3)  # [quality_metrics] -> [lr_fast, lr_med, lr_slow]

    def _setup_feature_adapters(self):
        """Setup learnable feature adapters for each backbone stage"""
        if self.base_model.model_provider == ModelProvider.Detectron2:
            if hasattr(self.base_model, 'backbone'):
                out_features = self.base_model.backbone._out_features
                out_channels = self.base_model.backbone._out_feature_channels

                for stage_name in out_features:
                    num_channels = out_channels[stage_name]
                    adapter = LearnableFeatureAdapter(
                        num_channels=num_channels,
                        alpha_base=self.config.feature_alpha
                    )
                    self.feature_adapters[stage_name] = adapter.to(self.device)

    def _get_adapter_params(self) -> List[nn.Parameter]:
        """Get feature adapter parameters (fast update)"""
        params = []
        for adapter in self.feature_adapters.values():
            if hasattr(adapter, 'scale'):
                params.append(adapter.scale)
            if hasattr(adapter, 'shift'):
                params.append(adapter.shift)
        return params

    def _get_bn_params(self) -> List[nn.Parameter]:
        """Get BatchNorm parameters (medium update)"""
        params = []
        for module in self.base_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                params.extend(module.parameters())
            if "FrozenBatchNorm2d" in module.__class__.__name__:
                # Unfreeze FrozenBatchNorm
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
        """Get FPN last layer parameters (slow update)"""
        params = []
        if self.config.update_fpn_last_layer and hasattr(self.base_model, 'backbone'):
            if hasattr(self.base_model.backbone, 'fpn'):
                # Get last FPN layer parameters
                fpn_layers = list(self.base_model.backbone.fpn.children())
                if len(fpn_layers) > 0:
                    params.extend(fpn_layers[-1].parameters())
        return params

    def _get_backbone_params(self) -> List[nn.Parameter]:
        """Get backbone last layer parameters (slow update)"""
        params = []
        if self.config.update_backbone_last_layer and hasattr(self.base_model, 'backbone'):
            # Get last ResNet block parameters
            if hasattr(self.base_model.backbone, 'res5'):
                params.extend(self.base_model.backbone.res5.parameters())
        return params

    @property
    def optimizer(self):
        """Create multi-timescale optimizer with parameter groups"""
        if self._optimizer is None:
            param_groups = []

            # Fast: Feature adapter parameters (update every frame)
            adapter_params = self._get_adapter_params()
            if adapter_params:
                param_groups.append({
                    'params': adapter_params,
                    'lr': self.config.fast_lr,
                    'name': 'fast',
                    'update_freq': self.config.fast_update_freq
                })

            # Medium: BatchNorm parameters (update every 5 frames)
            if self.config.update_bn:
                bn_params = self._get_bn_params()
                if bn_params:
                    param_groups.append({
                        'params': bn_params,
                        'lr': self.config.medium_lr,
                        'name': 'medium',
                        'update_freq': self.config.medium_update_freq
                    })

            # Slow: FPN + Backbone parameters (update every 20 frames)
            slow_params = []
            slow_params.extend(self._get_fpn_params())
            slow_params.extend(self._get_backbone_params())
            if slow_params:
                param_groups.append({
                    'params': slow_params,
                    'lr': self.config.slow_lr,
                    'name': 'slow',
                    'update_freq': self.config.slow_update_freq
                })

            # Create optimizer
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
        params.extend(self._get_adapter_params())
        if self.config.update_bn:
            params.extend(self._get_bn_params())
        params.extend(self._get_fpn_params())
        params.extend(self._get_backbone_params())
        return params

    @property
    def loss_function(self):
        """Get loss function based on config"""
        if self._loss_function is None:
            if self.config.loss_type == "smooth_l1":
                self._loss_function = nn.SmoothL1Loss()
            elif self.config.loss_type == "l1":
                self._loss_function = nn.L1Loss()
            elif self.config.loss_type == "l2":
                self._loss_function = nn.MSELoss()
            elif self.config.loss_type == "giou":
                self._loss_function = self._giou_loss
            else:
                raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        return self._loss_function

    def _giou_loss(self, pred_boxes, target_boxes):
        """Generalized IoU loss"""
        from torchvision.ops import generalized_box_iou_loss
        return generalized_box_iou_loss(pred_boxes, target_boxes).mean()

    def compute_regularization_loss(self) -> torch.Tensor:
        """
        L2 regularization to prevent adapters from deviating too much

        Penalizes:
        - scale deviating from 1.0
        - shift deviating from 0.0
        """
        if not self.config.use_l2_regularization:
            return torch.tensor(0.0, device=self.device)

        reg_loss = torch.tensor(0.0, device=self.device)

        for adapter in self.feature_adapters.values():
            if hasattr(adapter, 'scale'):
                reg_loss += ((adapter.scale - 1.0) ** 2).mean()
            if hasattr(adapter, 'shift'):
                reg_loss += (adapter.shift ** 2).mean()

        return self.config.l2_weight * reg_loss

    def _adjust_learning_rates(self):
        """
        Adaptive learning rate adjustment based on adaptation quality

        Uses meta-learning to dynamically adjust LRs based on:
        - Tracking quality (average IOU)
        - Loss trend
        - Domain shift magnitude
        """
        if not self.config.use_adaptive_lr:
            return

        # Compute adaptation quality metrics
        quality_metrics = torch.tensor([
            self.tracker.get_avg_iou() if hasattr(self.tracker, 'get_avg_iou') else 0.5,
            max(0, 1.0 - self.total_loss / max(1, self.adaptation_steps)),
            min(1.0, len(self.tracker.trackers) / 10.0)  # Normalized number of tracks
        ], device=self.device)

        # Predict new learning rates
        lr_multipliers = torch.sigmoid(self.lr_controller(quality_metrics))

        # Update optimizer learning rates
        for i, group in enumerate(self.optimizer.param_groups):
            if i < len(lr_multipliers):
                base_lr = group['lr']
                group['lr'] = base_lr * (0.5 + lr_multipliers[i].item())

    def extract_detections(self, outputs, conf_threshold: Optional[float] = None):
        """
        Extract detections from model outputs

        Returns:
            boxes: (N, 4) numpy array in [x1, y1, x2, y2] format
            scores: (N,) numpy array of confidence scores
            classes: (N,) numpy array of class labels
        """
        if conf_threshold is None:
            conf_threshold = self.config.conf_threshold

        if self.base_model.model_provider == ModelProvider.Detectron2:
            instances = outputs['instances']
            scores = instances.scores.detach().cpu().numpy()
            boxes = instances.pred_boxes.tensor.detach().cpu().numpy()
            classes = instances.pred_classes.detach().cpu().numpy()

            mask = scores >= conf_threshold
            return boxes[mask], scores[mask], classes[mask]

        raise NotImplementedError(f"Unsupported model provider: {self.base_model.model_provider}")

    def compute_temporal_loss(self, current_boxes, predicted_boxes, current_classes, predicted_classes):
        """
        Compute temporal consistency loss

        Matches current detections with Kalman predictions using IOU,
        then computes box regression loss for matched pairs.
        """
        if len(predicted_boxes) == 0 or len(current_boxes) == 0:
            return torch.tensor(0.0, device=self.device)

        # Convert to tensors if needed
        if isinstance(predicted_boxes, np.ndarray):
            predicted_boxes = torch.from_numpy(predicted_boxes).float().to(self.device)
        if isinstance(predicted_classes, np.ndarray):
            predicted_classes = torch.from_numpy(predicted_classes).long().to(self.device)

        from torchvision.ops import box_iou

        loss = torch.tensor(0.0, device=self.device)
        n_matched = 0

        # Match by class and IOU
        for cls_id in torch.unique(predicted_classes):
            pred_mask = predicted_classes == cls_id
            curr_mask = current_classes == cls_id

            if not curr_mask.any() or not pred_mask.any():
                continue

            pred_cls_boxes = predicted_boxes[pred_mask]
            curr_cls_boxes = current_boxes[curr_mask]

            # Compute IOU matrix
            iou_matrix = box_iou(curr_cls_boxes, pred_cls_boxes)

            if iou_matrix.numel() > 0:
                max_ious, max_indices = iou_matrix.max(dim=0)
                valid_matches = max_ious > self.config.iou_threshold

                if valid_matches.any():
                    matched_curr = curr_cls_boxes[max_indices[valid_matches]]
                    matched_pred = pred_cls_boxes[valid_matches]

                    # Compute loss
                    loss += self.loss_function(matched_curr, matched_pred)
                    n_matched += valid_matches.sum().item()

        # Normalize by number of matches
        if n_matched > 0:
            loss = loss / n_matched

        return loss * self.config.loss_weight

    def forward(self, *args, **kwargs):
        """
        Forward pass with multi-timescale adaptation

        Process:
        1. Apply feature adapters to backbone features
        2. Get model predictions
        3. Track objects with Kalman filter
        4. Compute temporal consistency loss
        5. Selective parameter updates based on frame count
        """
        # Increment frame counter
        self.frame_count += 1

        # Store original backbone forward
        original_backbone_forward = None
        if hasattr(self.base_model, 'backbone'):
            original_backbone_forward = self.base_model.backbone.forward

        # Patch backbone with feature adapters
        if self.adapting and self.config.use_feature_adaptation and original_backbone_forward:
            def adapted_backbone_forward(x):
                features = original_backbone_forward(x)

                # Apply learnable feature adapters
                # Gradient will flow through scale/shift parameters
                adapted_features = {}
                for stage_name, feat in features.items():
                    if stage_name in self.feature_adapters:
                        adapted_features[stage_name] = self.feature_adapters[stage_name](feat)
                    else:
                        adapted_features[stage_name] = feat

                return adapted_features

            self.base_model.backbone.forward = adapted_backbone_forward

        # Get model predictions
        outputs = self.base_model(*args, **kwargs)

        # Restore original forward
        if original_backbone_forward:
            self.base_model.backbone.forward = original_backbone_forward

        # If not adapting, return outputs directly
        if not self.adapting:
            return outputs

        # Adaptation process
        if self.base_model.model_provider == ModelProvider.Detectron2:
            if isinstance(outputs, list):
                losses = []

                for output in outputs:
                    # Extract detections
                    boxes, scores, classes = self.extract_detections(output)

                    # Update tracker and get predictions
                    if len(boxes) > 0:
                        predicted_boxes, predicted_classes, _ = self.tracker.update(boxes, classes)
                    else:
                        predicted_boxes, predicted_classes, _ = self.tracker.update(
                            np.empty((0, 4)), np.empty(0)
                        )

                    # Compute temporal consistency loss
                    if len(predicted_boxes) > 0:
                        current_boxes = output['instances'].pred_boxes.tensor
                        current_classes = output['instances'].pred_classes
                        current_scores = output['instances'].scores

                        conf_mask = current_scores >= self.config.conf_threshold

                        if conf_mask.any():
                            loss = self.compute_temporal_loss(
                                current_boxes[conf_mask],
                                predicted_boxes,
                                current_classes[conf_mask],
                                predicted_classes
                            )

                            if loss.item() > 0:
                                losses.append(loss)

                # Backpropagation with multi-timescale updates
                if len(losses) > 0:
                    temporal_loss = torch.stack(losses).mean()
                    reg_loss = self.compute_regularization_loss()
                    total_loss = temporal_loss + reg_loss

                    # Compute gradients
                    self.optimizer.zero_grad()
                    total_loss.backward()

                    # Selective parameter updates based on frame count
                    # This is the core of Nested Learning!
                    for group in self.optimizer.param_groups:
                        update_freq = group.get('update_freq', 1)
                        level_name = group.get('name', 'unknown')

                        if self.frame_count % update_freq == 0:
                            # Update this level
                            self.updates_per_level[level_name] = self.updates_per_level.get(level_name, 0) + 1
                        else:
                            # Zero out gradients for this level (no update)
                            for param in group['params']:
                                if param.grad is not None:
                                    param.grad = None

                    # Apply optimizer step (only non-zeroed gradients will update)
                    self.optimizer.step()

                    # Update statistics
                    self.adaptation_steps += 1
                    self.total_loss += temporal_loss.item()
                    self.total_reg_loss += reg_loss.item()

                    # Adaptive LR adjustment (optional)
                    if self.config.use_adaptive_lr and self.adaptation_steps % 10 == 0:
                        self._adjust_learning_rates()

        return outputs

    def reset(self):
        """Reset adaptation state"""
        self.tracker.reset()
        self.frame_buffer.clear()
        self.frame_count = 0
        self.adaptation_steps = 0
        self.total_loss = 0.0
        self.total_reg_loss = 0.0
        self.updates_per_level = {k: 0 for k in self.updates_per_level}

        # Reset feature adapters
        for adapter in self.feature_adapters.values():
            adapter.initialized = False

        super().reset()

    def get_adaptation_stats(self) -> Dict:
        """Get detailed adaptation statistics"""
        return {
            'adaptation_steps': self.adaptation_steps,
            'frame_count': self.frame_count,
            'avg_temporal_loss': self.total_loss / max(1, self.adaptation_steps),
            'avg_reg_loss': self.total_reg_loss / max(1, self.adaptation_steps),
            'total_loss': self.total_loss + self.total_reg_loss,
            'num_tracks': len(self.tracker.trackers),
            'updates_fast': self.updates_per_level.get('fast', 0),
            'updates_medium': self.updates_per_level.get('medium', 0),
            'updates_slow': self.updates_per_level.get('slow', 0),
        }

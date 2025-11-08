"""
APT with Temporal Confidence-Gated Feature Adaptation

Method 1: Global temporal quality controls feature adaptation momentum
"""
import torch
from torch import nn, optim
import numpy as np
from collections import deque

from ....base import AdaptationEngine
from .....models.base import BaseModel, ModelProvider

from .config import APTConfig
from .tracker import TemporalTracker


class ConfidenceGatedFeatureAdapter(nn.Module):
    """
    Feature adapter with temporal confidence-gated momentum
    """

    def __init__(self, num_channels, alpha_base=0.1):
        super().__init__()
        # Running statistics (track target distribution)
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones(num_channels))

        # Source statistics (preserve source distribution)
        self.register_buffer('source_mean', torch.zeros(num_channels))
        self.register_buffer('source_var', torch.ones(num_channels))

        # Base momentum
        self.alpha_base = alpha_base
        self.initialized = False

    def initialize_source_stats(self, features):
        """Initialize source statistics from first batch"""
        if not self.initialized:
            with torch.no_grad():
                self.source_mean.copy_(features.mean(dim=[0, 2, 3]))
                self.source_var.copy_(features.var(dim=[0, 2, 3]))
                self.running_mean.copy_(self.source_mean)
                self.running_var.copy_(self.source_var)
                self.initialized = True

    def forward(self, features, temporal_quality):
        """
        Args:
            features: [B, C, H, W]
            temporal_quality: scalar in [0, 1]
                - 1.0: Perfect temporal consistency
                - 0.0: No temporal consistency
        """
        # Initialize on first call
        self.initialize_source_stats(features)

        # Compute batch statistics
        batch_mean = features.mean(dim=[0, 2, 3])
        batch_var = features.var(dim=[0, 2, 3])

        # **Temporal quality gates the adaptation momentum**
        alpha_dynamic = self.alpha_base * temporal_quality

        # Update running statistics
        with torch.no_grad():
            self.running_mean = (1 - alpha_dynamic) * self.running_mean + alpha_dynamic * batch_mean
            self.running_var = (1 - alpha_dynamic) * self.running_var + alpha_dynamic * batch_var

        # Normalize using running stats
        features_norm = (features - self.running_mean.view(1, -1, 1, 1)) / \
                        torch.sqrt(self.running_var.view(1, -1, 1, 1) + 1e-5)

        # Re-scale to source distribution
        features_adapted = features_norm * torch.sqrt(self.source_var.view(1, -1, 1, 1) + 1e-5) + \
                           self.source_mean.view(1, -1, 1, 1)

        return features_adapted


class APTConfidenceGatedConfig(APTConfig):
    """Configuration for Confidence-Gated APT"""
    adaptation_name: str = "APT-ConfidenceGated"

    # Feature adaptation settings
    use_feature_adaptation: bool = True
    feature_alpha_base: float = 0.1

    # Quality computation method
    quality_method: str = "weighted"  # "iou", "ratio", "weighted"


class APTConfidenceGatedEngine(AdaptationEngine):
    """
    APT with Temporal Confidence-Gated Feature Adaptation

    Key idea: Temporal consistency quality controls feature adaptation momentum
    """
    model_name = "APT-ConfidenceGated"

    def __init__(self, base_model: BaseModel, config: APTConfidenceGatedConfig):
        super().__init__(base_model, config)
        self.config: APTConfidenceGatedConfig = config

        # Initialize temporal tracker
        self.tracker = TemporalTracker(
            max_age=config.max_age,
            min_hits=config.min_hits,
            iou_threshold=config.iou_threshold
        )

        # Frame buffer
        self.frame_buffer = deque(maxlen=config.buffer_size)

        # Statistics
        self.adaptation_steps = 0
        self.total_loss = 0.0
        self.temporal_quality_history = []

        # Feature adapters
        self.feature_adapters = nn.ModuleDict()
        if config.use_feature_adaptation:
            self._setup_feature_adapters()

    def _setup_feature_adapters(self):
        """Setup feature adapters for each backbone stage"""
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

    def compute_temporal_quality(self, current_boxes, predicted_boxes):
        """
        Compute temporal consistency quality

        Returns:
            quality: scalar in [0, 1]
        """
        if len(predicted_boxes) == 0 or len(current_boxes) == 0:
            return torch.tensor(0.0, device=self.device)

        # Convert to tensors
        if isinstance(current_boxes, np.ndarray):
            current_boxes = torch.from_numpy(current_boxes).float().to(self.device)
        if isinstance(predicted_boxes, np.ndarray):
            predicted_boxes = torch.from_numpy(predicted_boxes).float().to(self.device)

        # Compute IoU matrix
        from torchvision.ops import box_iou
        iou_matrix = box_iou(current_boxes, predicted_boxes)

        # Best matching for each predicted box
        if iou_matrix.numel() > 0:
            max_ious, max_indices = iou_matrix.max(dim=0)

            if self.config.quality_method == "iou":
                # Average IoU
                quality = max_ious.mean()

            elif self.config.quality_method == "ratio":
                # Matching ratio
                matched = (max_ious > self.config.iou_threshold).float()
                quality = matched.mean()

            elif self.config.quality_method == "weighted":
                # Weighted combination
                quality_iou = max_ious.mean()
                matched = (max_ious > self.config.iou_threshold).float()
                quality_ratio = matched.mean()

                if matched.sum() > 0:
                    quality_high = max_ious[max_ious > self.config.iou_threshold].mean()
                else:
                    quality_high = torch.tensor(0.0, device=self.device)

                quality = (quality_iou + quality_ratio + quality_high) / 3.0
        else:
            quality = torch.tensor(0.0, device=self.device)

        return quality.clamp(0.0, 1.0)

    @property
    def loss_function(self):
        if self._loss_function is None:
            if self.config.loss_type == "l1":
                self._loss_function = nn.L1Loss()
            elif self.config.loss_type == "l2":
                self._loss_function = nn.MSELoss()
            elif self.config.loss_type == "smooth_l1":
                self._loss_function = nn.SmoothL1Loss()
            else:
                raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        return self._loss_function

    @property
    def optimizer(self):
        if self._optimizer is None:
            from torch import optim
            if self.config.optim == "SGD":
                self._optimizer = optim.SGD(
                    self.online_parameters(),
                    lr=self.config.adapt_lr,
                    momentum=0.9
                )
            elif self.config.optim == "Adam":
                self._optimizer = optim.Adam(
                    self.online_parameters(),
                    lr=self.config.adapt_lr
                )
            elif self.config.optim == "AdamW":
                self._optimizer = optim.AdamW(
                    self.online_parameters(),
                    lr=self.config.adapt_lr
                )
        return self._optimizer

    def online_parameters(self):
        """Select parameters to adapt"""
        params = []

        if self.base_model.model_provider == ModelProvider.Detectron2:
            if self.config.update_bn:
                for module in self.base_model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        params.extend(module.parameters())
                    if "FrozenBatchNorm2d" in module.__class__.__name__:
                        if hasattr(module, 'weight'):
                            module.weight = nn.Parameter(module.weight)
                        if hasattr(module, 'bias'):
                            module.bias = nn.Parameter(module.bias)
                        params.extend(module.parameters())

        return params

    def extract_detections(self, outputs, conf_threshold=None):
        """Extract detections from model outputs"""
        if conf_threshold is None:
            conf_threshold = self.config.conf_threshold

        if self.base_model.model_provider == ModelProvider.Detectron2:
            instances = outputs['instances']
            scores = instances.scores.detach().cpu().numpy()
            boxes = instances.pred_boxes.tensor.detach().cpu().numpy()
            classes = instances.pred_classes.detach().cpu().numpy()

            mask = scores >= conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask]

            return boxes, scores, classes

        raise NotImplementedError(f"Unsupported model provider: {self.base_model.model_provider}")

    def compute_temporal_loss(self, current_boxes, predicted_boxes, current_classes, predicted_classes):
        """Compute temporal consistency loss"""
        if len(predicted_boxes) == 0 or len(current_boxes) == 0:
            return torch.tensor(0.0, device=self.device)

        if isinstance(predicted_boxes, np.ndarray):
            predicted_boxes = torch.from_numpy(predicted_boxes).float().to(self.device)
        if isinstance(predicted_classes, np.ndarray):
            predicted_classes = torch.from_numpy(predicted_classes).long().to(self.device)

        from torchvision.ops import box_iou

        loss = torch.tensor(0.0, device=self.device)
        n_matched = 0

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

        if n_matched > 0:
            loss = loss / n_matched

        return loss * self.config.loss_weight

    def forward(self, *args, **kwargs):
        """Forward pass with temporal-gated feature adaptation"""

        # Store original backbone forward
        if hasattr(self.base_model, 'backbone'):
            original_backbone_forward = self.base_model.backbone.forward
        else:
            original_backbone_forward = None

        # Temporal quality (initially 0.5)
        temporal_quality = torch.tensor(0.5, device=self.device)

        if self.adapting and self.config.use_feature_adaptation and original_backbone_forward:
            # Patch backbone forward to apply feature adapters
            def adapted_backbone_forward(x):
                features = original_backbone_forward(x)

                # Apply temporal-gated feature adaptation
                adapted_features = {}
                for stage_name, feat in features.items():
                    if stage_name in self.feature_adapters:
                        adapted_features[stage_name] = self.feature_adapters[stage_name](
                            feat, temporal_quality.item()
                        )
                    else:
                        adapted_features[stage_name] = feat

                return adapted_features

            self.base_model.backbone.forward = adapted_backbone_forward

        # Get base model predictions
        outputs = self.base_model(*args, **kwargs)

        # Restore original forward
        if original_backbone_forward:
            self.base_model.backbone.forward = original_backbone_forward

        # If not adapting, return directly
        if not self.adapting:
            return outputs

        # Extract detections
        if self.base_model.model_provider == ModelProvider.Detectron2:
            if isinstance(outputs, list):
                losses = []

                for output in outputs:
                    boxes, scores, classes = self.extract_detections(output)

                    # Get temporal predictions
                    if len(boxes) > 0:
                        predicted_boxes, predicted_classes, _ = self.tracker.update(boxes, classes)
                    else:
                        predicted_boxes, predicted_classes, _ = self.tracker.update(
                            np.empty((0, 4)), np.empty(0)
                        )

                    # Compute temporal quality
                    if len(predicted_boxes) > 0 and len(boxes) > 0:
                        temporal_quality = self.compute_temporal_quality(
                            output['instances'].pred_boxes.tensor,
                            predicted_boxes
                        )
                        self.temporal_quality_history.append(temporal_quality.item())

                    # Compute temporal loss
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

                # Backpropagation
                if len(losses) > 0:
                    total_loss = torch.stack(losses).mean()

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    self.adaptation_steps += 1
                    self.total_loss += total_loss.item()

        return outputs

    def reset(self):
        """Reset adaptation state"""
        self.tracker.reset()
        self.frame_buffer.clear()
        self.adaptation_steps = 0
        self.total_loss = 0.0
        self.temporal_quality_history = []
        super().reset()

    def get_adaptation_stats(self):
        """Get adaptation statistics"""
        avg_quality = np.mean(self.temporal_quality_history) if self.temporal_quality_history else 0.0

        return {
            'adaptation_steps': self.adaptation_steps,
            'avg_loss': self.total_loss / max(1, self.adaptation_steps),
            'total_loss': self.total_loss,
            'num_tracks': len(self.tracker.trackers),
            'avg_temporal_quality': avg_quality,
            'latest_quality': self.temporal_quality_history[-1] if self.temporal_quality_history else 0.0
        }

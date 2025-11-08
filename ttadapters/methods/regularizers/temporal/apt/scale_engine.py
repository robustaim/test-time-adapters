"""
APT with Multi-Scale Temporal-Feature Coupling

Method 4: Each feature scale has its own temporal quality and adaptation rate
"""
import torch
from torch import nn, optim
import numpy as np
from collections import deque, defaultdict

from ....base import AdaptationEngine
from .....models.base import BaseModel, ModelProvider

from .config import APTConfig
from .tracker import TemporalTracker


class MultiScaleFeatureAdapter(nn.Module):
    """
    Feature adapter with scale-specific adaptation
    """

    def __init__(self, num_channels, alpha_base=0.1, stage_name="res2"):
        super().__init__()
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones(num_channels))

        # Source statistics
        self.register_buffer('source_mean', torch.zeros(num_channels))
        self.register_buffer('source_var', torch.ones(num_channels))

        self.alpha_base = alpha_base
        self.stage_name = stage_name
        self.initialized = False

    def initialize_source_stats(self, features):
        """Initialize source statistics"""
        if not self.initialized:
            with torch.no_grad():
                self.source_mean.copy_(features.mean(dim=[0, 2, 3]))
                self.source_var.copy_(features.var(dim=[0, 2, 3]))
                self.running_mean.copy_(self.source_mean)
                self.running_var.copy_(self.source_var)
                self.initialized = True

    def forward(self, features, scale_quality):
        """
        Args:
            features: [B, C, H, W]
            scale_quality: scalar in [0, 1] - quality for this specific scale
        """
        self.initialize_source_stats(features)

        # Compute batch statistics
        batch_mean = features.mean(dim=[0, 2, 3])
        batch_var = features.var(dim=[0, 2, 3])

        # **Scale-specific adaptation momentum**
        alpha_dynamic = self.alpha_base * scale_quality

        # Update running statistics
        with torch.no_grad():
            self.running_mean = (1 - alpha_dynamic) * self.running_mean + alpha_dynamic * batch_mean
            self.running_var = (1 - alpha_dynamic) * self.running_var + alpha_dynamic * batch_var

        # Normalize
        features_norm = (
            features - self.running_mean.view(1, -1, 1, 1)
        ) / torch.sqrt(self.running_var.view(1, -1, 1, 1) + 1e-5)

        # Re-scale to source
        features_adapted = features_norm * torch.sqrt(
            self.source_var.view(1, -1, 1, 1) + 1e-5
        ) + self.source_mean.view(1, -1, 1, 1)

        return features_adapted


class APTMultiScaleConfig(APTConfig):
    """Configuration for Multi-Scale APT"""
    adaptation_name: str = "APT-MultiScale"

    # Feature adaptation settings
    use_feature_adaptation: bool = True
    feature_alpha_base: float = 0.1

    # Scale-specific settings
    scale_weights: dict = None  # {'res2': weight, 'res3': weight, ...}

    # Quality computation
    size_threshold_small: int = 1000   # Area < this = small object
    size_threshold_large: int = 10000  # Area > this = large object


class APTMultiScaleEngine(AdaptationEngine):
    """
    APT with Multi-Scale Temporal-Feature Coupling

    Key idea: Different feature scales have different temporal qualities
    """
    model_name = "APT-MultiScale"

    def __init__(self, base_model: BaseModel, config: APTMultiScaleConfig):
        super().__init__(base_model, config)
        self.config: APTMultiScaleConfig = config

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

        # Scale-specific quality tracking
        self.scale_quality_history = defaultdict(list)

        # Feature adapters
        self.feature_adapters = nn.ModuleDict()

        # Scale information (stride for each stage)
        self.scale_strides = {}

        if config.use_feature_adaptation:
            self._setup_feature_adapters()

    def _setup_feature_adapters(self):
        """Setup multi-scale feature adapters"""
        if self.base_model.model_provider == ModelProvider.Detectron2:
            if hasattr(self.base_model, 'backbone'):
                out_features = self.base_model.backbone._out_features
                out_channels = self.base_model.backbone._out_feature_channels

                # Get strides for each stage
                if hasattr(self.base_model.backbone, '_out_feature_strides'):
                    self.scale_strides = self.base_model.backbone._out_feature_strides
                else:
                    # Default strides for ResNet
                    self.scale_strides = {
                        'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32,
                        'stage2': 4, 'stage3': 8, 'stage4': 16, 'stage5': 32
                    }

                for stage_name in out_features:
                    num_channels = out_channels[stage_name]
                    adapter = MultiScaleFeatureAdapter(
                        num_channels=num_channels,
                        alpha_base=self.config.feature_alpha_base,
                        stage_name=stage_name
                    )
                    self.feature_adapters[stage_name] = adapter.to(self.device)

    def compute_box_area(self, boxes):
        """Compute area of boxes"""
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes).float()
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def compute_scale_specific_quality(self, current_boxes, predicted_boxes, stage_name):
        """
        Compute temporal quality specific to a feature scale

        Args:
            current_boxes: [N, 4]
            predicted_boxes: [M, 4]
            stage_name: 'res2', 'res3', etc.

        Returns:
            quality: scalar in [0, 1]
        """
        if len(current_boxes) == 0 or len(predicted_boxes) == 0:
            return torch.tensor(0.0, device=self.device)

        if isinstance(current_boxes, np.ndarray):
            current_boxes = torch.from_numpy(current_boxes).float().to(self.device)
        if isinstance(predicted_boxes, np.ndarray):
            predicted_boxes = torch.from_numpy(predicted_boxes).float().to(self.device)

        # Compute areas
        current_areas = self.compute_box_area(current_boxes)

        # Determine which objects are relevant for this scale
        stride = self.scale_strides.get(stage_name, 8)

        # Scale-specific weighting based on object size
        # High-res (res2, stride=4): small objects get high weight
        # Low-res (res5, stride=32): large objects get high weight

        if stride <= 4:  # High resolution
            # Small objects: weight increases as area decreases
            weights = torch.exp(-current_areas / self.config.size_threshold_small)
        elif stride >= 16:  # Low resolution
            # Large objects: weight increases as area increases
            weights = 1 - torch.exp(-current_areas / self.config.size_threshold_large)
        else:  # Medium resolution
            # Medium objects: balanced weighting
            weights = torch.ones_like(current_areas)

        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()

        # Compute IoU
        from torchvision.ops import box_iou
        iou_matrix = box_iou(current_boxes, predicted_boxes)

        if iou_matrix.numel() > 0:
            # For each current box, find best matching predicted box
            max_ious, max_indices = iou_matrix.max(dim=1)

            # Weighted quality (emphasize objects relevant to this scale)
            quality = (max_ious * weights).sum()
        else:
            quality = torch.tensor(0.0, device=self.device)

        return quality.clamp(0.0, 1.0)

    def compute_scale_qualities(self, current_boxes, predicted_boxes):
        """
        Compute temporal quality for each feature scale
        
        Returns:
            qualities_dict: {'res2': quality, 'res3': quality, ...}
        """
        qualities = {}

        for stage_name in self.feature_adapters.keys():
            quality = self.compute_scale_specific_quality(
                current_boxes, predicted_boxes, stage_name
            )
            qualities[stage_name] = quality.item()
            self.scale_quality_history[stage_name].append(quality.item())

        return qualities

    @property
    def loss_function(self):
        if self._loss_function is None:
            if self.config.loss_type == "smooth_l1":
                self._loss_function = nn.SmoothL1Loss()
            elif self.config.loss_type == "l1":
                self._loss_function = nn.L1Loss()
            elif self.config.loss_type == "l2":
                self._loss_function = nn.MSELoss()
        return self._loss_function

    @property
    def optimizer(self):
        if self._optimizer is None:
            from torch import optim
            params = self.online_parameters()
            if self.config.optim == "SGD":
                self._optimizer = optim.SGD(params, lr=self.config.adapt_lr, momentum=0.9)
            elif self.config.optim == "Adam":
                self._optimizer = optim.Adam(params, lr=self.config.adapt_lr)
            elif self.config.optim == "AdamW":
                self._optimizer = optim.AdamW(params, lr=self.config.adapt_lr)
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
            return boxes[mask], scores[mask], classes[mask]

        raise NotImplementedError(f"Unsupported model provider")

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
        """Forward with multi-scale feature adaptation"""

        # Store original backbone forward
        original_backbone_forward = None
        if hasattr(self.base_model, 'backbone'):
            original_backbone_forward = self.base_model.backbone.forward

        # Scale qualities (will be updated during detection processing)
        scale_qualities = {stage: 0.5 for stage in self.feature_adapters.keys()}

        # Patch backbone
        if self.adapting and self.config.use_feature_adaptation and original_backbone_forward:
            def adapted_backbone_forward(x):
                features = original_backbone_forward(x)

                # Apply scale-specific feature adaptation
                adapted_features = {}
                for stage_name, feat in features.items():
                    if stage_name in self.feature_adapters:
                        quality = scale_qualities.get(stage_name, 0.5)
                        adapted_features[stage_name] = self.feature_adapters[stage_name](
                            feat, quality
                        )
                    else:
                        adapted_features[stage_name] = feat

                return adapted_features

            self.base_model.backbone.forward = adapted_backbone_forward

        # Get predictions
        outputs = self.base_model(*args, **kwargs)

        # Restore
        if original_backbone_forward:
            self.base_model.backbone.forward = original_backbone_forward

        if not self.adapting:
            return outputs

        # Process detections
        if self.base_model.model_provider == ModelProvider.Detectron2:
            if isinstance(outputs, list):
                losses = []

                for output in outputs:
                    boxes, scores, classes = self.extract_detections(output)

                    # Get predictions from tracker
                    if len(boxes) > 0:
                        predicted_boxes, predicted_classes, _ = self.tracker.update(boxes, classes)
                    else:
                        predicted_boxes, predicted_classes, _ = self.tracker.update(
                            np.empty((0, 4)), np.empty(0)
                        )

                    # **Compute scale-specific qualities**
                    if len(boxes) > 0 and len(predicted_boxes) > 0:
                        scale_qualities = self.compute_scale_qualities(boxes, predicted_boxes)

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
        self.scale_quality_history = defaultdict(list)
        super().reset()

    def get_adaptation_stats(self):
        """Get adaptation statistics"""
        # Compute average quality for each scale
        avg_qualities = {}
        for stage, qualities in self.scale_quality_history.items():
            avg_qualities[f'{stage}_quality'] = np.mean(qualities) if qualities else 0.0

        return {
            'adaptation_steps': self.adaptation_steps,
            'avg_loss': self.total_loss / max(1, self.adaptation_steps),
            'total_loss': self.total_loss,
            'num_tracks': len(self.tracker.trackers),
            **avg_qualities
        }

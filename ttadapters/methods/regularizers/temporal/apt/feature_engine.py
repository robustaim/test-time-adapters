"""
APT with Learnable Feature Adaptation

Method 3: APT temporal loss directly trains learnable feature adapter parameters
"""
import torch
from torch import nn, optim
import numpy as np
from collections import deque

from ....base import AdaptationEngine
from .....models.base import BaseModel, ModelProvider

from .config import APTConfig
from .tracker import TemporalTracker


class LearnableFeatureAdapter(nn.Module):
    """
    Feature adapter with learnable scale/shift parameters

    Key: APT temporal loss trains these parameters via gradient descent
    """

    def __init__(self, num_channels, alpha_base=0.1):
        super().__init__()
        # Running statistics (gradient-free, NORM-style)
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones(num_channels))

        # Source statistics
        self.register_buffer('source_mean', torch.zeros(num_channels))
        self.register_buffer('source_var', torch.ones(num_channels))

        # **Learnable adaptation parameters (gradient-based)**
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.shift = nn.Parameter(torch.zeros(num_channels))

        self.alpha = alpha_base
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

    def forward(self, features):
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

        # 3. **Learnable transformation (gradient flows here!)**
        # This is where APT temporal loss can optimize
        features_adapted = features_norm * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)

        return features_adapted


class APTLearnableConfig(APTConfig):
    """Configuration for Learnable APT"""
    adaptation_name: str = "APT-Learnable"

    # Feature adaptation settings
    use_feature_adaptation: bool = True
    feature_alpha: float = 0.1
    feature_lr: float = None  # If None, use same as adapt_lr

    # Regularization
    use_l2_regularization: bool = True
    l2_weight: float = 0.01


class APTLearnableEngine(AdaptationEngine):
    """
    APT with Learnable Feature Adaptation
    
    Key idea: APT temporal loss trains feature adapter parameters directly
    """
    model_name = "APT-Learnable"

    def __init__(self, base_model: BaseModel, config: APTLearnableConfig):
        super().__init__(base_model, config)
        self.config: APTLearnableConfig = config

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
        self.total_reg_loss = 0.0

        # Feature adapters
        self.feature_adapters = nn.ModuleDict()

        if config.use_feature_adaptation:
            self._setup_feature_adapters()

    def _setup_feature_adapters(self):
        """Setup learnable feature adapters"""
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

            # Determine learning rate for feature adapters
            feature_lr = self.config.feature_lr if self.config.feature_lr else self.config.adapt_lr

            # Create parameter groups with different learning rates
            param_groups = []

            # Group 1: BN parameters
            bn_params = []
            for module in self.base_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    bn_params.extend(module.parameters())
                if "FrozenBatchNorm2d" in module.__class__.__name__:
                    if hasattr(module, 'weight'):
                        module.weight = nn.Parameter(module.weight)
                        bn_params.append(module.weight)
                    if hasattr(module, 'bias'):
                        module.bias = nn.Parameter(module.bias)
                        bn_params.append(module.bias)

            if bn_params:
                param_groups.append({'params': bn_params, 'lr': self.config.adapt_lr})

            # Group 2: Feature adapter parameters (learnable scale/shift)
            adapter_params = []
            for adapter in self.feature_adapters.values():
                if hasattr(adapter, 'scale'):
                    adapter_params.append(adapter.scale)
                if hasattr(adapter, 'shift'):
                    adapter_params.append(adapter.shift)

            if adapter_params:
                param_groups.append({'params': adapter_params, 'lr': feature_lr})

            # Create optimizer
            if self.config.optim == "SGD":
                self._optimizer = optim.SGD(param_groups, momentum=0.9)
            elif self.config.optim == "Adam":
                self._optimizer = optim.Adam(param_groups)
            elif self.config.optim == "AdamW":
                self._optimizer = optim.AdamW(param_groups)

        return self._optimizer

    def online_parameters(self):
        """All parameters that will be updated"""
        params = []

        # BN parameters
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

        # Feature adapter parameters
        for adapter in self.feature_adapters.values():
            if hasattr(adapter, 'scale'):
                params.append(adapter.scale)
            if hasattr(adapter, 'shift'):
                params.append(adapter.shift)

        return params

    def compute_regularization_loss(self):
        """
        L2 regularization to prevent adapters from deviating too much
        """
        if not self.config.use_l2_regularization:
            return torch.tensor(0.0, device=self.device)

        reg_loss = torch.tensor(0.0, device=self.device)

        for adapter in self.feature_adapters.values():
            # Penalize deviation from identity transformation
            # scale should be close to 1, shift should be close to 0
            if hasattr(adapter, 'scale'):
                reg_loss += ((adapter.scale - 1.0) ** 2).mean()
            if hasattr(adapter, 'shift'):
                reg_loss += (adapter.shift ** 2).mean()

        return self.config.l2_weight * reg_loss

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
        """Forward with learnable feature adaptation"""

        # Store original backbone forward
        original_backbone_forward = None
        if hasattr(self.base_model, 'backbone'):
            original_backbone_forward = self.base_model.backbone.forward

        # Patch backbone
        if self.adapting and self.config.use_feature_adaptation and original_backbone_forward:
            def adapted_backbone_forward(x):
                features = original_backbone_forward(x)

                # Apply learnable feature adapters
                # **Gradient will flow through scale/shift parameters!**
                adapted_features = {}
                for stage_name, feat in features.items():
                    if stage_name in self.feature_adapters:
                        adapted_features[stage_name] = self.feature_adapters[stage_name](feat)
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
                    temporal_loss = torch.stack(losses).mean()

                    # **Add regularization loss**
                    reg_loss = self.compute_regularization_loss()

                    # **Total loss = temporal + regularization**
                    total_loss = temporal_loss + reg_loss

                    self.optimizer.zero_grad()
                    total_loss.backward()  # **Gradient flows to scale/shift!**
                    self.optimizer.step()

                    self.adaptation_steps += 1
                    self.total_loss += temporal_loss.item()
                    self.total_reg_loss += reg_loss.item()

        return outputs

    def reset(self):
        """Reset adaptation state"""
        self.tracker.reset()
        self.frame_buffer.clear()
        self.adaptation_steps = 0
        self.total_loss = 0.0
        self.total_reg_loss = 0.0
        super().reset()

    def get_adaptation_stats(self):
        """Get adaptation statistics"""
        return {
            'adaptation_steps': self.adaptation_steps,
            'avg_temporal_loss': self.total_loss / max(1, self.adaptation_steps),
            'avg_reg_loss': self.total_reg_loss / max(1, self.adaptation_steps),
            'total_loss': self.total_loss + self.total_reg_loss,
            'num_tracks': len(self.tracker.trackers)
        }

"""
APT with Per-Object Feature Adaptation

Method 2: Only features from well-tracked objects contribute to statistics update
"""
import torch
from torch import nn, optim
import numpy as np
from collections import deque

from ....base import AdaptationEngine
from .....models.base import BaseModel, ModelProvider

from .config import APTConfig
from .tracker import TemporalTracker


class ObjectAwareFeatureAdapter(nn.Module):
    """
    Feature adapter that selectively uses well-tracked objects
    """

    def __init__(self, num_channels, alpha_base=0.1, quality_threshold=0.7):
        super().__init__()
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_channels))
        self.register_buffer('running_var', torch.ones(num_channels))

        # Source statistics
        self.register_buffer('source_mean', torch.zeros(num_channels))
        self.register_buffer('source_var', torch.ones(num_channels))

        self.alpha_base = alpha_base
        self.quality_threshold = quality_threshold
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

    def forward(self, features, object_masks=None, object_qualities=None):
        """
        Args:
            features: [B, C, H, W]
            object_masks: [B, H, W] - which pixel belongs to which object (0 = background)
            object_qualities: [N] - quality score for each object
        """
        self.initialize_source_stats(features)

        # If no object information, use global statistics (fallback to standard)
        if object_masks is None or object_qualities is None:
            batch_mean = features.mean(dim=[0, 2, 3])
            batch_var = features.var(dim=[0, 2, 3])
            alpha = self.alpha_base * 0.5  # Default moderate update

        else:
            # **Selective update: only use good objects**
            good_objects = object_qualities > self.quality_threshold

            if good_objects.sum() == 0:
                # No good objects → no update, return features as-is
                return features

            # Create mask for good objects
            good_mask = torch.zeros_like(object_masks, dtype=torch.float32)
            for obj_id in torch.where(good_objects)[0]:
                good_mask[object_masks == (obj_id + 1)] = 1.0

            # Masked features (only good object regions)
            masked_features = features * good_mask.unsqueeze(1)

            # Compute statistics from valid pixels only
            valid_pixels = good_mask.sum()

            if valid_pixels > 0:
                # Mean: sum over valid pixels / count
                batch_mean = masked_features.sum(dim=[0, 2, 3]) / (valid_pixels + 1e-5)

                # Variance: E[(x - μ)²] over valid pixels
                diff = (masked_features - batch_mean.view(1, -1, 1, 1)) * good_mask.unsqueeze(1)
                batch_var = (diff ** 2).sum(dim=[0, 2, 3]) / (valid_pixels + 1e-5)

                # Quality-weighted momentum
                avg_quality = object_qualities[good_objects].mean()
                alpha = self.alpha_base * avg_quality.item()
            else:
                # Fallback
                return features

        # Update running statistics
        with torch.no_grad():
            self.running_mean = (1 - alpha) * self.running_mean + alpha * batch_mean
            self.running_var = (1 - alpha) * self.running_var + alpha * batch_var

        # Normalize using running stats
        features_norm = (features - self.running_mean.view(1, -1, 1, 1)) / \
                        torch.sqrt(self.running_var.view(1, -1, 1, 1) + 1e-5)

        # Re-scale to source distribution
        features_adapted = features_norm * torch.sqrt(self.source_var.view(1, -1, 1, 1) + 1e-5) + self.source_mean.view(1, -1, 1, 1)

        return features_adapted


class APTPerObjectConfig(APTConfig):
    """Configuration for Per-Object APT"""
    adaptation_name: str = "APT-PerObject"

    # Feature adaptation settings
    use_feature_adaptation: bool = True
    feature_alpha_base: float = 0.1
    object_quality_threshold: float = 0.7

    # Mask generation settings
    mask_stages: list = None  # Which stages to apply (None = all)


class APTPerObjectEngine(AdaptationEngine):
    """
    APT with Per-Object Feature Adaptation

    Key idea: Only well-tracked objects contribute to feature statistics
    """
    model_name = "APT-PerObject"

    def __init__(self, base_model: BaseModel, config: APTPerObjectConfig):
        super().__init__(base_model, config)
        self.config: APTPerObjectConfig = config

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

        # Feature adapters
        self.feature_adapters = nn.ModuleDict()

        # Cache for object information
        self.current_object_info = None

        if config.use_feature_adaptation:
            self._setup_feature_adapters()

    def _setup_feature_adapters(self):
        """Setup per-object feature adapters"""
        if self.base_model.model_provider == ModelProvider.Detectron2:
            if hasattr(self.base_model, 'backbone'):
                out_features = self.base_model.backbone._out_features
                out_channels = self.base_model.backbone._out_feature_channels

                mask_stages = self.config.mask_stages or out_features

                for stage_name in out_features:
                    if stage_name in mask_stages:
                        num_channels = out_channels[stage_name]
                        adapter = ObjectAwareFeatureAdapter(
                            num_channels=num_channels,
                            alpha_base=self.config.feature_alpha_base,
                            quality_threshold=self.config.object_quality_threshold
                        )
                        self.feature_adapters[stage_name] = adapter.to(self.device)

    def create_object_masks(self, features_dict, detections, object_qualities, image_size):
        """
        Create object masks for each feature stage

        Args:
            features_dict: {'res2': [B,C,H,W], ...}
            detections: list of boxes [N, 4]
            object_qualities: [N] quality scores
            image_size: (H, W) original image size

        Returns:
            masks_dict: {'res2': [B,H,W], ...}
        """
        masks_dict = {}
        orig_h, orig_w = image_size

        for stage_name, features in features_dict.items():
            B, C, feat_h, feat_w = features.shape

            # Compute stride
            stride_h = orig_h / feat_h
            stride_w = orig_w / feat_w

            # Create mask
            object_mask = torch.zeros(B, feat_h, feat_w, device=features.device)

            # Project each box to feature map
            for obj_id, box in enumerate(detections):
                x1, y1, x2, y2 = box

                # Convert to feature map coordinates
                x1_feat = int(x1 / stride_w)
                y1_feat = int(y1 / stride_h)
                x2_feat = int(x2 / stride_w)
                y2_feat = int(y2 / stride_h)

                # Clamp to valid range
                x1_feat = max(0, min(x1_feat, feat_w - 1))
                y1_feat = max(0, min(y1_feat, feat_h - 1))
                x2_feat = max(0, min(x2_feat, feat_w - 1))
                y2_feat = max(0, min(y2_feat, feat_h - 1))

                # Assign object ID (1-indexed, 0 = background)
                object_mask[0, y1_feat:y2_feat+1, x1_feat:x2_feat+1] = obj_id + 1

            masks_dict[stage_name] = object_mask

        return masks_dict

    def compute_object_qualities(self, current_boxes, predicted_boxes, matched_indices):
        """
        Compute per-object quality scores based on matching

        Returns:
            qualities: [N] quality score for each current detection
        """
        if len(current_boxes) == 0:
            return torch.tensor([], device=self.device)

        if isinstance(current_boxes, np.ndarray):
            current_boxes = torch.from_numpy(current_boxes).float().to(self.device)
        if isinstance(predicted_boxes, np.ndarray):
            predicted_boxes = torch.from_numpy(predicted_boxes).float().to(self.device)

        qualities = torch.zeros(len(current_boxes), device=self.device)

        if len(predicted_boxes) == 0:
            return qualities

        # Compute IoU for each current box with all predicted boxes
        from torchvision.ops import box_iou
        iou_matrix = box_iou(current_boxes, predicted_boxes)

        # For each current detection, find best matching predicted box
        max_ious, max_indices = iou_matrix.max(dim=1)

        # Quality = IoU with best match
        qualities = max_ious

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
        """Forward with per-object feature adaptation"""

        # Get image size from batch (Detectron2 format)
        image_size = None
        if len(args) > 0 and isinstance(args[0], list):
            batch = args[0]
            if len(batch) > 0 and 'height' in batch[0]:
                image_size = (batch[0]['height'], batch[0]['width'])

        # Store original backbone forward
        original_backbone_forward = None
        if hasattr(self.base_model, 'backbone'):
            original_backbone_forward = self.base_model.backbone.forward

        # Patch backbone if adapting
        if self.adapting and self.config.use_feature_adaptation and original_backbone_forward:
            current_object_info = self.current_object_info

            def adapted_backbone_forward(x):
                features = original_backbone_forward(x)

                # Apply per-object feature adaptation
                if current_object_info is not None:
                    masks_dict = current_object_info['masks']
                    qualities = current_object_info['qualities']

                    adapted_features = {}
                    for stage_name, feat in features.items():
                        if stage_name in self.feature_adapters:
                            mask = masks_dict.get(stage_name, None)
                            adapted_features[stage_name] = self.feature_adapters[stage_name](
                                feat, mask, qualities
                            )
                        else:
                            adapted_features[stage_name] = feat
                    return adapted_features

                return features

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
                        predicted_boxes, predicted_classes, matched = self.tracker.update(boxes, classes)
                    else:
                        predicted_boxes, predicted_classes, matched = self.tracker.update(
                            np.empty((0, 4)), np.empty(0)
                        )

                    # Compute per-object qualities
                    if len(boxes) > 0 and len(predicted_boxes) > 0:
                        object_qualities = self.compute_object_qualities(
                            boxes, predicted_boxes, matched
                        )

                        # Create object masks for next forward pass
                        if image_size is not None and hasattr(self.base_model, 'backbone'):
                            # Get feature shapes (need to do a dummy forward)
                            with torch.no_grad():
                                dummy_input = output['instances'].image_size
                                # Use a simplified approach: store boxes and qualities
                                self.current_object_info = {
                                    'boxes': boxes,
                                    'qualities': object_qualities,
                                    'masks': {}  # Will be filled in adapted_backbone_forward
                                }

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
        self.current_object_info = None
        super().reset()

    def get_adaptation_stats(self):
        """Get adaptation statistics"""
        return {
            'adaptation_steps': self.adaptation_steps,
            'avg_loss': self.total_loss / max(1, self.adaptation_steps),
            'total_loss': self.total_loss,
            'num_tracks': len(self.tracker.trackers)
        }

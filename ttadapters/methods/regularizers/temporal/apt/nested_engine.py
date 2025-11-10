"""
Nested APT (Basic): Multi-timescale Updates without Feature Adaptation

This is a minimal implementation that adds ONLY multi-timescale updates
to basic APT, without any feature adaptation layer.

Purpose: Ablation study to isolate the effect of Nested Learning's
multi-timescale updates from feature adaptation.

Key differences from NestedAPT:
- No learnable feature adapters
- No feature-level adaptation
- Only multi-timescale updates on existing parameters (BN, FPN, Backbone)
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


class BasicNestedAPTConfig(APTConfig):
    """Configuration for Basic Nested APT"""
    adaptation_name: str = "BasicNestedAPT"

    # Nested Learning: Multi-timescale update frequencies
    fast_update_freq: int = 1      # BatchNorm updates every frame
    medium_update_freq: int = 5    # FPN updates every 5 frames
    slow_update_freq: int = 20     # Backbone updates every 20 frames

    # Nested Learning: Learning rates for each timescale
    fast_lr: float = 1e-5          # BatchNorm parameters
    medium_lr: float = 1e-6        # FPN parameters
    slow_lr: float = 1e-7          # Backbone parameters

    # Update strategy
    update_bn: bool = True
    update_fpn_last_layer: bool = False
    update_backbone_last_layer: bool = False


class BasicNestedAPTEngine(AdaptationEngine):
    """
    Basic Nested APT: Multi-timescale Updates Only

    This engine implements ONLY multi-timescale parameter updates
    without any feature adaptation layers.

    Use for:
    - Ablation study: isolate multi-timescale effect
    - Baseline comparison
    - Understanding pure Nested Learning contribution

    Parameter hierarchy:
    - Fast (every frame): BatchNorm parameters
    - Medium (every 5 frames): FPN parameters
    - Slow (every 20 frames): Backbone parameters
    """
    model_name = "BasicNestedAPT"

    def __init__(self, base_model: BaseModel, config: BasicNestedAPTConfig):
        super().__init__(base_model, config)
        self.config: BasicNestedAPTConfig = config

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

        # Nested Learning: Track updates per level
        self.updates_per_level = {
            'fast': 0,
            'medium': 0,
            'slow': 0
        }

    def _get_bn_params(self) -> List[nn.Parameter]:
        """Get BatchNorm parameters (fast update)"""
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
        """Get FPN last layer parameters (medium update)"""
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

            # Fast: BatchNorm parameters (update every frame)
            if self.config.update_bn:
                bn_params = self._get_bn_params()
                if bn_params:
                    param_groups.append({
                        'params': bn_params,
                        'lr': self.config.fast_lr,
                        'name': 'fast',
                        'update_freq': self.config.fast_update_freq
                    })

            # Medium: FPN parameters (update every 5 frames)
            fpn_params = self._get_fpn_params()
            if fpn_params:
                param_groups.append({
                    'params': fpn_params,
                    'lr': self.config.medium_lr,
                    'name': 'medium',
                    'update_freq': self.config.medium_update_freq
                })

            # Slow: Backbone parameters (update every 20 frames)
            backbone_params = self._get_backbone_params()
            if backbone_params:
                param_groups.append({
                    'params': backbone_params,
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
        """
        Forward pass with multi-timescale adaptation

        No feature adaptation - just standard forward pass
        with selective parameter updates.
        """
        # Increment frame counter
        self.frame_count += 1

        # Get model predictions (standard forward)
        outputs = self.base_model(*args, **kwargs)

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

                    # Compute gradients
                    self.optimizer.zero_grad()
                    temporal_loss.backward()

                    # Selective parameter updates based on frame count
                    # This is the ONLY contribution: multi-timescale updates
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

                    # Apply optimizer step
                    self.optimizer.step()

                    # Update statistics
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
        self.updates_per_level = {k: 0 for k in self.updates_per_level}

        super().reset()

    def get_adaptation_stats(self) -> Dict:
        """Get detailed adaptation statistics"""
        return {
            'adaptation_steps': self.adaptation_steps,
            'frame_count': self.frame_count,
            'avg_temporal_loss': self.total_loss / max(1, self.adaptation_steps),
            'total_loss': self.total_loss,
            'num_tracks': len(self.tracker.trackers),
            'updates_fast': self.updates_per_level.get('fast', 0),
            'updates_medium': self.updates_per_level.get('medium', 0),
            'updates_slow': self.updates_per_level.get('slow', 0),
        }

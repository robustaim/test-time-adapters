"""
APT: Adaptive Plugin for Test-time Adaptation via Temporal Consistency

This module implements test-time adaptation for object detection models
using temporal consistency between consecutive frames via Kalman filter tracking.
"""
import torch
from torch import nn, optim
import numpy as np
from collections import deque

from ...base import AdaptationEngine
from ....models.base import BaseModel, ModelProvider

from .apt_config import APTConfig
from .kalman_tracker import TemporalTracker


class APTPlugin(AdaptationEngine):
    """APT: Adaptive Plugin using Temporal Consistency
    
    Uses Kalman filter tracking to predict temporally consistent bounding boxes
    and adapts the model using the consistency loss between predictions and
    tracked boxes.
    """
    model_name = "APT"

    def __init__(self, base_model: BaseModel, config: APTConfig):
        super().__init__(base_model, config)
        self.config: APTConfig = config

        # Initialize temporal tracker
        self.tracker = TemporalTracker(
            max_age=config.max_age,
            min_hits=config.min_hits,
            iou_threshold=config.iou_threshold
        )

        # Frame buffer for temporal window
        self.frame_buffer = deque(maxlen=config.buffer_size)

        # Previous frame info
        self.prev_detections = None
        self.prev_classes = None

        # Statistics
        self.adaptation_steps = 0
        self.total_loss = 0.0

    @property
    def loss_function(self):
        """Return appropriate loss function based on config."""
        if self._loss_function is None:
            if self.config.loss_type == "l1":
                self._loss_function = nn.L1Loss()
            elif self.config.loss_type == "l2":
                self._loss_function = nn.MSELoss()
            elif self.config.loss_type == "smooth_l1":
                self._loss_function = nn.SmoothL1Loss()
            elif self.config.loss_type == "giou":
                self._loss_function = self._giou_loss
            else:
                raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        return self._loss_function

    def _giou_loss(self, pred_boxes, target_boxes):
        """Generalized IoU loss."""
        from torchvision.ops import generalized_box_iou_loss
        return generalized_box_iou_loss(pred_boxes, target_boxes).mean()

    @property
    def optimizer(self):
        if self._optimizer is None:
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
            else:
                raise ValueError(f"Unknown optimizer: {self.config.optim}")
        return self._optimizer

    def online_parameters(self):
        """Select parameters to adapt based on config."""
        params = []

        # Detectron2 models
        if self.base_model.model_provider == ModelProvider.Detectron2:
            if self.config.update_backbone:
                # FPN and backbone
                if hasattr(self.base_model, 'backbone'):
                    params.extend(self.base_model.backbone.parameters())

            if self.config.update_head:
                # RoI heads
                if hasattr(self.base_model, 'roi_heads'):
                    params.extend(self.base_model.roi_heads.parameters())

            if self.config.update_bn:
                # Batch norm
                for module in self.base_model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        params.extend(module.parameters())
                    if "FrozenBatchNorm2d" in module.__class__.__name__:
                        if hasattr(module, 'weight'):
                            module.weight = nn.Parameter(module.weight)
                        if hasattr(module, 'bias'):
                            module.bias = nn.Parameter(module.bias)
                        params.extend(module.parameters())
        else:
            # For other models, update all parameters
            params = self.base_model.parameters()

        return params

    def extract_detections(self, outputs, conf_threshold=None):
        """Extract detections from model outputs.
        
        Returns:
            boxes: (N, 4) numpy array in [x1, y1, x2, y2] format
            scores: (N,) numpy array of confidence scores
            classes: (N,) numpy array of class labels
        """
        if conf_threshold is None:
            conf_threshold = self.config.conf_threshold

        if self.base_model.model_provider == ModelProvider.Detectron2:
            # Detectron2 format
            instances = outputs['instances']
            scores = instances.scores.detach().cpu().numpy()
            boxes = instances.pred_boxes.tensor.detach().cpu().numpy()
            classes = instances.pred_classes.detach().cpu().numpy()

            # Filter by confidence
            mask = scores >= conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            classes = classes[mask]

            return boxes, scores, classes

        else:
            raise NotImplementedError(f"Unsupported model provider: {self.base_model.model_provider}")

    def compute_temporal_loss(self, current_boxes, predicted_boxes, current_classes, predicted_classes):
        """Compute temporal consistency loss.

        Args:
            current_boxes: (N, 4) tensor of current frame predictions
            predicted_boxes: (M, 4) tensor of tracked predictions (pseudo-labels)
            current_classes: (N,) tensor of current frame class labels
            predicted_classes: (M,) tensor of tracked class labels

        Returns:
            loss: scalar tensor
        """
        if len(predicted_boxes) == 0 or len(current_boxes) == 0:
            return torch.tensor(0.0, device=self.device)

        # Convert to tensors if needed
        if isinstance(predicted_boxes, np.ndarray):
            predicted_boxes = torch.from_numpy(predicted_boxes).float().to(self.device)
        if isinstance(predicted_classes, np.ndarray):
            predicted_classes = torch.from_numpy(predicted_classes).long().to(self.device)

        # Match current detections with tracked predictions by class and IOU
        from torchvision.ops import box_iou

        loss = torch.tensor(0.0, device=self.device)
        n_matched = 0

        for cls_id in torch.unique(predicted_classes):
            # Get boxes for this class
            pred_mask = predicted_classes == cls_id
            curr_mask = current_classes == cls_id

            if not curr_mask.any() or not pred_mask.any():
                continue

            pred_cls_boxes = predicted_boxes[pred_mask]
            curr_cls_boxes = current_boxes[curr_mask]

            # Compute IOU matrix
            iou_matrix = box_iou(curr_cls_boxes, pred_cls_boxes)

            # For each predicted box, find best matching current box
            if iou_matrix.numel() > 0:
                max_ious, max_indices = iou_matrix.max(dim=0)

                # Only use matches with sufficient IOU
                valid_matches = max_ious > self.config.iou_threshold

                if valid_matches.any():
                    matched_curr = curr_cls_boxes[max_indices[valid_matches]]
                    matched_pred = pred_cls_boxes[valid_matches]

                    # Compute loss for matched boxes
                    if self.config.loss_type == "giou":
                        loss += self.loss_function(matched_curr, matched_pred)
                    else:
                        loss += self.loss_function(matched_curr, matched_pred)

                    n_matched += valid_matches.sum().item()

        # Normalize by number of matches
        if n_matched > 0:
            loss = loss / n_matched

        return loss * self.config.loss_weight

    def forward(self, *args, **kwargs):
        """Forward pass with optional adaptation."""
        # Get base model predictions
        outputs = self.base_model(*args, **kwargs)

        # If not in adaptation mode, return predictions directly
        if not self.adapting:
            return outputs

        # Extract detections from current frame
        if self.base_model.model_provider == ModelProvider.Detectron2:
            # For Detectron2, we get list of outputs for each image
            if isinstance(outputs, list):
                losses = []

                for output in outputs:
                    boxes, scores, classes = self.extract_detections(output)

                    # Get temporal predictions from tracker
                    if len(boxes) > 0:
                        predicted_boxes, predicted_classes, _ = self.tracker.update(boxes, classes)
                    else:
                        predicted_boxes, predicted_classes, _ = self.tracker.update(
                            np.empty((0, 4)), np.empty(0)
                        )

                    # Compute temporal consistency loss if we have predictions
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

                # Backpropagate if we have valid losses
                if len(losses) > 0:
                    total_loss = torch.stack(losses).mean()

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()

                    self.adaptation_steps += 1
                    self.total_loss += total_loss.item()

        return outputs

    def reset(self):
        """Reset adaptation state including tracker."""
        self.tracker.reset()
        self.frame_buffer.clear()
        self.prev_detections = None
        self.prev_classes = None
        self.adaptation_steps = 0
        self.total_loss = 0.0

        # Reset model to base state
        super().reset()

    def get_adaptation_stats(self):
        """Get adaptation statistics."""
        return {
            'adaptation_steps': self.adaptation_steps,
            'avg_loss': self.total_loss / max(1, self.adaptation_steps),
            'total_loss': self.total_loss,
            'num_tracks': len(self.tracker.trackers)
        }

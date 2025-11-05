"""
APT: Adaptive Plugin for Test-Time Adaptation via Temporal Consistency

This module implements test-time adaptation for object detection models
using temporal consistency between consecutive frames via Kalman filter tracking.

Key differences from standard approaches:
- Uses motion predictions (NOT pseudo-labels) as self-supervised signals
- Implements proper temporal delay (predict frame t, compare with frame t detections)
- Confidence-weighted loss for stability
"""
import torch
from torch import nn, optim
import numpy as np
from collections import deque

from ....base import AdaptationEngine
from .....models.base import BaseModel, ModelProvider

from .config import APTConfig
from .tracker import TemporalTracker


class APTEngine(AdaptationEngine):
    """
    APT: Adaptive Plugin using Temporal Consistency

    Uses Kalman filter tracking to predict temporally consistent bounding boxes
    and adapts the model using the consistency loss between current frame detections
    and motion-based predictions from previous frame.
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

        # Previous frame info for temporal delay
        self.prev_detections = None
        self.prev_classes = None
        self.prev_scores = None

        # Current frame motion predictions (from tracker)
        self.current_motion_predictions = None
        self.current_predicted_classes = None

        # Statistics
        self.adaptation_steps = 0
        self.total_loss = 0.0
        self.loss_scale_ema = 1.0  # For loss normalization

        # Domain change detection (based on loss, not mAP)
        self.loss_history_ema = None  # EMA of recent losses
        self.domain_change_detected_count = 0

    @property
    def loss_function(self):
        """Return appropriate loss function based on config."""
        if self._loss_function is None:
            if self.config.loss_type == "l1":
                self._loss_function = nn.L1Loss(reduction='none')
            elif self.config.loss_type == "l2":
                self._loss_function = nn.MSELoss(reduction='none')
            elif self.config.loss_type == "smooth_l1":
                self._loss_function = nn.SmoothL1Loss(reduction='none')
            elif self.config.loss_type == "giou":
                self._loss_function = self._giou_loss
            else:
                raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        return self._loss_function

    def _giou_loss(self, pred_boxes, target_boxes, reduction='none'):
        """Generalized IoU loss with optional reduction."""
        from torchvision.ops import generalized_box_iou_loss
        loss = generalized_box_iou_loss(pred_boxes, target_boxes, reduction='none')
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    @property
    def optimizer(self):
        if self._optimizer is None:
            # Group parameters by learning rate
            param_groups = []

            if self.config.optim == "SGD":
                optimizer_class = optim.SGD
                optimizer_kwargs = {
                    "momentum": self.config.momentum,
                    "weight_decay": self.config.weight_decay
                }
            elif self.config.optim == "Adam":
                optimizer_class = optim.Adam
                optimizer_kwargs = {
                    "weight_decay": self.config.weight_decay
                }
            elif self.config.optim == "AdamW":
                optimizer_class = optim.AdamW
                optimizer_kwargs = {
                    "weight_decay": self.config.weight_decay
                }
            else:
                raise ValueError(f"Unknown optimizer: {self.config.optim}")

            # Collect parameters with different learning rates
            for name, param_list, lr in self._get_parameter_groups():
                if len(param_list) > 0:
                    param_groups.append({
                        'params': param_list,
                        'lr': lr,
                        'name': name
                    })

            if len(param_groups) == 0:
                raise ValueError("No parameters to optimize!")

            self._optimizer = optimizer_class(param_groups, **optimizer_kwargs)

        return self._optimizer

    def _get_parameter_groups(self):
        """Get parameter groups with different learning rates."""
        # Detectron2 models
        if self.base_model.model_provider == ModelProvider.Detectron2:
            bn_params = []
            backbone_params = []
            head_params = []
            fpn_last_params = []
            box_reg_last_params = []

            for name, module in self.base_model.named_modules():
                # Batch Normalization
                if self.config.update_bn:
                    if isinstance(module, nn.BatchNorm2d):
                        bn_params.extend(module.parameters())
                    elif "FrozenBatchNorm2d" in module.__class__.__name__:
                        # Unfreeze FrozenBatchNorm2d parameters
                        if hasattr(module, 'weight') and not isinstance(module.weight, nn.Parameter):
                            module.weight = nn.Parameter(module.weight.clone())
                        if hasattr(module, 'bias') and not isinstance(module.bias, nn.Parameter):
                            module.bias = nn.Parameter(module.bias.clone())
                        if hasattr(module, 'weight'):
                            bn_params.append(module.weight)
                        if hasattr(module, 'bias'):
                            bn_params.append(module.bias)

                # FPN last layer
                if self.config.update_fpn_last_layer:
                    if 'fpn_output' in name or 'fpn_lateral' in name:
                        # Check if it's the last layer
                        if any(x in name for x in ['fpn_output5', 'fpn_lateral5', 'top_block']):
                            fpn_last_params.extend(module.parameters())

                # Box regressor last layer
                if self.config.update_box_regressor_last_layer:
                    if 'box_predictor.bbox_pred' in name:
                        # Get the last linear layer
                        if isinstance(module, nn.Linear):
                            box_reg_last_params.extend(module.parameters())

                # Backbone
                if self.config.update_backbone and 'backbone' in name:
                    if not any(isinstance(module, t) for t in [nn.BatchNorm2d]):
                        backbone_params.extend([p for p in module.parameters() if p not in bn_params])

                # RoI Heads
                if self.config.update_head and 'roi_heads' in name:
                    if not any(isinstance(module, t) for t in [nn.BatchNorm2d]):
                        head_params.extend([
                            p for p in module.parameters() if p not in bn_params and p not in box_reg_last_params
                        ])

            # Remove duplicates while preserving order
            bn_params = list(dict.fromkeys(bn_params))
            fpn_last_params = list(dict.fromkeys(fpn_last_params))
            box_reg_last_params = list(dict.fromkeys(box_reg_last_params))
            backbone_params = list(dict.fromkeys(backbone_params))
            head_params = list(dict.fromkeys(head_params))

            param_groups = [
                ("BatchNorm", bn_params, self.config.adapt_lr),
                ("FPN_last", fpn_last_params, self.config.head_lr),
                ("BoxReg_last", box_reg_last_params, self.config.head_lr),
                ("Backbone", backbone_params, self.config.backbone_lr),
                ("Head", head_params, self.config.head_lr),
            ]
        else:
            # For other models, update all parameters
            param_groups = [("All", list(self.base_model.parameters()), self.config.adapt_lr)]

        return param_groups

    def online_parameters(self):
        """Select parameters to adapt based on config."""
        params = []
        for _, param_list, _ in self._get_parameter_groups():
            params.extend(param_list)
        return params

    def detect_domain_change(self, current_loss_value: float) -> bool:
        """Detect domain change based on loss spike (NOT mAP).

        Args:
            current_loss_value: Current loss value (scalar)

        Returns:
            True if domain change detected, False otherwise
        """
        if not self.config.enable_domain_change_reset:
            return False

        # Initialize loss history on first call
        if self.loss_history_ema is None:
            self.loss_history_ema = current_loss_value
            return False

        # Calculate relative loss change
        if self.loss_history_ema > 1e-6:  # Avoid division by zero
            loss_change_ratio = abs(current_loss_value - self.loss_history_ema) / self.loss_history_ema

            # Detect spike
            if loss_change_ratio > self.config.domain_change_loss_threshold:
                return True

        # Update EMA
        self.loss_history_ema = 0.9 * self.loss_history_ema + 0.1 * current_loss_value
        return False

    def reset_optimizer_state(self):
        """Reset optimizer momentum/state but keep model parameters.

        This is useful when domain changes - we want to forget the momentum
        from previous domain but keep the adapted parameters.
        """
        # Force optimizer recreation
        self._optimizer = None
        # Next call to self.optimizer will create a new one
        self.domain_change_detected_count += 1
        print(f"[APT] Domain change detected! Resetting optimizer (count: {self.domain_change_detected_count})")

    def extract_detections(self, outputs, conf_threshold=None):
        """Extract detections from model outputs.

        Returns:
            boxes: (N, 4) numpy array in [x1, y1, x2, y2] format
            scores: (N,) numpy array of confidence scores
            classes: (N,) numpy array of class labels
        """
        if conf_threshold is None:
            conf_threshold = self.config.min_confidence_for_update

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

    def compute_temporal_consistency_loss(
            self, current_boxes, current_scores, current_classes,
            motion_predictions, predicted_classes
    ):
        """Compute temporal consistency loss with confidence weighting.

        Args:
            current_boxes: (N, 4) tensor of current frame detections
            current_scores: (N,) tensor of detection confidence scores
            current_classes: (N,) tensor of current frame class labels
            motion_predictions: (M, 4) tensor of motion-based predictions (from tracker)
            predicted_classes: (M,) tensor of predicted class labels

        Returns:
            loss: scalar tensor
            n_matched: number of matched boxes
        """
        if len(motion_predictions) == 0 or len(current_boxes) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        # Convert to tensors if needed
        if isinstance(motion_predictions, np.ndarray):
            motion_predictions = torch.from_numpy(motion_predictions).float().to(self.device)
        if isinstance(predicted_classes, np.ndarray):
            predicted_classes = torch.from_numpy(predicted_classes).long().to(self.device)

        # Match current detections with motion predictions by class and IOU
        from torchvision.ops import box_iou

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        n_matched = 0

        for cls_id in torch.unique(predicted_classes):
            # Get boxes for this class
            pred_mask = predicted_classes == cls_id
            curr_mask = current_classes == cls_id

            if not curr_mask.any() or not pred_mask.any():
                continue

            pred_cls_boxes = motion_predictions[pred_mask]
            curr_cls_boxes = current_boxes[curr_mask]
            curr_cls_scores = current_scores[curr_mask]

            # Compute IOU matrix
            iou_matrix = box_iou(curr_cls_boxes, pred_cls_boxes)

            # For each motion prediction, find best matching current detection
            if iou_matrix.numel() > 0:
                max_ious, max_indices = iou_matrix.max(dim=0)

                # Only use matches with sufficient IOU
                valid_matches = max_ious > self.config.iou_threshold

                if valid_matches.any():
                    matched_curr = curr_cls_boxes[max_indices[valid_matches]]
                    matched_pred = pred_cls_boxes[valid_matches]
                    matched_scores = curr_cls_scores[max_indices[valid_matches]]

                    # Compute loss for matched boxes
                    if self.config.loss_type == "giou":
                        box_losses = self._giou_loss(matched_curr, matched_pred, reduction='none')
                    else:
                        # For L1/L2/SmoothL1, compute per-box loss
                        box_losses = self.loss_function(matched_curr, matched_pred).mean(dim=1)

                    # Apply confidence weighting if enabled
                    if self.config.use_confidence_weighting:
                        # Higher confidence = higher weight
                        weights = matched_scores
                        weighted_loss = (box_losses * weights).sum()
                        total_loss = total_loss + weighted_loss
                    else:
                        total_loss = total_loss + box_losses.sum()

                    n_matched += valid_matches.sum().item()

        return total_loss, n_matched

    def forward(self, *args, **kwargs):
        """Forward pass with optional adaptation using temporal delay."""
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
                    # Extract current frame detections
                    boxes, scores, classes = self.extract_detections(
                        output,
                        conf_threshold=self.config.min_confidence_for_update
                    )

                    # === TEMPORAL DELAY: Use predictions from PREVIOUS frame ===
                    # If we have motion predictions from previous frame
                    if self.current_motion_predictions is not None and len(self.current_motion_predictions) > 0:
                        # Get high-confidence detections for loss computation
                        current_boxes = output['instances'].pred_boxes.tensor
                        current_classes = output['instances'].pred_classes
                        current_scores = output['instances'].scores

                        conf_mask = current_scores >= self.config.conf_threshold

                        if conf_mask.any():
                            # Compute temporal consistency loss
                            loss, n_matched = self.compute_temporal_consistency_loss(
                                current_boxes[conf_mask],
                                current_scores[conf_mask],
                                current_classes[conf_mask],
                                self.current_motion_predictions,
                                self.current_predicted_classes
                            )

                            if n_matched > 0 and loss.item() > 0:
                                # Normalize loss with EMA-based scaling
                                normalized_loss = loss / (1.0 + n_matched)

                                # Update EMA of loss scale
                                with torch.no_grad():
                                    current_scale = loss.item() / max(1.0, n_matched)
                                    self.loss_scale_ema = (
                                            self.config.loss_ema_decay * self.loss_scale_ema +
                                            (1 - self.config.loss_ema_decay) * current_scale
                                    )

                                # Apply loss weight
                                final_loss = normalized_loss * self.config.loss_weight
                                losses.append(final_loss)

                    # === Update tracker with CURRENT frame detections ===
                    # Use lower threshold for tracker update
                    if len(boxes) > 0:
                        motion_predictions, predicted_classes, _ = self.tracker.update(
                            boxes, classes
                        )
                    else:
                        motion_predictions, predicted_classes, _ = self.tracker.update(
                            np.empty((0, 4)), np.empty(0)
                        )

                    # Store motion predictions for NEXT frame
                    self.current_motion_predictions = motion_predictions
                    self.current_predicted_classes = predicted_classes

                # Backpropagate if we have valid losses
                if len(losses) > 0:
                    total_loss = torch.stack(losses).mean()

                    # Check for domain change based on loss spike
                    if self.detect_domain_change(total_loss.item()):
                        self.reset_optimizer_state()

                    self.optimizer.zero_grad()
                    total_loss.backward()

                    # Optional: Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.online_parameters(), max_norm=1.0)

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
        self.prev_scores = None
        self.current_motion_predictions = None
        self.current_predicted_classes = None
        self.adaptation_steps = 0
        self.total_loss = 0.0
        self.loss_scale_ema = 1.0
        self.loss_history_ema = None
        self.domain_change_detected_count = 0

        # Reset model to base state
        super().reset()

    def get_adaptation_stats(self):
        """Get adaptation statistics."""
        return {
            'adaptation_steps': self.adaptation_steps,
            'avg_loss': self.total_loss / max(1, self.adaptation_steps),
            'total_loss': self.total_loss,
            'num_tracks': len(self.tracker.trackers),
            'loss_scale_ema': self.loss_scale_ema,
            'domain_changes': self.domain_change_detected_count,
            'loss_ema': self.loss_history_ema if self.loss_history_ema else 0.0
        }

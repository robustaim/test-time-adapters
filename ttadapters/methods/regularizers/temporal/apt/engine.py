"""
APT: Enhanced with Detection Quality Awareness

Key improvements:
1. Detection-quality-aware domain change detection
2. Adaptive thresholds based on detection quality
3. Separate handling for high/low detection scenarios
4. Confidence-based loss normalization
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
    
    Enhanced with detection quality awareness.
    """
    model_name = "APT_Enhanced"

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
        self.loss_scale_ema = 1.0

        # Idea A: Loss history for spike detection
        self.loss_history = deque(maxlen=config.loss_history_size) if config.enable_loss_history else None
        self.domain_change_detected_count = 0
        self.skipped_updates_count = 0

        # NEW: Detection quality tracking
        self.detection_count_history = deque(maxlen=50)
        self.avg_detection_count = 0.0
        self.detection_quality_mode = "unknown"  # "high", "medium", "low"

        # Idea D: Track BN modules for statistics update
        self.bn_modules = []
        self._setup_bn_modules()

    def _setup_bn_modules(self):
        """Idea D: Setup BN modules for statistics tracking and update."""
        if not self.config.update_bn_stats:
            return
            
        for module in self.base_model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self.bn_modules.append(module)
                module.track_running_stats = True
                if hasattr(module, 'running_mean') and module.running_mean is not None:
                    module.running_mean.requires_grad = False
                if hasattr(module, 'running_var') and module.running_var is not None:
                    module.running_var.requires_grad = False

    def update_detection_quality_mode(self, num_detections: int):
        """Track detection count and determine quality mode."""
        self.detection_count_history.append(num_detections)
        
        if len(self.detection_count_history) >= 20:
            self.avg_detection_count = np.mean(list(self.detection_count_history))
            
            # Determine quality mode based on average detection count
            if self.avg_detection_count > 20:
                self.detection_quality_mode = "high"
            elif self.avg_detection_count > 10:
                self.detection_quality_mode = "medium"
            else:
                self.detection_quality_mode = "low"

    def get_adaptive_threshold(self) -> float:
        """Get domain change threshold adapted to detection quality."""
        base_threshold = self.config.domain_change_loss_spike_threshold
        
        # Adjust threshold based on detection quality
        if self.detection_quality_mode == "high":
            # High quality: MORE CONSERVATIVE (higher threshold)
            # More detections = naturally higher loss
            return base_threshold * 1.5
        elif self.detection_quality_mode == "low":
            # Low quality: MORE SENSITIVE (lower threshold)
            # Few detections = need to catch actual changes
            return base_threshold * 0.7
        else:
            # Medium quality: use base threshold
            return base_threshold

    def normalize_loss_by_detection_quality(self, loss: torch.Tensor, n_matched: int) -> torch.Tensor:
        """Normalize loss considering detection quality."""
        # Base normalization
        normalized = loss / (1.0 + n_matched)
        
        # Additional scaling based on detection quality
        if self.detection_quality_mode == "high":
            # High detection count: scale down to prevent over-reaction
            scale_factor = 0.7
        elif self.detection_quality_mode == "low":
            # Low detection count: scale up to maintain learning signal
            scale_factor = 1.5
        else:
            scale_factor = 1.0
        
        return normalized * scale_factor

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
        """Idea B: Get parameter groups including Conv before BN and MLP after norm."""
        if self.base_model.model_provider == ModelProvider.Detectron2:
            bn_params = []
            conv_before_bn_params = []
            mlp_after_norm_params = []
            backbone_params = []
            head_params = []
            fpn_last_params = []
            box_reg_last_params = []

            module_list = list(self.base_model.named_modules())
            
            for idx, (name, module) in enumerate(module_list):
                if self.config.update_bn:
                    if isinstance(module, nn.BatchNorm2d):
                        bn_params.extend(module.parameters())
                        
                        if self.config.update_conv_before_bn and idx > 0:
                            for prev_idx in range(idx - 1, max(0, idx - 5), -1):
                                prev_name, prev_module = module_list[prev_idx]
                                if isinstance(prev_module, nn.Conv2d):
                                    conv_before_bn_params.extend(prev_module.parameters())
                                    break
                                    
                    elif "FrozenBatchNorm2d" in module.__class__.__name__:
                        if hasattr(module, 'weight') and not isinstance(module.weight, nn.Parameter):
                            module.weight = nn.Parameter(module.weight.clone())
                        if hasattr(module, 'bias') and not isinstance(module.bias, nn.Parameter):
                            module.bias = nn.Parameter(module.bias.clone())
                        if hasattr(module, 'weight'):
                            bn_params.append(module.weight)
                        if hasattr(module, 'bias'):
                            bn_params.append(module.bias)
    
                        if self.config.update_conv_before_bn and idx > 0:
                            for prev_idx in range(idx - 1, max(0, idx - 5), -1):
                                prev_name, prev_module = module_list[prev_idx]
                                if isinstance(prev_module, nn.Conv2d):
                                    conv_before_bn_params.extend(prev_module.parameters())
                                    break

                if self.config.update_mlp_after_norm:
                    if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                        if idx < len(module_list) - 1:
                            for next_idx in range(idx + 1, min(len(module_list), idx + 5)):
                                next_name, next_module = module_list[next_idx]
                                if isinstance(next_module, nn.Linear):
                                    mlp_after_norm_params.extend(next_module.parameters())
                                    break

                if self.config.update_fpn_last_layer:
                    if 'fpn_output' in name or 'fpn_lateral' in name:
                        if any(x in name for x in ['fpn_output5', 'fpn_lateral5', 'top_block']):
                            fpn_last_params.extend(module.parameters())

                if self.config.update_box_regressor_last_layer:
                    if 'box_predictor.bbox_pred' in name:
                        if isinstance(module, nn.Linear):
                            box_reg_last_params.extend(module.parameters())

                if self.config.update_backbone and 'backbone' in name:
                    if not any(isinstance(module, t) for t in [nn.BatchNorm2d, nn.Conv2d, nn.Linear]):
                        backbone_params.extend([p for p in module.parameters() if p not in bn_params])

                if self.config.update_head and 'roi_heads' in name:
                    if not any(isinstance(module, t) for t in [nn.BatchNorm2d, nn.Conv2d, nn.Linear]):
                        head_params.extend([
                            p for p in module.parameters() 
                            if p not in bn_params and p not in box_reg_last_params
                        ])

            # Remove duplicates while preserving order
            def global_deduplicate(param_groups):
                seen = set()
                result_groups = []
                
                for group_name, params, lr in param_groups:
                    unique_params = []
                    for p in params:
                        if id(p) not in seen:
                            seen.add(id(p))
                            unique_params.append(p)
                    result_groups.append((group_name, unique_params, lr))
                
                return result_groups

            param_groups = [
                ("BatchNorm", bn_params, self.config.adapt_lr),
                ("Conv_before_BN", conv_before_bn_params, self.config.adapt_lr * 0.5),
                ("MLP_after_norm", mlp_after_norm_params, self.config.adapt_lr * 0.5),
                ("FPN_last", fpn_last_params, self.config.head_lr),
                ("BoxReg_last", box_reg_last_params, self.config.head_lr),
                ("Backbone", backbone_params, self.config.backbone_lr),
                ("Head", head_params, self.config.head_lr),
            ]
            param_groups = global_deduplicate(param_groups)
        else:
            param_groups = [("All", list(self.base_model.parameters()), self.config.adapt_lr)]

        return param_groups

    def online_parameters(self):
        """Select parameters to adapt based on config."""
        params = []
        for _, param_list, _ in self._get_parameter_groups():
            params.extend(param_list)
        return params

    def detect_domain_change(self, current_loss_value: float) -> bool:
        """Idea A: Detection-quality-aware domain change detection."""
        if not self.config.enable_domain_change_reset or self.loss_history is None:
            return False

        if len(self.loss_history) < self.config.min_history_for_spike_detection:
            return False

        if self.adaptation_steps < self.config.min_history_for_spike_detection * 2:
            return False

        loss_array = np.array(list(self.loss_history))
        loss_mean = loss_array.mean()
        loss_std = loss_array.std()

        # Check if loss is decreasing (good adaptation)
        recent_losses = list(self.loss_history)[-10:]
        if len(recent_losses) >= 5:
            recent_mean = np.mean(recent_losses)
            if current_loss_value < recent_mean * 0.8:
                return False

        # Get adaptive threshold based on detection quality
        adaptive_threshold = self.get_adaptive_threshold()

        # Spike detection method 1: Relative to mean (with adaptive threshold)
        if loss_mean > 1e-6:
            relative_spike = current_loss_value / loss_mean
            if relative_spike > adaptive_threshold:
                print(f"[APT] Domain change detected! Loss spike: {current_loss_value:.4f} vs mean {loss_mean:.4f} ({relative_spike:.2f}x) [Quality: {self.detection_quality_mode}, Threshold: {adaptive_threshold:.1f}x]")
                return True

        # Spike detection method 2: Std-based (with adaptive threshold)
        if loss_std > 1e-6:
            z_score = abs(current_loss_value - loss_mean) / loss_std
            adaptive_std_threshold = self.config.domain_change_loss_spike_std_multiplier
            if self.detection_quality_mode == "high":
                adaptive_std_threshold *= 1.3
            elif self.detection_quality_mode == "low":
                adaptive_std_threshold *= 0.8
                
            if z_score > adaptive_std_threshold:
                print(f"[APT] Domain change detected! Loss outlier: z-score {z_score:.2f} [Quality: {self.detection_quality_mode}]")
                return True

        return False

    def should_skip_update(self, current_loss_value: float) -> bool:
        """Idea A: Determine if update should be skipped."""
        if not self.config.enable_loss_threshold_skip:
            return False

        # Absolute threshold
        if current_loss_value > self.config.loss_threshold_skip_value:
            print(f"[APT] Skipping update: loss {current_loss_value:.4f} exceeds absolute threshold {self.config.loss_threshold_skip_value}")
            return True

        # Relative threshold (to EMA)
        if self.loss_scale_ema > 1e-6:
            relative_loss = current_loss_value / self.loss_scale_ema
            if relative_loss > self.config.loss_threshold_skip_relative:
                print(f"[APT] Skipping update: loss {current_loss_value:.4f} is {relative_loss:.2f}x EMA {self.loss_scale_ema:.4f}")
                return True

        return False

    def compute_gradient_scaling_factor(self, loss_value: float) -> float:
        """Idea C: Compute gradient scaling factor based on loss magnitude."""
        if not self.config.enable_gradient_scaling:
            return 1.0

        base = self.config.gradient_scaling_base_loss
        
        if self.config.gradient_scaling_mode == "inverse_loss":
            if loss_value > 1e-6:
                scale = base / loss_value
            else:
                scale = self.config.gradient_scaling_max
                
        elif self.config.gradient_scaling_mode == "inverse_sqrt":
            if loss_value > 1e-6:
                scale = base / np.sqrt(loss_value)
            else:
                scale = self.config.gradient_scaling_max
                
        elif self.config.gradient_scaling_mode == "adaptive":
            if self.loss_scale_ema > 1e-6:
                loss_ratio = loss_value / self.loss_scale_ema
                scale = base / max(loss_ratio, 0.1)
            else:
                scale = 1.0
        else:
            scale = 1.0

        scale = np.clip(scale, self.config.gradient_scaling_min, self.config.gradient_scaling_max)
        
        return scale

    def reset_optimizer_state(self):
        """Reset optimizer momentum/state but keep model parameters."""
        self._optimizer = None
        if self.loss_history is not None:
            self.loss_history.clear()
        self.domain_change_detected_count += 1
        print(f"[APT] Optimizer reset complete (count: {self.domain_change_detected_count})")

    def update_bn_statistics(self, running_mean_grad: torch.Tensor, running_var_grad: torch.Tensor, module: nn.Module):
        """Idea D: Update BN statistics using gradient information."""
        if not self.config.update_bn_stats:
            return
            
        momentum = 0.1
        
        with torch.no_grad():
            if running_mean_grad is not None and module.running_mean is not None:
                module.running_mean -= momentum * running_mean_grad
                
            if running_var_grad is not None and module.running_var is not None:
                module.running_var -= momentum * running_var_grad
                module.running_var.clamp_(min=1e-5)

    def extract_detections(self, outputs, conf_threshold=None):
        """Extract detections from model outputs."""
        if conf_threshold is None:
            conf_threshold = self.config.min_confidence_for_update

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
        else:
            raise NotImplementedError(f"Unsupported model provider: {self.base_model.model_provider}")

    def compute_temporal_consistency_loss(
            self, current_boxes, current_scores, current_classes,
            motion_predictions, predicted_classes
    ):
        """Compute temporal consistency loss with confidence weighting."""
        if len(motion_predictions) == 0 or len(current_boxes) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        if isinstance(motion_predictions, np.ndarray):
            motion_predictions = torch.from_numpy(motion_predictions).float().to(self.device)
        if isinstance(predicted_classes, np.ndarray):
            predicted_classes = torch.from_numpy(predicted_classes).long().to(self.device)

        from torchvision.ops import box_iou

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        n_matched = 0

        for cls_id in torch.unique(predicted_classes):
            pred_mask = predicted_classes == cls_id
            curr_mask = current_classes == cls_id

            if not curr_mask.any() or not pred_mask.any():
                continue

            pred_cls_boxes = motion_predictions[pred_mask]
            curr_cls_boxes = current_boxes[curr_mask]
            curr_cls_scores = current_scores[curr_mask]

            iou_matrix = box_iou(curr_cls_boxes, pred_cls_boxes)

            if iou_matrix.numel() > 0:
                max_ious, max_indices = iou_matrix.max(dim=0)
                valid_matches = max_ious > self.config.iou_threshold

                if valid_matches.any():
                    matched_curr = curr_cls_boxes[max_indices[valid_matches]]
                    matched_pred = pred_cls_boxes[valid_matches]
                    matched_scores = curr_cls_scores[max_indices[valid_matches]]

                    if self.config.loss_type == "giou":
                        box_losses = self._giou_loss(matched_curr, matched_pred, reduction='none')
                    else:
                        box_losses = self.loss_function(matched_curr, matched_pred).mean(dim=1)

                    if self.config.use_confidence_weighting:
                        weights = matched_scores
                        weighted_loss = (box_losses * weights).sum()
                        total_loss = total_loss + weighted_loss
                    else:
                        total_loss = total_loss + box_losses.sum()

                    n_matched += valid_matches.sum().item()

        return total_loss, n_matched

    def forward(self, *args, **kwargs):
        """Forward pass with detection-quality-aware adaptation."""
        outputs = self.base_model(*args, **kwargs)

        if not self.adapting:
            return outputs

        if self.base_model.model_provider == ModelProvider.Detectron2:
            if isinstance(outputs, list):
                losses = []

                for output in outputs:
                    boxes, scores, classes = self.extract_detections(
                        output,
                        conf_threshold=self.config.min_confidence_for_update
                    )

                    # Update detection quality tracking
                    self.update_detection_quality_mode(len(boxes))

                    if self.current_motion_predictions is not None and len(self.current_motion_predictions) > 0:
                        current_boxes = output['instances'].pred_boxes.tensor
                        current_classes = output['instances'].pred_classes
                        current_scores = output['instances'].scores

                        conf_mask = current_scores >= self.config.conf_threshold

                        if conf_mask.any():
                            loss, n_matched = self.compute_temporal_consistency_loss(
                                current_boxes[conf_mask],
                                current_scores[conf_mask],
                                current_classes[conf_mask],
                                self.current_motion_predictions,
                                self.current_predicted_classes
                            )

                            if n_matched > 0 and loss.item() > 0:
                                # Apply detection-quality-aware normalization
                                normalized_loss = self.normalize_loss_by_detection_quality(loss, n_matched)
                                
                                with torch.no_grad():
                                    current_scale = loss.item() / max(1.0, n_matched)
                                    self.loss_scale_ema = (
                                            self.config.loss_ema_decay * self.loss_scale_ema +
                                            (1 - self.config.loss_ema_decay) * current_scale
                                    )

                                final_loss = normalized_loss * self.config.loss_weight
                                losses.append(final_loss)

                    if len(boxes) > 0:
                        motion_predictions, predicted_classes, _ = self.tracker.update(
                            boxes, classes
                        )
                    else:
                        motion_predictions, predicted_classes, _ = self.tracker.update(
                            np.empty((0, 4)), np.empty(0)
                        )

                    self.current_motion_predictions = motion_predictions
                    self.current_predicted_classes = predicted_classes

                if len(losses) > 0:
                    total_loss = torch.stack(losses).mean()
                    loss_value = total_loss.item()

                    if self.loss_history is not None:
                        self.loss_history.append(loss_value)

                    if self.should_skip_update(loss_value):
                        self.skipped_updates_count += 1
                        return outputs

                    if self.detect_domain_change(loss_value):
                        self.reset_optimizer_state()

                    if self.config.update_bn_stats:
                        bn_running_means_before = []
                        bn_running_vars_before = []
                        for module in self.bn_modules:
                            if module.running_mean is not None:
                                bn_running_means_before.append(module.running_mean.clone())
                            if module.running_var is not None:
                                bn_running_vars_before.append(module.running_var.clone())

                    self.optimizer.zero_grad()
                    total_loss.backward()

                    if self.config.enable_gradient_scaling:
                        scale_factor = self.compute_gradient_scaling_factor(loss_value)
                        if scale_factor != 1.0:
                            for param in self.online_parameters():
                                if param.grad is not None:
                                    param.grad *= scale_factor

                    torch.nn.utils.clip_grad_norm_(self.online_parameters(), max_norm=1.0)

                    self.optimizer.step()

                    if self.config.update_bn_stats:
                        for idx, module in enumerate(self.bn_modules):
                            if hasattr(module, 'running_mean') and module.running_mean is not None:
                                if idx < len(bn_running_means_before):
                                    mean_grad = module.running_mean - bn_running_means_before[idx]
                                    var_grad = None
                                    if hasattr(module, 'running_var') and module.running_var is not None:
                                        if idx < len(bn_running_vars_before):
                                            var_grad = module.running_var - bn_running_vars_before[idx]
                                    self.update_bn_statistics(mean_grad, var_grad, module)

                    self.adaptation_steps += 1
                    self.total_loss += loss_value

        return outputs

    def reset(self):
        """Reset adaptation state including tracker and detection quality tracking."""
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
        self.domain_change_detected_count = 0
        self.skipped_updates_count = 0
        
        # Reset detection quality tracking
        self.detection_count_history.clear()
        self.avg_detection_count = 0.0
        self.detection_quality_mode = "unknown"
        
        if self.loss_history is not None:
            self.loss_history.clear()

        super().reset()

    def get_adaptation_stats(self):
        """Get comprehensive adaptation statistics."""
        stats = {
            'adaptation_steps': self.adaptation_steps,
            'avg_loss': self.total_loss / max(1, self.adaptation_steps),
            'total_loss': self.total_loss,
            'num_tracks': len(self.tracker.trackers),
            'loss_scale_ema': self.loss_scale_ema,
            'domain_changes': self.domain_change_detected_count,
            'skipped_updates': self.skipped_updates_count,
            'avg_detection_count': self.avg_detection_count,
            'detection_quality_mode': self.detection_quality_mode,
        }
        
        if self.loss_history is not None and len(self.loss_history) > 0:
            loss_array = np.array(list(self.loss_history))
            stats.update({
                'loss_history_mean': loss_array.mean(),
                'loss_history_std': loss_array.std(),
                'loss_history_min': loss_array.min(),
                'loss_history_max': loss_array.max(),
            })
        
        return stats

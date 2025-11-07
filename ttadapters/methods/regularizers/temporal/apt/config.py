from dataclasses import dataclass
from typing import Literal

from ....base import AdaptationConfig


@dataclass
class APTConfig(AdaptationConfig):
    """
    Configuration for APT: Adaptive Plugin using Temporal Consistency
    
    This method adapts object detection models at test-time by leveraging
    temporal consistency between consecutive frames using Kalman filter tracking.

    Key concept: Uses motion predictions from Kalman filter as self-supervised
    signals (NOT pseudo-labels from model predictions).
    """
    adaptation_name: str = "APT"

    # Optimization settings
    optim: Literal["SGD", "Adam", "AdamW"] = "SGD"
    adapt_lr: float = 1e-5
    backbone_lr: float = 1e-6  # Learning rate for backbone (if updated)
    head_lr: float = 1e-6  # Learning rate for head (if updated)
    momentum: float = 0.9  # Momentum for SGD
    weight_decay: float = 0.0  # Weight decay for regularization (use with AdamW)

    # Temporal consistency settings
    max_age: int = 3  # Maximum frames to keep track without detection
    min_hits: int = 1  # Minimum hits to confirm a track
    iou_threshold: float = 0.8  # IOU threshold for matching detections to tracks

    # Loss settings
    loss_type: Literal["l1", "l2", "smooth_l1", "giou"] = "smooth_l1"
    loss_weight: float = 1.0
    use_confidence_weighting: bool = True  # Weight loss by detection confidence

    # Confidence thresholding
    conf_threshold: float = 0.7  # Confidence threshold for using detections
    min_confidence_for_update: float = 0.3  # Minimum confidence to update tracker

    # Update strategy
    update_backbone: bool = False  # Whether to update backbone/encoder
    update_head: bool = False  # Whether to update detection head
    update_bn: bool = True  # Whether to update scale param of batch norm
    update_fpn_last_layer: bool = False  # Whether to update FPN's last layer
    update_box_regressor_last_layer: bool = False  # Whether to update box regressor's last layer

    # Memory settings
    buffer_size: int = 500  # Number of frames to keep in memory for adaptation

    # Loss stabilization
    loss_ema_decay: float = 0.9  # EMA decay for loss scale normalization

    # Domain change detection (based on loss, NOT mAP)
    enable_domain_change_reset: bool = False  # Reset optimizer on domain change
    domain_change_loss_threshold: float = 0.5  # Loss spike threshold (50% increase)

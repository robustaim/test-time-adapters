from dataclasses import dataclass
from typing import Literal

from ....base import AdaptationConfig


@dataclass
class APTConfig(AdaptationConfig):
    """Configuration for APT: Adaptive Plugin using Temporal Consistency
    
    This method adapts object detection models at test-time by leveraging
    temporal consistency between consecutive frames using Kalman filter tracking.

    Key concept: Uses motion predictions from Kalman filter as self-supervised
    signals (NOT pseudo-labels from model predictions).
    """
    adaptation_name: str = "APT"

    # Optimization settings
    optim: Literal["SGD", "Adam", "AdamW"] = "Adam"
    adapt_lr: float = 1e-4

    # Temporal consistency settings
    max_age: int = 3  # Maximum frames to keep track without detection
    min_hits: int = 1  # Minimum hits to confirm a track
    iou_threshold: float = 0.3  # IOU threshold for matching detections to tracks

    # Loss settings
    loss_type: Literal["l1", "l2", "smooth_l1", "giou"] = "smooth_l1"
    loss_weight: float = 1.0

    # Confidence thresholding
    conf_threshold: float = 0.5  # Confidence threshold for using detections

    # Update strategy
    update_backbone: bool = True  # Whether to update backbone/encoder
    update_head: bool = True  # Whether to update detection head
    update_bn: bool = True  # Whether to update scale param of batch norm

    # Memory settings
    buffer_size: int = 10  # Number of frames to keep in memory for adaptation

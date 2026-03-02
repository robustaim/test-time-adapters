from typing import Literal
from dataclasses import dataclass

from ....base import AdaptationConfig


@dataclass
class MeanTeacherConfig(AdaptationConfig):
    adaptation_name = "MeanTeacherEngine"

    base_type: Literal["rcnn", "swinrcnn", "rtdetr", "yolo11"] = "rcnn"

    # Optimizer
    optim: Literal["SGD", "AdamW"] = "SGD"
    adapt_lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Mean Teacher
    conf_threshold: float = 0.3
    ema_alpha: float = 0.999
    weight_reg: float = 0.0

    # Augmentation
    augment_strength_n: int = 2
    augment_strength_m: int = 10
    cutout_size: int = 16

    @classmethod
    def from_preset(cls, base_model, **kwargs):
        """Create configuration from preset."""
        from .....models import (
            FasterRCNNForObjectDetection, SwinRCNNForObjectDetection,
            RTDetrForObjectDetection, YOLO11ForObjectDetection
        )
        if isinstance(base_model, FasterRCNNForObjectDetection):
            return cls(base_type="rcnn", **kwargs)
        elif isinstance(base_model, SwinRCNNForObjectDetection):
            return cls(base_type="swinrcnn", adapt_lr=3e-4, **kwargs)
        elif isinstance(base_model, RTDetrForObjectDetection):
            return cls(base_type="rtdetr", **kwargs)
        elif isinstance(base_model, YOLO11ForObjectDetection):
            return cls(base_type="yolo11", **kwargs)
        else:
            raise ValueError(f"Unsupported base model type: {type(base_model)}")

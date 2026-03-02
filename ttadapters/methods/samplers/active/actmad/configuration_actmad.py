from typing import Literal, Optional
from dataclasses import dataclass

from ....base import AdaptationConfig


@dataclass
class ActMADConfig(AdaptationConfig):
    adaptation_name = "ActMADEngine"

    base_type: Literal["rcnn", "swinrcnn", "rtdetr", "yolo11"] = "rcnn"
    adaptation_layers: Literal["backbone", "encoder", "backbone+encoder"] = "backbone+encoder"

    # Optimizer
    optim: Literal["SGD", "AdamW"] = "SGD"
    adapt_lr: float = 1e-6
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Loss
    loss_type: str = "L1"
    
    # Statistic
    statistic_save_path: Optional[str] = None
    clean_bn_extract_batch: int = 32

    @classmethod
    def from_preset(cls, base_model, **kwargs):
        """Create configuration from preset."""
        from .....models import (
            FasterRCNNForObjectDetection, SwinRCNNForObjectDetection,
            RTDetrForObjectDetection, YOLO11ForObjectDetection
        )
        if isinstance(base_model, FasterRCNNForObjectDetection):
            return cls(base_type="rcnn", adaptation_layers="backbone", **kwargs)
        elif isinstance(base_model, SwinRCNNForObjectDetection):
            return cls(base_type="swinrcnn", adaptation_layers="backbone", **kwargs)
        elif isinstance(base_model, RTDetrForObjectDetection):
            return cls(base_type="rtdetr", adaptation_layers="backbone+encoder", **kwargs)
        elif isinstance(base_model, YOLO11ForObjectDetection):
            return cls(base_type="yolo11", adaptation_layers="backbone", **kwargs)
        else:
            raise ValueError(f"Unsupported base model type: {type(base_model)}")

from typing import Literal
from dataclasses import dataclass

from ....base import AdaptationConfig


@dataclass
class NORMConfig(AdaptationConfig):
    adaptation_name = "NORMEngine"

    base_type: Literal["rcnn", "rtdetr", "yolo11"] = "rcnn"
    source_sum: int = 128
    adaptation_layers: Literal["backbone", "encoder", "backbone+encoder"] = "backbone+encoder"

    @classmethod
    def from_preset(cls, base_model, **kwargs):
        """Create configuration from preset."""
        from .....models import (
            FasterRCNNForObjectDetection, RTDetrForObjectDetection, YOLO11ForObjectDetection
        )
        if isinstance(base_model, FasterRCNNForObjectDetection):
            return cls(base_type="rcnn", adaptation_layers="backbone", **kwargs)
        elif isinstance(base_model, RTDetrForObjectDetection):
            return cls(base_type="rtdetr", adaptation_layers="backbone+encoder", **kwargs)
        elif isinstance(base_model, YOLO11ForObjectDetection):
            return cls(base_type="yolo11", adaptation_layers="backbone", **kwargs)
        else:
            raise ValueError(f"Unsupported base model type: {type(base_model)}")

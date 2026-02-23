from typing import Literal, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ...base import AdaptationConfig


class TargetKeyPreset(Enum):
    """
    Preset patterns for cascade_target.
    Strategies (S1~S5) are applied to the **EARLY 5 Blocks** of the backbone to fast optimization.
    """
    RESNET = [  # ResNet: res2 (2 blocks) + res3 (3 blocks) / stages 0 + 1 = 5 blocks
        r"\.res[23]\.[012]$",  # BottleneckBlock (Detectron2)
        r"\.stages\.[01]\.layers\.[012]$",  # RTDetrResNetBottleNeckLayer (RT-DETR)
    ]
    SWIN = [  # Swin: layer0 (2 blocks) + layer1 (2 blocks) + layer2-0 (1 block) = 5 blocks
        r"\.layers\.[01]\.blocks\.[012]$",  # SwinTransformerBlock
        #r"\.layers\.2\.blocks\.0$",  # SwinTransformerBlock
    ]
    C3K2 = [  # Yolo 11: model.2(2) + model.4(2) + model.6(2) = 6 blocks
        r"(^|\.)model\.[24]\.m\.0\.m\.[01]$",  # C3k2 Bottleneck
        #r"(^|\.)model\.6\.m\.0$",  # C3k2 Bottleneck
    ]


@dataclass
class FlowAdaptationConfig(AdaptationConfig):
    """Configuration for FlowAdaptation."""
    adaptation_name: str = "FlowAdaptationEngine"
    adapt_lr: float = 1e-3
    optim: Literal["SGD", "Adam", "AdamW"] = "Adam"

    # Engine configuration
    itm_type: Literal["clahe", "gamma", "clahe-gamma", "clahe-gamma-residual"] = "gamma"
    cascade_target: list[str] = None
    exclude_target: list[str] = field(default_factory=lambda: ["stem", "patch_embed", "embedder"])
    mask_value: int = 114  # YOLO11 default padding value
    masked_processing: bool = False

    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    # Gamma parameters
    gamma_temperature: float = 0.01
    gamma_range: tuple[float, float] = (0.5, 2.0)  # *2 to /2
    gamma_noise_floor: float = 2.0
    gamma_saturation_limit: float = 98.0

    # Anchor configutation
    use_kl_divergence: bool = True
    reduce_dim: tuple[int, ...] | None = None  # if None, use all dimensions

    @classmethod
    def from_preset(cls, base_model, **kwargs):
        """Create configuration from preset."""
        from ....models import (
            FasterRCNNForObjectDetection, SwinRCNNForObjectDetection,
            RTDetrForObjectDetection, YOLO11ForObjectDetection
        )
        if isinstance(base_model, FasterRCNNForObjectDetection):
            return cls(cascade_target=TargetKeyPreset.RESNET.value, **kwargs)
        elif isinstance(base_model, SwinRCNNForObjectDetection):
            return cls(cascade_target=TargetKeyPreset.SWIN.value, **kwargs)
        elif isinstance(base_model, RTDetrForObjectDetection):
            return cls(cascade_target=TargetKeyPreset.RESNET.value, **kwargs)
        elif isinstance(base_model, YOLO11ForObjectDetection):
            return cls(
                cascade_target=TargetKeyPreset.C3K2.value,
                masked_processing=True, mask_value=114, **kwargs
            )
        else:
            raise ValueError(f"Unsupported base model type: {type(base_model)}")

TARGET_KEY_PRESET = TargetKeyPreset

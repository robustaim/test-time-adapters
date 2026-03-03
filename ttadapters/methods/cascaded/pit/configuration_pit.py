from typing import Literal
from dataclasses import dataclass, field
from enum import Enum

from ...base import AdaptationConfig


class TargetKeyPreset(Enum):
    """
    Preset patterns for cascade_target.
    Strategies are applied to the **EARLY Blocks** of the backbone to fast optimization.
    """
    RESNET = [  # ResNet: res2 (3 blocks) / stages 0 = 3 blocks
        r"\.res2.*\.conv[123]\.norm$",  # BottleneckBlock (Detectron2)
        r"\.stages\.0.*\.layer\.[012]\.normalization$",  # RTDetrResNetBottleNeckLayer (RT-DETR)
    ]
    SWIN = [  # Swin: layer0 (2 blocks) + layer1 (2 blocks) = 4 blocks
        r"\.layers\.[01]\.blocks\..*\.norm[12]$"  # SwinTransformerBlock
    ]
    YOLO11_C3K2 = [  # YOLO11: backbone C3k2 only (model.2,4,6), concat-prior pathway BNs
        r"(^|\.)model\.[246]\.cv2\.bn$",  # C3k2 shortcut path final BN
        r"(^|\.)model\.[246]\.m\.\d+\.cv3\.bn$",  # C3k bottleneck path final BN
    ]
    YOLO11_STEM = [  # YOLO11: stem only (model.0)
        r"(^|\.)model\.0\.bn$",  # stem BN (closest to input)
    ]
    YOLO11_STEM_C3K2 = [  # YOLO11: stem (model.0) + backbone C3k2 (model.2,4,6)
        r"(^|\.)model\.0\.bn$",  # stem BN (closest to input)
        r"(^|\.)model\.[246]\.cv2\.bn$",  # C3k2 shortcut path final BN
        r"(^|\.)model\.[246]\.m\.\d+\.cv3\.bn$",  # C3k bottleneck path final BN
    ]
    YOLO11 = [  # YOLO11: all C3k2 blocks (backbone + neck + head-side), concat-prior pathway BNs
        r"(^|\.)model\.\d+\.cv2\.bn$",  # C3k2 shortcut path final BN
        r"(^|\.)model\.\d+\.m\.\d+\.cv3\.bn$",  # C3k bottleneck path final BN
    ]


@dataclass
class PITConfig(AdaptationConfig):
    """Configuration for PITConfig."""
    adaptation_name = "PITEngine"

    adapt_lr: float = 1e-3
    optim: Literal["SGD", "Adam", "AdamW"] = "Adam"

    # Engine configuration
    itm_type: Literal["clahe", "gamma", "clahe-gamma", "clahe-gamma-residual"] = "gamma"
    cascade_target: list[str] = None
    exclude_target: list[str] = field(default_factory=lambda: ["stem", "patch_embed", "embedder"])
    disable_blending: bool = False
    mask_value: int = 114  # YOLO11 default padding value
    masked_processing: bool = False

    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    # Gamma parameters
    use_differentiable_stretch: bool = True
    gamma_temperature: float = 0.01
    gamma_range: tuple[float, float] = (0.5, 2.0)  # *2 to /2
    gamma_noise_floor: float = 0.0
    gamma_saturation_limit: float = 100.0

    # Anchor configuration
    use_kl_divergence: bool = True  # if false, use MSE loss

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
                cascade_target=TargetKeyPreset.YOLO11_STEM_C3K2.value,
                masked_processing=True, mask_value=114, **kwargs
            )
        else:
            raise ValueError(f"Unsupported base model type: {type(base_model)}")

TARGET_KEY_PRESET = TargetKeyPreset

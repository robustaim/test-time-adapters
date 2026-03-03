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
    C3K2 = [  # YOLO11 Early stages only (layers 2, 4)
        r"(^|\.)model\.[24]\..*\.bn$",  # C3k2 block
    ]
    C3K2_S1 = [
        r"(^|\.)model\.[24]\.cv1\.bn$",  # C3k2 block
        r"(^|\.)model\.[24]\.m\.0\.cv1\.bn$",  # C3k block
    ]
    C3K2_S2 = [
        r"(^|\.)model\.[24]\.cv2\.bn$",  # C3k2 block
        r"(^|\.)model\.[24]\.m\.0\.cv3\.bn$",  # C3k block
    ]
    C3K2_S3 = [
        r"(^|\.)model\.[24]\.cv1\.bn$",  # C3k2 block
        r"(^|\.)model\.[24]\.m\.0\.cv[12]\.bn$",  # C3k block
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
    use_differentiable_stretch: bool = False
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
                cascade_target=TargetKeyPreset.C3K2.value,
                masked_processing=True, mask_value=114, **kwargs
            )
        else:
            raise ValueError(f"Unsupported base model type: {type(base_model)}")

TARGET_KEY_PRESET = TargetKeyPreset

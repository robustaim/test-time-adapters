from typing import Literal, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ...base import AdaptationConfig


class TargetKeyPreset(Enum):
    """
    Preset patterns for cascade_target.
    Strategies (S1~S5) are applied to the **EARLY Blocks** of the backbone to fast optimization.

    Principles:
    - STRATEGY 0 (All): Adapt all intra-block normalizations.
    - STRATEGY 1 (First): Adapt only the first BN/LN (Source Anchor).
    - STRATEGY 2 (Last): Adapt only the last BN/LN (Target Anchor).
    - STRATEGY 3 (Proximal): Adapt First and Immediate Next.
    - STRATEGY 4 (Distal): Adapt Last and Immediate Prev.
    """
    RESNET = [  # ResNet: res2 + res3 / stages 0 + 1
        r"\.res[23].*\.conv[123]\.norm$",  # BottleneckBlock (Detectron2)
        r"\.stages\.[01].*\.layer\.[012]\.normalization$",  # RTDetrResNetBottleNeckLayer (RT-DETR)
    ]
    RESNET_S1 = [
        r"\.res[23].*\.conv1\.norm$",  # BottleneckBlock (Detectron2)
        r"\.stages\.[01].*\.layer\.0\.normalization$",  # RTDetrResNetBottleNeckLayer (RT-DETR)
    ]
    RESNET_S2 = [
        r"\.res[23].*\.conv3\.norm$",  # BottleneckBlock (Detectron2)
        r"\.stages\.[01].*\.layer\.2\.normalization$",  # RTDetrResNetBottleNeckLayer (RT-DETR)
    ]
    RESNET_S3 = [
        r"\.res[23].*\.conv[12]\.norm$",  # BottleneckBlock (Detectron2)
        r"\.stages\.[01].*\.layer\.[01]\.normalization$",  # RTDetrResNetBottleNeckLayer (RT-DETR)
    ]
    RESNET_S4 = [
        r"\.res[23].*\.conv[23]\.norm$",  # BottleneckBlock (Detectron2)
        r"\.stages\.[01].*\.layer\.[12]\.normalization$",  # RTDetrResNetBottleNeckLayer (RT-DETR)
    ]
    SWINT = [  # SwinT: layer0 + layer1
        r"\.layers\.[01]\.blocks\..*\.norm[12]$"  # SwinTransformerBlock
    ]
    SWINT_S1 = [
        r"\.layers\.[01]\.blocks\..*\.norm1$"  # SwinTransformerBlock
    ]
    SWINT_S2 = [
        r"\.layers\.[01]\.blocks\..*\.norm2$"  # SwinTransformerBlock
    ]
    SWINT_S4 = SWINT_S3 = SWINT  # Only 2 LNs in Swin, so S4=S3=ALL
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
    C3K2_S4 = [
        r"(^|\.)model\.[24]\.cv2\.bn$",  # C3k2 block
        r"(^|\.)model\.[24]\.m\.0\.cv[23]\.bn$",  # C3k block
    ]


@dataclass
class CascadedNormConfig(AdaptationConfig):
    """Configuration for CascadedNorm."""
    adaptation_name: str = "CascadedNormEngine"
    adapt_lr: float = 1e-3
    optim: Literal["SGD", "Adam", "AdamW"] = "Adam"

    # Engine configuration
    itm_type: Literal["clahe", "gamma", "clahe-gamma"] = "gamma"
    itm_combination_method: Literal["residual", "hierarchical", "frequency"] | None = None  # only used when itm_type == "clahe-gamma"
    cascade_target: List[str] = None
    exclude_target: List[str] = field(default_factory=lambda: ["stem", "patch_embed", "embedder"])
    mask_value: int = 114  # YOLO11 default padding value
    masked_processing: bool = False

    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    # Gamma parameters
    gamma_temperature: float = 0.01
    gamma_range: Tuple[float, float] = (0.5, 2.0)  # *2 to /2
    gamma_noise_floor: float = 2.0
    gamma_saturation_limit: float = 98.0

    # CLAHE-Gamma parameters
    frequency_combination_kernel_size: int = 3
    frequency_combination_sigma: float = 1.0

    # Anchor configutation
    use_kl_divergence: bool = False
    use_feature_alignment: bool = False
    use_bn_running_stat: bool = False

    @classmethod
    def from_preset(cls, base_model, **kwargs):
        """Create configuration from preset."""
        from ....models import (
            FasterRCNNForObjectDetection, SwinRCNNForObjectDetection,
            RTDetrForObjectDetection, YOLO11ForObjectDetection
        )
        if isinstance(base_model, FasterRCNNForObjectDetection):
            return cls(cascade_target=TargetKeyPreset.RESNET_S3.value, **kwargs)
        elif isinstance(base_model, SwinRCNNForObjectDetection):
            return cls(cascade_target=TargetKeyPreset.SWINT_S3.value, **kwargs)
        elif isinstance(base_model, RTDetrForObjectDetection):
            return cls(cascade_target=TargetKeyPreset.RESNET_S3.value, **kwargs)
        elif isinstance(base_model, YOLO11ForObjectDetection):
            return cls(
                cascade_target=TargetKeyPreset.C3K2_S3.value,
                masked_processing=True, mask_value=114, **kwargs
            )
        else:
            raise ValueError(f"Unsupported base model type: {type(base_model)}")

TARGET_KEY_PRESET = TargetKeyPreset

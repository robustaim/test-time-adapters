from typing import Literal, List, Tuple
from dataclasses import dataclass
from enum import Enum

from ...base import AdaptationConfig


class TargetKeyPreset(Enum):
    """
    Preset patterns for cascade_target to match the LAST normalization layer in residual blocks.
    
    Flow Adaptation Principle:
    - First BN/LN in block: applies source-aligned transformation
    - Last BN/LN in block: target for N(0,1) alignment (soft bypassing)
    - Anchors are inserted at the LAST norm layer of each block
    """
    SWINT = [
        r"\.norm2$",  # Last LayerNorm in SwinTransformerBlock (after MLP)
    ]
    RESNET = [
        r"\.conv3\.norm$",  # Last BN in BottleneckBlock (Detectron2)
        r"\.layer\.2\.normalization$",  # Last BN in RTDetrResNetBottleNeckLayer (RT-DETR)
    ]
    C3K2 = [  # YOLO11
        # Single level granularity - block level only (simpler and less anchors)
        r"C3k2.*\.cv2\.bn$",  # Last BN of C3k2 block itself
    ]
    C3K2_BTL = [  # YOLO11
        # Dual level granularities
        r"Bottleneck.*\.cv2\.bn$",  # Last BN in each Bottleneck within C3k2->C3k->m
        r"C3k2.*\.cv2\.bn$",  # Last BN of C3k2 block itself
    ]


@dataclass
class CascadedNormConfig(AdaptationConfig):
    """Configuration for CascadedNorm."""
    adaptation_name: str = "CascadedNormEngine"
    adapt_lr: float = 1e-3
    optim: Literal["SGD", "Adam", "AdamW"] = "AdamW"

    # Engine configuration
    itm_type: Literal["clahe", "gamma"] = "gamma"
    cascade_target: List[str] = None

    # CLAHE parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    # Gamma parameters
    gamma_temperature: float = 0.01
    gamma_range: Tuple[float, float] = (0.5, 2.0)  # *2 to /2
    gamma_noise_floor: float = 2.0
    gamma_saturation_limit: float = 98.0

    # Anchor configutation
    frozen_bn_num_samples: int = 128  # same value from NORM

    @classmethod
    def from_preset(cls, base_model: "nn.Module", **kwargs):
        """Create configuration from preset."""
        from ....models import (
            FasterRCNNForObjectDetection, SwinRCNNForObjectDetection,
            RTDetrForObjectDetection, YOLO11ForObjectDetection
        )
        if isinstance(base_model, FasterRCNNForObjectDetection):
            return cls(cascade_target=TargetKeyPreset.RESNET.value, **kwargs)
        elif isinstance(base_model, SwinRCNNForObjectDetection):
            return cls(cascade_target=TargetKeyPreset.SWINT.value, **kwargs)
        elif isinstance(base_model, RTDetrForObjectDetection):
            return cls(cascade_target=TargetKeyPreset.RESNET.value, **kwargs)
        elif isinstance(base_model, YOLO11ForObjectDetection):
            return cls(cascade_target=TargetKeyPreset.C3K2.value, **kwargs)
        else:
            raise ValueError(f"Unsupported base model type: {type(base_model)}")

TARGET_KEY_PRESET = TargetKeyPreset

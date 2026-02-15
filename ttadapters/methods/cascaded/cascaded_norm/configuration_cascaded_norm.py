from typing import Literal, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ...base import AdaptationConfig


@dataclass
class CascadedNormConfig(AdaptationConfig):
    """Configuration for CascadedNorm."""
    adaptation_name: str = "CascadedNormEngine"
    adapt_lr: float = 1e-3
    optim: Literal["SGD", "Adam", "AdamW"] = "Adam"

    # Engine configuration
    itm_type: Literal["clahe", "gamma", "clahe-gamma"] = "gamma"
    itm_combination_method: Literal["residual", "hierarchical", "frequency"] | None = None  # only used when itm_type == "clahe-gamma"
    cascade_target: List[str] = [r"(^|\.)backbone$"]
    exclude_target: List[str] = field(default_factory=lambda: [])
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

    @classmethod
    def from_preset(cls, base_model, **kwargs):
        """Create configuration from preset."""
        from ....models import (
            FasterRCNNForObjectDetection, SwinRCNNForObjectDetection,
            RTDetrForObjectDetection, YOLO11ForObjectDetection
        )
        if isinstance(base_model, FasterRCNNForObjectDetection):
            return cls(**kwargs)
        elif isinstance(base_model, SwinRCNNForObjectDetection):
            return cls(**kwargs)
        elif isinstance(base_model, RTDetrForObjectDetection):
            return cls(**kwargs)
        elif isinstance(base_model, YOLO11ForObjectDetection):
            return cls(
                masked_processing=True, mask_value=114, **kwargs
            )
        else:
            raise ValueError(f"Unsupported base model type: {type(base_model)}")

TARGET_KEY_PRESET = TargetKeyPreset

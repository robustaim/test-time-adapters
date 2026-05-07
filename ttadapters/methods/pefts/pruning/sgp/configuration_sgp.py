from typing import Literal
from dataclasses import dataclass, field
from enum import Enum

from ....base import AdaptationConfig


class ExcludeLayerPreset(Enum):
    """
    Preset patterns for BN layers to EXCLUDE from pruning.
    Higher layers (closer to output) are excluded to preserve task-specific features.
    """
    RESNET_FPN = [  # FasterRCNN (ResNet-50-FPN): exclude res5 (last stage)
        r"\.res5\.",
    ]


@dataclass
class SGPConfig(AdaptationConfig):
    """Configuration for SGPEngine (Sensitivity-Guided Pruning).

    Reference:
        "Efficient Test-time Adaptive Object Detection via Sensitivity-Guided Pruning" (CVPR 2025)

    Note:
        SGP relies on BatchNorm2d scaling factors (γ) for channel pruning.
        Models using LayerNorm (e.g. SwinTransformer) are NOT supported.
        Currently supports Detectron2 ResNet-FPN based detectors only.
    """
    adaptation_name = "SGPEngine"

    # --- Optimizer ---
    optim: Literal["SGD", "Adam", "AdamW"] = "Adam"
    adapt_lr: float = 5e-3
    momentum: float = 0.9

    # --- Pruning Parameters ---
    pruning_rate: float = 0.10          # target pruning ratio (p)
    pruning_threshold: float = 0.05     # |γ| < threshold → prune

    # --- Loss Weights ---
    lambda_align: float = 1.0           # L_adp alignment loss weight
    lambda_sparse: float = 0.05         # λ for L_wreg (weighted sparsity)

    # --- EMA for target mean tracking (Sec 3.4) ---
    ema_momentum: float = 0.1           # γ in μ_t = (1-γ)·μ_t + γ·batch_mean

    # --- Stochastic Channel Reactivation ---
    reactivation_prob: float = 0.01     # Bernoulli probability for pruned channel reactivation

    # --- Instance-Level Sensitivity & Alignment ---
    use_instance_sensitivity: bool = True
    roi_output_size: int = 7
    fg_confidence_threshold: float = 0.5  # bg confidence < 0.5 → foreground

    # --- Layer Targeting ---
    exclude_layers: list[str] | None = None

    # --- Source Statistics ---
    source_stats_path: str | None = None

    @classmethod
    def from_preset(cls, base_model, **kwargs):
        from .....models import FasterRCNNForObjectDetection
        if isinstance(base_model, FasterRCNNForObjectDetection):
            return cls(
                exclude_layers=ExcludeLayerPreset.RESNET_FPN.value,
                **kwargs
            )
        else:
            raise NotImplementedError(
                f"SGP requires a ResNet-backbone detector with BatchNorm2d layers. "
                f"Got: {type(base_model).__name__}"
            )

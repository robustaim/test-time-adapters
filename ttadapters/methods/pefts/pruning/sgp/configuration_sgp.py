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

    The SGP algorithm prunes domain-sensitive BN channels via weighted sparsity
    regularization and adapts the remaining domain-invariant channels through
    feature alignment.

    Note:
        SGP relies on BatchNorm2d scaling factors (γ) for channel pruning.
        Models using LayerNorm (e.g. SwinTransformer) are NOT supported.
        Currently supports Detectron2 ResNet-FPN based detectors only.
    """
    adaptation_name = "SGPEngine"

    # --- Optimizer ---
    optim: Literal["SGD", "Adam", "AdamW"] = "Adam"
    adapt_lr: float = 1e-3
    momentum: float = 0.9

    # --- Pruning Parameters ---
    pruning_rate: float = 0.20          # target pruning ratio (p)
    pruning_threshold: float = 0.04     # |γ| < threshold → prune

    # --- Loss Weights ---
    lambda_align: float = 1.0           # L_adp alignment loss weight
    lambda_mu_std: float = 1.0          # std alignment weight within L_adp
    lambda_sparse: float = 0.1          # λ for L_wreg (weighted sparsity)

    # --- Stochastic Channel Reactivation ---
    reactivation_prob: float = 0.1      # Bernoulli probability for pruned channel reactivation

    # --- Instance-Level Sensitivity ---
    use_instance_sensitivity: bool = True   # enable instance-level (RoI) sensitivity
    roi_output_size: int = 7                # spatial size for RoI-Align
    fg_confidence_threshold: float = 0.5    # foreground confidence threshold for RoI selection

    # --- Layer Targeting ---
    exclude_layers: list[str] | None = None     # regex patterns for BN layers to exclude from pruning

    # --- Source Statistics ---
    source_stats_path: str | None = None        # path to pre-computed source stats (.pt file)

    @classmethod
    def from_preset(cls, base_model, **kwargs):
        """Create configuration from a preset based on the base model type.

        Only ResNet-FPN based Detectron2 detectors are supported because SGP
        requires BatchNorm2d layers for channel pruning.
        """
        from .....models import FasterRCNNForObjectDetection
        if isinstance(base_model, FasterRCNNForObjectDetection):
            return cls(
                exclude_layers=ExcludeLayerPreset.RESNET_FPN.value,
                **kwargs
            )
        else:
            raise NotImplementedError(
                f"SGP requires a ResNet-backbone detector with BatchNorm2d layers. "
                f"Models using LayerNorm (e.g. SwinTransformer) are not supported. "
                f"Got: {type(base_model).__name__}"
            )

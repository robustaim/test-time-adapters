from typing import Literal
from dataclasses import dataclass, field
from enum import Enum

from ....base import AdaptationConfig


class ExcludeLayerPreset(Enum):
    """
    Preset patterns for BN layers to EXCLUDE from pruning.
    Higher layers (closer to output) are excluded to preserve task-specific features.
    """
    RESNET_FPN = [  # FasterRCNN (ResNet-50-FPN)
        # Last stage exclusion: preserve high-level task-specific features.
        r"\.res5\.",
        # R50-Bottleneck conv3.norm layers have a bimodal γ distribution by
        # design: 15-24% of channels are naturally below the paper's t=0.05
        # threshold because the residual-identity path makes them dormant in
        # clean source training. This poisons the current_rate computation
        # (paper Eq.12) and forces the L1 sparsity loss to over-prune the
        # remaining unimodal layers. Excluding conv3.norm restricts pruning to
        # the unimodal conv1/conv2 layers where γ ∈ [0.4, 1.5] and the paper's
        # threshold/ratio assumptions match. The paper used R18-BasicBlock
        # which has no conv3 analog, so this exclusion is R50-specific.
        r"\.conv3\.norm",
    ]


@dataclass
class SGPConfig(AdaptationConfig):
    """Configuration for SGPEngine (Sensitivity-Guided Pruning).

    Reference:
        Wang et al., "Efficient Test-time Adaptive Object Detection via
        Sensitivity-Guided Pruning", CVPR 2025 (arXiv:2506.02462).

    The SGP algorithm prunes domain-sensitive BN channels via weighted
    sparsity regularization (L_wreg, Eq.3) and adapts the remaining
    domain-invariant channels through feature-distribution alignment
    (L_adp = L_img + L_ins, Eq.9-11). Stochastic channel reactivation
    (Eq.14) restores randomly selected pruned channels to their source
    pre-trained γ to mitigate over-pruning.

    Note:
        SGP relies on BatchNorm2d scaling factors (γ) for channel pruning.
        Models using LayerNorm (e.g. SwinTransformer) are NOT supported.
        Currently supports Detectron2 ResNet-FPN based detectors only.

    Paper-vs-implementation differences:
        Paper trained R18 + batch_size=4 + lr=5e-3. This codebase uses
        R50-FPN + batch_size=1 (online TTA). Several hyperparameters
        deviate to compensate; each deviating field documents its paper
        value and the rationale below.
    """
    adaptation_name = "SGPEngine"

    # --- Optimizer ---
    optim: Literal["SGD", "Adam", "AdamW"] = "Adam"
    adapt_lr: float = 5e-5      # paper: 5e-3 (Sec 4.1) | (100x lower).

    # --- Pruning Parameters (paper Algorithm 1, Eq.12) ---
    pruning_rate: float = 0.10          # target pruning ratio p (paper).
    pruning_threshold: float = 0.05     # γ < t → prune (paper).

    # --- Loss Weights (paper Eq.13) ---
    # L_total = λ_align · L_adp + λ_sparse · L_wreg     if ρ < p
    # L_total = λ_align · L_adp                         if ρ ≥ p
    lambda_align: float = 1.0           # implicit weight on L_adp (paper has no explicit symbol).
    lambda_sparse: float = 0.05         # λ in Eq.13 (paper Sec 4.1).

    # --- L_adp KL parameters (paper Eq.9) ---
    # μ_t is EMA-estimated; the paper says "exponentially moving average"
    # without specifying α. We use 0.5; an ablation sweep over {0.99, 0.5, 0.1}
    # showed 0.5 best for our R50+B=1 setup but the difference is small.
    target_ema_momentum: float = 0.5
    # Numerical floor on σ²_s before division in the KL closed form
    # 0.5 · (μ_s − μ_t)² / σ²_s. Paper has no floor; we add one for safety
    # against rare near-zero σ²_s channels. Set tight enough not to alter
    # the loss for the >99% of channels with σ²_s ≫ floor.
    source_var_floor: float = 1e-4

    # --- Stochastic Channel Reactivation (paper Eq.14) ---
    reactivation_prob: float = 0.01     # r — Bernoulli prob per pruned channel (paper Sec 4.1).

    # --- Instance-Level Sensitivity (paper Eq.7) ---
    use_instance_sensitivity: bool = True   # enable per-BN-layer instance sensitivity (D8).
    roi_output_size: int = 7                # spatial size for RoI-Align (paper unspecified; standard).
    fg_confidence_threshold: float = 0.5    # paper Sec 3.3: "RoIs with background confidence < 0.5".

    # --- Layer Targeting ---
    # Regex patterns for BN layers to EXCLUDE from pruning. ``None`` means
    # all backbone BN layers are eligible. The ``RESNET_FPN`` preset (set via
    # ``from_preset``) excludes ``res5`` and ``conv3.norm`` for R50-specific
    # reasons documented in ``ExcludeLayerPreset``.
    exclude_layers: list[str] | None = None

    # --- Source Statistics ---
    # Path to pre-computed source stats (.pt file). When ``None``, ``fit()``
    # collects fresh statistics from the source dataset before TTA. Schema:
    #   {"bn":           {layer_name: {"mean": Tensor, "var": Tensor}},
    #    "roi":          {stage_name: {"mean": Tensor, "var": Tensor}},
    #    "roi_per_class":{class_idx:  {"mean": Tensor, "var": Tensor, "count": int}},
    #    "bn_roi":       {layer_name: {"mean": Tensor, "var": Tensor}}}
    source_stats_path: str | None = None

    @classmethod
    def from_preset(cls, base_model, **kwargs):
        """Create configuration from a preset based on the base model type.

        Only ResNet-FPN based Detectron2 detectors are supported because SGP
        requires BatchNorm2d layers for channel pruning.
        """
        from .....models import FasterRCNNForObjectDetection
        if isinstance(base_model, FasterRCNNForObjectDetection):
            return cls(exclude_layers=ExcludeLayerPreset.RESNET_FPN.value, **kwargs)
        else:
            raise NotImplementedError(
                f"SGP requires a ResNet-backbone detector with BatchNorm2d layers. "
                f"Models using LayerNorm (e.g. SwinTransformer) are not supported. "
                f"Got: {type(base_model).__name__}"
            )

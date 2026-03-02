from typing import Literal
from dataclasses import dataclass

from ....base import AdaptationConfig


@dataclass
class WHWConfig(AdaptationConfig):
    adaptation_name = "WHWEngine"

    # Backbone type
    backbone: Literal["rcnn", "swinrcnn"] = "rcnn"

    # Optimizer
    optim: Literal["SGD", "AdamW"] = "SGD"
    adapt_lr: float = 1e-6
    momentum: float = 0.9
    weight_decay: float = 1e-4
    clip_grad_enabled: bool = True
    clip_grad_type: Literal["value", "norm"] = "value"
    clip_grad_value: float = 1.0

    # Adapter
    adapter_ratio: int = 32

    # Alignment
    alpha_gl: float = 1.0
    alpha_fg: float = 1.0
    gl_align: Literal["KL", "bn_stats", None] = "KL"
    fg_align: Literal["KL", None] = "KL"
    ema_gamma: int = 128
    freq_weight: bool = True

    # Source statistics
    source_stats_path: str = None

    # Skip settings
    skip_redundant: Literal["stat", "period", "ema", "stat-period-ema", None] = None
    skip_tau: float = 1.1
    skip_period: int = 10
    skip_beta: float = 1.05

    # Misc
    num_classes: int = 6
    collect_iou_thr: float = 0.5

    # SwinT adapter
    adapter_layernorm_option: str = "in"
    adapter_scalar: str = "constant"
    adapter_init_option: str = "lora"
    adapter_dropout: float = 0.0
    out_batch_norm: bool = False
    out_batch_resolution: int = 32

    @classmethod
    def from_preset(cls, base_model, **kwargs):
        """Create configuration from preset."""
        from .....models import FasterRCNNForObjectDetection, SwinRCNNForObjectDetection
        if isinstance(base_model, FasterRCNNForObjectDetection):
            return cls(backbone="rcnn", skip_redundant="stat-period-ema", **kwargs)
        elif isinstance(base_model, SwinRCNNForObjectDetection):
            return cls(backbone="swinrcnn", skip_redundant="stat-period-ema", **kwargs)
        else:
            raise ValueError(f"Unsupported base model type: {type(base_model)}")

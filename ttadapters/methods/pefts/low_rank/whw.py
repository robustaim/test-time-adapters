"""
WHW (What, How, and Where to adapt) Engine

Framework-agnostic implementation using adapter pattern.
Uses parallel adapters for efficient test-time adaptation.

Note: Full implementation currently supports Detectron2 only.
Extension to other frameworks requires framework-specific adapter injection.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim

from ...base import AdaptationEngine, AdaptationConfig
from ...framework_adapters import FrameworkAdapter, create_adapter


@dataclass
class WHWConfig(AdaptationConfig):
    """Configuration for WHW engine"""
    adaptation_name: str = "WHW"

    # Adaptation settings
    adaptation_where: str = "adapter"  # "adapter", "normalization", "full"
    adapter_bottleneck_ratio: int = 32

    # Optimization settings
    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    optimizer_type: str = "SGD"  # "SGD" or "AdamW"

    # Alignment settings
    fg_align: Optional[str] = "KL"  # Foreground alignment: "KL" or None
    gl_align: Optional[str] = "KL"  # Global alignment: "KL" or None
    alpha_fg: float = 1.0  # Foreground alignment weight
    alpha_gl: float = 1.0  # Global alignment weight
    ema_gamma: int = 128  # EMA update gamma

    # Redundancy skipping
    skip_redundant: Optional[str] = None  # "ema+", "period+stat", etc.
    skip_period: int = 10
    skip_beta: float = 1.2
    skip_tau: float = 1.0

    # Source statistics
    source_statistics_path: Optional[Path] = None

    # Device
    device: str = "cuda"


class WHWEngine(AdaptationEngine):
    """
    WHW: What, How, and Where to adapt

    Uses parallel adapters for efficient test-time adaptation with
    feature alignment.

    Framework-agnostic interface, but full implementation currently
    supports Detectron2 only. Extension to other frameworks requires
    framework-specific adapter injection strategies.

    For Transformers and Ultralytics: Use ActMADEngine, DUAEngine, or NORMEngine instead.
    """

    model_name = "WHW"

    def __init__(
        self,
        basemodel: nn.Module,
        config: WHWConfig,
        adapter: Optional[FrameworkAdapter] = None
    ):
        super().__init__(basemodel, config)

        # Auto-detect framework adapter if not provided
        if adapter is None:
            adapter = create_adapter(basemodel)

        self.adapter = adapter
        self.cfg = config
        self.device = torch.device(config.device)

        # Check framework support
        if self.adapter.framework_name != "detectron2":
            warnings.warn(
                f"WHWEngine full implementation currently supports Detectron2 only. "
                f"Detected framework: {self.adapter.framework_name}. "
                f"Consider using ActMADEngine, DUAEngine, or NORMEngine instead."
            )

        # Optimizer
        self.optimizer = None

        # Statistics
        self.source_stats: Optional[Dict[str, Any]] = None
        self.target_stats: Dict[str, Any] = {}

        # Adaptation counters
        self.adaptation_steps = 0
        self.used_steps = 0

        # Loss EMA for skipping
        self.loss_ema99 = 0.0
        self.loss_ema95 = 0.0
        self.loss_ema90 = 0.0

        # Setup
        self._setup()

    def _setup(self):
        """Initialize the WHW engine"""
        # Move model to device
        self.basemodel.to(self.device)

        # Load source statistics if available
        self._load_source_statistics()

        # Setup adapters (framework-specific)
        if self.adapter.framework_name == "detectron2":
            self._setup_detectron2_adapters()
        else:
            warnings.warn(
                f"Adapter setup not implemented for {self.adapter.framework_name}. "
                f"Model will run without adaptation."
            )

        # Setup optimizer
        self._setup_optimizer()

    def _load_source_statistics(self):
        """Load source domain statistics if available"""
        if self.cfg.source_statistics_path and self.cfg.source_statistics_path.exists():
            print(f"Loading source statistics from {self.cfg.source_statistics_path}")
            self.source_stats = torch.load(self.cfg.source_statistics_path)
        else:
            warnings.warn(
                "No source statistics provided. WHW will run without feature alignment. "
                "Provide source_statistics_path for full functionality."
            )

    def _setup_detectron2_adapters(self):
        """
        Setup parallel adapters for Detectron2 ResNet

        This is framework-specific and requires intimate knowledge of model architecture.
        """
        # Import baseline.py implementation details
        from ...other_method.baseline import (
            ParallelAdapter,
            ParallelAdapterWithProjection,
            ConvTaskWrapper
        )

        # Check if model has ResNet backbone
        if not (hasattr(self.basemodel, 'backbone') and
                hasattr(self.basemodel.backbone, 'bottom_up')):
            warnings.warn("Detectron2 model does not have expected ResNet structure")
            return

        # This would require detailed implementation from baseline.py
        # For now, just a placeholder
        warnings.warn(
            "Detectron2 adapter setup is complex and requires migration of "
            "ParallelAdapter, ConvTaskWrapper, and ResNet patching code. "
            "Please refer to baseline.py WHW implementation for details."
        )

    def _setup_optimizer(self):
        """Setup optimizer for adapted parameters"""
        # Collect trainable parameters
        params = [p for p in self.basemodel.parameters() if p.requires_grad]

        if len(params) == 0:
            warnings.warn("No trainable parameters found. WHW will not adapt.")
            self.optimizer = None
            return

        if self.cfg.optimizer_type == "SGD":
            self.optimizer = optim.SGD(
                params,
                lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer_type == "AdamW":
            self.optimizer = optim.AdamW(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.cfg.optimizer_type}")

        print(f"WHWEngine: Training {len(params)} parameters")

    def forward(self, *args, **kwargs):
        """
        Forward pass with WHW adaptation

        Args:
            batch: Input batch (framework-specific format)

        Returns:
            Model outputs
        """
        # Get batch from args or kwargs
        if len(args) > 0:
            batch = args[0]
        else:
            batch = kwargs

        # If no optimizer, just run inference
        if self.optimizer is None:
            self.basemodel.eval()
            return self.adapter.execute_forward(self.basemodel, batch, self.device)

        # Adaptation step
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.adapter.execute_forward(self.basemodel, batch, self.device)

        # Extract adaptation loss (framework-specific)
        # This requires detailed implementation
        adapt_loss = self._compute_adaptation_loss(outputs)

        if adapt_loss and self._should_adapt(adapt_loss):
            total_loss = sum(adapt_loss.values())

            if total_loss > 0:
                total_loss.backward()

                # Gradient clipping
                if hasattr(self.basemodel, 'backbone'):
                    torch.nn.utils.clip_grad_norm_(
                        self.basemodel.backbone.parameters(), 1.0
                    )

                self.optimizer.step()
                self.used_steps += 1

            # Update loss EMA
            if 'global_align' in adapt_loss:
                self._update_loss_ema(adapt_loss['global_align'].item())

        self.adaptation_steps += 1

        return outputs

    def _compute_adaptation_loss(self, outputs) -> Dict[str, torch.Tensor]:
        """
        Compute adaptation loss (framework and method specific)

        This requires:
        1. Feature extraction
        2. Statistics computation
        3. KL divergence calculation

        Returns:
            Dictionary of losses
        """
        # Placeholder - requires full implementation
        return {}

    def _should_adapt(self, adapt_loss: Dict[str, torch.Tensor]) -> bool:
        """Determine whether to perform adaptation based on redundancy skipping"""
        if self.cfg.skip_redundant is None:
            return True

        if 'global_align' not in adapt_loss:
            return True

        loss_value = adapt_loss['global_align'].item()

        # Period-based skipping
        if 'period' in self.cfg.skip_redundant:
            if self.adaptation_steps % self.cfg.skip_period == 0:
                return True

        # EMA-based skipping
        if 'ema' in self.cfg.skip_redundant and self.loss_ema99 > 0:
            ema_ratio = loss_value / (self.loss_ema99 + 1e-7)
            if ema_ratio > self.cfg.skip_beta:
                return True

        return False

    def _update_loss_ema(self, loss_value: float):
        """Update loss EMA for redundancy skipping"""
        self.loss_ema99 = 0.99 * self.loss_ema99 + 0.01 * loss_value
        self.loss_ema95 = 0.95 * self.loss_ema95 + 0.05 * loss_value
        self.loss_ema90 = 0.9 * self.loss_ema90 + 0.1 * loss_value

    def online(self, mode=True):
        """Enable/disable online adaptation mode"""
        self.adapting = mode
        return self

    def offline(self):
        """Disable online adaptation mode"""
        return self.online(False)


# Note: Full WHW implementation requires migrating:
# 1. ParallelAdapter and ConvTaskWrapper classes
# 2. ResNet block patching logic
# 3. Feature statistics collection (both global and foreground)
# 4. KL divergence computation
# 5. ROI-based foreground feature extraction
#
# This is highly Detectron2-specific and requires significant effort to generalize.
# For now, users should refer to baseline.py for full Detectron2 implementation.

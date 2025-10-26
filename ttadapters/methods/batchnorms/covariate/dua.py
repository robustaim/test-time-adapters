"""
DUA (Dynamic Update Adaptation) Engine

Framework-agnostic implementation using adapter pattern.
Adapts normalization layer statistics dynamically with exponential decay.
"""

import copy
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ...base import AdaptationEngine, AdaptationConfig
from ...framework_adapters import FrameworkAdapter, create_adapter


@dataclass
class DUAConfig(AdaptationConfig):
    """Configuration for DUA engine"""
    adaptation_name: str = "DUA"

    # Adaptation settings
    adaptation_layers: str = "backbone+encoder"  # "backbone", "encoder", "backbone+encoder"

    # DUA-specific parameters
    min_momentum_constant: float = 0.0001
    decay_factor: float = 0.94
    mom_pre: float = 0.01

    # Device
    device: str = "cuda"


class DUAEngine(AdaptationEngine):
    """
    DUA: Dynamic Update Adaptation

    Dynamically updates batch normalization statistics with exponentially
    decaying momentum to adapt to distribution shifts.

    Framework-agnostic implementation that works with:
    - Detectron2 (Faster R-CNN, Mask R-CNN, etc.)
    - Transformers (RT-DETR, DETR, etc.)
    - Ultralytics (YOLO v8, v11, etc.)
    """

    model_name = "DUA"

    def __init__(
        self,
        basemodel: nn.Module,
        config: DUAConfig,
        adapter: Optional[FrameworkAdapter] = None
    ):
        super().__init__(basemodel, config)

        # Auto-detect framework adapter if not provided
        if adapter is None:
            adapter = create_adapter(basemodel)

        self.adapter = adapter
        self.cfg = config
        self.device = torch.device(config.device)

        # Setup
        self._setup()

    def _setup(self):
        """Initialize the DUA engine"""
        # Move model to device
        self.basemodel.to(self.device)

        # Apply DUA adaptation to normalization layers
        self._apply_dua_to_layers()

    def _apply_dua_to_layers(self):
        """Apply DUA adaptation to normalization layers"""
        # Create layer filter based on adaptation_layers setting
        layer_filter = self._create_layer_filter()

        # Identify normalization layers using adapter
        norm_layers = self.adapter.identify_normalization_layers(
            self.basemodel,
            layer_filter=layer_filter
        )

        print(f"DUAEngine: Applying DUA to {len(norm_layers)} normalization layers")

        # Apply DUA to each layer
        for name, module in norm_layers:
            self._patch_layer_with_dua(module)

    def _create_layer_filter(self):
        """Create layer filter based on adaptation_layers config"""
        if self.cfg.adaptation_layers == "backbone":
            if hasattr(self.adapter, 'create_layer_filter_backbone'):
                return self.adapter.create_layer_filter_backbone()
            else:
                return lambda name, module: 'backbone' in name.lower()

        elif self.cfg.adaptation_layers == "encoder":
            if hasattr(self.adapter, 'create_layer_filter_encoder'):
                return self.adapter.create_layer_filter_encoder()
            else:
                return lambda name, module: 'encoder' in name.lower()

        elif self.cfg.adaptation_layers == "backbone+encoder":
            if hasattr(self.adapter, 'create_layer_filter_backbone_and_encoder'):
                return self.adapter.create_layer_filter_backbone_and_encoder()
            else:
                return lambda name, module: 'decoder' not in name.lower()

        else:
            return None

    def _patch_layer_with_dua(self, module: nn.Module):
        """Patch a normalization layer to use DUA"""
        # Store DUA configuration
        module.adapt_type = "DUA"
        module.min_momentum_constant = self.cfg.min_momentum_constant
        module.decay_factor = self.cfg.decay_factor
        module.mom_pre = self.cfg.mom_pre

        # Store original statistics for potential reset
        if hasattr(module, 'running_mean') and not hasattr(module, 'original_running_mean'):
            module.original_running_mean = module.running_mean.clone()
            module.original_running_var = module.running_var.clone()

        # Replace forward method with DUA version
        original_forward = module.forward

        def dua_forward(x):
            """Forward with DUA adaptation"""
            if hasattr(module, 'adapt_type') and module.adapt_type == "DUA":
                with torch.no_grad():
                    # Compute current momentum
                    current_momentum = module.mom_pre + module.min_momentum_constant

                    # Compute batch statistics based on layer type
                    if isinstance(module, nn.BatchNorm2d):
                        batch_mean = x.mean(dim=[0, 2, 3])
                        batch_var = x.var(dim=[0, 2, 3], unbiased=True)

                    elif isinstance(module, nn.LayerNorm):
                        # For LayerNorm, compute over normalized dimensions
                        dims = tuple(range(-len(module.normalized_shape), 0))
                        batch_mean = x.mean(dim=dims, keepdim=True).squeeze()
                        batch_var = x.var(dim=dims, keepdim=True, unbiased=True).squeeze()

                    else:
                        # For other norm types (e.g., FrozenBatchNorm2d variants)
                        try:
                            batch_mean = x.mean(dim=[0, 2, 3])
                            batch_var = x.var(dim=[0, 2, 3], unbiased=True)
                        except:
                            # Fallback: don't update statistics
                            return self._standard_forward(module, x)

                    # Update running statistics with current momentum
                    if hasattr(module, 'running_mean') and module.running_mean is not None:
                        module.running_mean.mul_(1 - current_momentum).add_(
                            batch_mean, alpha=current_momentum
                        )
                        module.running_var.mul_(1 - current_momentum).add_(
                            batch_var, alpha=current_momentum
                        )

                    # Decay momentum for next iteration
                    module.mom_pre *= module.decay_factor

            # Standard normalization using (potentially updated) running stats
            return self._standard_forward(module, x)

        # Bind the new forward method
        module.forward = dua_forward.__get__(module, module.__class__)

    def _standard_forward(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Standard normalization forward pass"""
        if isinstance(module, nn.BatchNorm2d):
            # BatchNorm2d forward
            scale = module.weight * (module.running_var + module.eps).rsqrt()
            bias = module.bias - module.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype
            return x * scale.to(out_dtype) + bias.to(out_dtype)

        elif isinstance(module, nn.LayerNorm):
            # LayerNorm forward
            return nn.functional.layer_norm(
                x, module.normalized_shape, module.weight, module.bias, module.eps
            )

        else:
            # For FrozenBatchNorm2d and variants
            try:
                eps = getattr(module, 'eps', 1e-5)
                scale = module.weight * (module.running_var + eps).rsqrt()
                bias = module.bias - module.running_mean * scale
                scale = scale.reshape(1, -1, 1, 1)
                bias = bias.reshape(1, -1, 1, 1)
                out_dtype = x.dtype
                return x * scale.to(out_dtype) + bias.to(out_dtype)
            except Exception as e:
                warnings.warn(f"DUA forward failed for {type(module)}: {e}")
                return x

    def forward(self, *args, **kwargs):
        """
        Forward pass with DUA adaptation

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

        # Set model to eval mode (DUA updates happen in layer forward)
        self.basemodel.eval()

        # Forward pass through adapter
        outputs = self.adapter.execute_forward(self.basemodel, batch, self.device)

        return outputs

    def reset_momentum(self, mom_pre: Optional[float] = None):
        """
        Reset DUA momentum for new task/domain

        Args:
            mom_pre: New initial momentum (uses config default if None)
        """
        if mom_pre is None:
            mom_pre = self.cfg.mom_pre

        for module in self.basemodel.modules():
            if hasattr(module, 'adapt_type') and module.adapt_type == "DUA":
                module.mom_pre = mom_pre

                # Optionally reset to original statistics
                if hasattr(module, 'original_running_mean'):
                    module.running_mean = module.original_running_mean.clone()
                    module.running_var = module.original_running_var.clone()

    def online(self, mode=True):
        """Enable/disable online adaptation mode"""
        self.adapting = mode
        return self

    def offline(self):
        """Disable online adaptation mode"""
        return self.online(False)

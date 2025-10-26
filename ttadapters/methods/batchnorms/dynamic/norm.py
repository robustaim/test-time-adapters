"""
NORM (Normalization) Engine

Framework-agnostic implementation using adapter pattern.
Blends source and target statistics based on batch size.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ...base import AdaptationEngine, AdaptationConfig
from ...framework_adapters import FrameworkAdapter, create_adapter


@dataclass
class NORMConfig(AdaptationConfig):
    """Configuration for NORM engine"""
    adaptation_name: str = "NORM"

    # Adaptation settings
    adaptation_layers: str = "backbone+encoder"  # "backbone", "encoder", "backbone+encoder"

    # NORM-specific parameters
    source_sum: int = 128  # Number of source samples for weighting

    # Device
    device: str = "cuda"


class NORMEngine(AdaptationEngine):
    """
    NORM: Normalization with Source-Target Blending

    Blends source (training) and target (test) batch normalization statistics
    based on the relative batch size.

    Framework-agnostic implementation that works with:
    - Detectron2 (Faster R-CNN, Mask R-CNN, etc.)
    - Transformers (RT-DETR, DETR, etc.)
    - Ultralytics (YOLO v8, v11, etc.)
    """

    model_name = "NORM"

    def __init__(
        self,
        basemodel: nn.Module,
        config: NORMConfig,
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
        """Initialize the NORM engine"""
        # Move model to device
        self.basemodel.to(self.device)

        # Apply NORM adaptation to normalization layers
        self._apply_norm_to_layers()

    def _apply_norm_to_layers(self):
        """Apply NORM adaptation to normalization layers"""
        # Create layer filter based on adaptation_layers setting
        layer_filter = self._create_layer_filter()

        # Identify normalization layers using adapter
        norm_layers = self.adapter.identify_normalization_layers(
            self.basemodel,
            layer_filter=layer_filter
        )

        print(f"NORMEngine: Applying NORM to {len(norm_layers)} normalization layers")

        # Apply NORM to each layer
        for name, module in norm_layers:
            self._patch_layer_with_norm(module)

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

    def _patch_layer_with_norm(self, module: nn.Module):
        """Patch a normalization layer to use NORM"""
        # Store NORM configuration
        module.adapt_type = "NORM"
        module.source_sum = self.cfg.source_sum

        # Replace forward method with NORM version
        original_forward = module.forward

        def norm_forward(x):
            """Forward with NORM adaptation"""
            if hasattr(module, 'adapt_type') and module.adapt_type == "NORM":
                # Compute blending weight based on batch size
                alpha = x.shape[0] / (module.source_sum + x.shape[0])

                # Compute blended statistics based on layer type
                if isinstance(module, nn.BatchNorm2d):
                    # Compute batch statistics
                    batch_mean = x.mean(dim=[0, 2, 3])
                    batch_var = x.var(dim=[0, 2, 3])

                    # Blend with source statistics
                    running_mean = (1 - alpha) * module.running_mean + alpha * batch_mean
                    running_var = (1 - alpha) * module.running_var + alpha * batch_var

                    # Normalize
                    scale = module.weight * (running_var + module.eps).rsqrt()
                    bias = module.bias - running_mean * scale
                    scale = scale.reshape(1, -1, 1, 1)
                    bias = bias.reshape(1, -1, 1, 1)

                    out_dtype = x.dtype
                    return x * scale.to(out_dtype) + bias.to(out_dtype)

                elif isinstance(module, nn.LayerNorm):
                    # LayerNorm: use standard operation (NORM typically for spatial norms)
                    return nn.functional.layer_norm(
                        x, module.normalized_shape, module.weight, module.bias, module.eps
                    )

                else:
                    # For FrozenBatchNorm2d and variants
                    try:
                        batch_mean = x.mean(dim=[0, 2, 3])
                        batch_var = x.var(dim=[0, 2, 3])

                        running_mean = (1 - alpha) * module.running_mean + alpha * batch_mean
                        running_var = (1 - alpha) * module.running_var + alpha * batch_var

                        eps = getattr(module, 'eps', 1e-5)
                        scale = module.weight * (running_var + eps).rsqrt()
                        bias = module.bias - running_mean * scale
                        scale = scale.reshape(1, -1, 1, 1)
                        bias = bias.reshape(1, -1, 1, 1)

                        out_dtype = x.dtype
                        return x * scale.to(out_dtype) + bias.to(out_dtype)
                    except Exception as e:
                        warnings.warn(f"NORM forward failed for {type(module)}: {e}")
                        return self._standard_forward(module, x)

            else:
                # Standard forward without adaptation
                return self._standard_forward(module, x)

        # Bind the new forward method
        module.forward = norm_forward.__get__(module, module.__class__)

    def _standard_forward(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Standard normalization forward pass"""
        if isinstance(module, nn.BatchNorm2d):
            scale = module.weight * (module.running_var + module.eps).rsqrt()
            bias = module.bias - module.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype
            return x * scale.to(out_dtype) + bias.to(out_dtype)

        elif isinstance(module, nn.LayerNorm):
            return nn.functional.layer_norm(
                x, module.normalized_shape, module.weight, module.bias, module.eps
            )

        else:
            try:
                eps = getattr(module, 'eps', 1e-5)
                scale = module.weight * (module.running_var + eps).rsqrt()
                bias = module.bias - module.running_mean * scale
                scale = scale.reshape(1, -1, 1, 1)
                bias = bias.reshape(1, -1, 1, 1)
                out_dtype = x.dtype
                return x * scale.to(out_dtype) + bias.to(out_dtype)
            except Exception as e:
                warnings.warn(f"Standard forward failed for {type(module)}: {e}")
                return x

    def forward(self, *args, **kwargs):
        """
        Forward pass with NORM adaptation

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

        # Set model to eval mode (NORM updates happen in layer forward)
        self.basemodel.eval()

        # Forward pass through adapter
        outputs = self.adapter.execute_forward(self.basemodel, batch, self.device)

        return outputs

    def online(self, mode=True):
        """Enable/disable online adaptation mode"""
        self.adapting = mode
        return self

    def offline(self):
        """Disable online adaptation mode"""
        return self.online(False)

"""
CascadedNorm: Input Transformation for Norm Statistics Alignment

Test-time adaptation that transforms input images to align with source BN/LN statistics.

Key Innovation:
    Instead of adapting norm layers, we adapt the INPUT to match what the frozen
    BN layers expect (their running_mean/var from source domain training).

Mathematical Foundation:
    1. Transform input: x̃ = T(x; θ)  where T is differentiable histogram stretching
    2. Forward through model: features pass through BN layers
    3. Compute batch statistics at each BN: μ_batch, σ²_batch
    4. Loss: L = Σ_i ||μ_batch^i - μ_source^i||² + ||σ²_batch^i - σ²_source^i||²
    5. Update θ via backprop (BN layers stay frozen)

Pipeline:
    [Input] → [Transform T(θ)] → [Frozen Model with BN] → [Output]
                    ↑
              Update via BN alignment loss

Advantages:
    1. Architecture-agnostic
    2. Only transform parameters are learned (adaptation stability)
    3. No source data needed (BN.running_mean/var contains source info)
"""

from typing import List
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ..base import AdaptationEngine, AdaptationConfig
from ...models.base import BaseModel


@dataclass
class CascadedNormConfig(AdaptationConfig):
    """Configuration for CascadedNorm."""
    adaptation_name: str = "CascadedNormEngine"
    adapt_lr: float = 1e-3

    param_regularization: float = 0.01
    temperature: float = 0.01


class DifferentiableHistogramStretcher(nn.Module):
    """Differentiable histogram stretching."""

    def __init__(self, temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature

    def soft_percentile(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Differentiable percentile approximation."""
        x_flat = x.flatten()
        n = x_flat.shape[0]
        p = p.to(x.device)

        idx = (p / 100.0) * (n - 1)
        indices = torch.arange(n, device=x.device, dtype=x.dtype)
        weights = F.softmax(-(indices - idx).abs() / (self.temperature * n), dim=0)

        sorted_x, _ = torch.sort(x_flat)
        return (weights * sorted_x).sum()

    def stretch_channel(self, channel, clip_low, clip_high, gamma):
        """Apply stretching to single channel with gamma correction."""
        low_val = self.soft_percentile(channel, clip_low)
        high_val = self.soft_percentile(channel, clip_high)

        scale = 50.0
        clipped = low_val + F.softplus((channel - low_val) * scale) / scale
        clipped = high_val - F.softplus((high_val - clipped) * scale) / scale

        range_val = high_val - low_val + 1e-6
        normalized = (clipped - low_val) / range_val

        gamma_corrected = torch.pow(normalized + 1e-6, gamma)

        return torch.clamp(gamma_corrected * 255.0, 0, 255)

    def forward(self, image, clip_low, clip_high, gamma):
        """Apply stretching to image with gamma correction."""
        C = image.shape[0]
        stretched = torch.zeros_like(image)

        for c in range(C):
            stretched[c] = self.stretch_channel(image[c], clip_low, clip_high, gamma)

        return stretched


class GammaTransform(nn.Module):
    """
    Learnable parameters for histogram stretching with gamma correction.

    Manages learnable parameters for histogram stretching:
    - clip_low: [0, 10] (percentile values)
    - clip_high: [90, 100] (percentile values)
    - gamma: [0.5, 2.0] (gamma correction factor)
    """

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config

        # Learnable parameters (raw, will be constrained in forward)
        self.clip_low = nn.Parameter(torch.tensor(2.0))
        self.clip_high = nn.Parameter(torch.tensor(98.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))  # Gamma correction

        self.stretcher = DifferentiableHistogramStretcher(config.temperature)

    def forward(self):
        """
        Get constrained parameters (percentile mode).

        Returns:
            tuple: (clip_low, clip_high, gamma)
        """
        # Percentile mode: output percentile values (0-100)
        clip_low = torch.sigmoid(self.clip_low) * 10         # [0, 10] percentile
        clip_high = 90 + torch.sigmoid(self.clip_high) * 10  # [90, 100] percentile

        # Gamma correction factor
        gamma = 0.5 + torch.sigmoid(self.gamma) * 1.5  # [0.5, 2.0]

        return clip_low, clip_high, gamma


class CascadedNorm(nn.Module):
    """
    CascadedNorm: Cascaded Input Distribution Normalization
        by Statistical Norm Layer Alignment

    Learns to normalize input pixel distribution through adaptive
    transformation with cascaded layer-wise supervision.

    V1 implementation uses percentile-based gamma clamp.

    Args:
        config: CascadedNormConfig with learning parameters
    """

    def __init__(self, config: CascadedNormConfig):
        super().__init__()

        self.config = config
        self.current_params = (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))

        # Initialize transform controller (gamma clamp only for V1)
        self.transform_controller = GammaTransform(config)

        # Norm layer tracking (will be populated by Engine)
        self.norm_layers: nn.ModuleList[nn.Module] = nn.ModuleList()
        self.source_means: List[torch.Tensor] = []
        self.source_vars: List[torch.Tensor] = []

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        self.current_params = self.transform_controller()
        return self.transform_controller.stretcher(
            img, *self.current_params  # Always percentile mode in V1
        )

    def online_parameters(self):
        """Get learnable parameters for optimization."""
        return self.transform_controller.parameters()

    def compute_alignment_loss(self) -> torch.Tensor:
        """
        Compute alignment loss between batch and source statistics.
        
        Original V1 implementation: per-layer MSE loss accumulation.
        """
        total_loss = torch.tensor(0.0, device=self.source_means[0].device)

        for i, (norm_layer, source_mean, source_var) in enumerate(
            zip(self.norm_layers, self.source_means, self.source_vars)
        ):
            if not hasattr(norm_layer, 'current_mean') or norm_layer.current_mean is None:
                continue

            batch_mean = norm_layer.current_mean
            batch_var = norm_layer.current_var

            src_mean = source_mean.to(batch_mean.device)
            src_var = source_var.to(batch_var.device)

            # For BN with multiple channels, average to scalar
            if batch_mean.ndim > 0:
                batch_mean = batch_mean.mean()
                batch_var = batch_var.mean()

            loss_mean = F.mse_loss(batch_mean, src_mean)
            loss_var = F.mse_loss(batch_var, src_var)

            total_loss = total_loss + loss_mean + loss_var

        return total_loss


class CascadedNormEngine(AdaptationEngine):
    """
    Cascaded Norm Adaptation Engine (V1)
    Injects CascadedNorm module into the base model.

    Supports:
    - nn.BatchNorm2d / nn.BatchNorm1d / FrozenBatchNorm2d
    - nn.LayerNorm

    The engine extracts normalization layers from the base model,
    overrides foward to capture batch statistics, and optimizes
    input transformation to align batch stats with source stats.

    Args:
        base_model: Pre-trained model
        config: CascadedNormConfig with learning parameters

    Example:
        >>> config = CascadedNormConfig()
        >>> adaptive_model = CascadedNormEngine(base_model, config)
        >>> output = adaptive_model(batch)
    """
    model_name: str = "CascadedNormEngine"

    def __init__(self, base_model: BaseModel, config: CascadedNormConfig):
        self.dist_norm: CascadedNorm  # will be initialized in _pre_init()
        self.dist_norm_state: dict
        self.config = config

        super().__init__(base_model, config)

    def _pre_init(self):
        # Transformation modules
        self.dist_norm = CascadedNorm(self.config)

    def _post_init(self):
        self.dist_norm.to(self.device)
        self.dist_norm_state = {key: value.cpu() for key, value in self.dist_norm.state_dict().items()}

        # Inject dist_norm into base model
        self._inject_dist_norm()

        # Extract norm layers from base model
        self._extract_norm_layers()

        if self.config.verbose:
            print(f"\n[CascadedNorm V1] Summary:")
            print(f"  Total layers: {len(self.dist_norm.norm_layers)}")
            for cls in self.dist_norm.norm_layers:
                print(f"      {cls.__class__.__name__}")

    def online_parameters(self):
        """Get learnable parameters for optimization."""
        return self.dist_norm.online_parameters()

    def _inject_dist_norm(self):
        """
        Inject dist_norm into the first submodule's forward.

        Wraps the first actual layer that processes images (e.g., backbone, conv1, stem)
        to ensure transformation is applied to the pure image tensor, avoiding
        provider-specific input format complexities in base_model.forward.
        """
        # Find first submodule (backbone, stem, conv1, patch_embed, etc.)
        first_module = self._find_first_image_module()

        if first_module is None:
            if self.config.verbose:
                print("[Warning] No suitable first module found, falling back to base_model.forward wrapping")
            # Fallback: wrap base_model.forward
            original_forward = self.base_model.forward
            def preprocessing_forward(x, *args, **kwargs):
                if isinstance(x, torch.Tensor) and x.ndim == 4:
                    x = self.dist_norm(x)
                return original_forward(x, *args, **kwargs)
            self.base_model.forward = preprocessing_forward
            return

        # Wrap the first module's forward
        original_forward = first_module.forward

        def preprocessing_forward(x, *args, **kwargs):
            # Apply dist_norm to image input
            x = self.dist_norm(x)
            return original_forward(x, *args, **kwargs)

        first_module.forward = preprocessing_forward

        if self.config.verbose:
            print(f"[CascadedNorm] Injected dist_norm into: {first_module.__class__.__name__}")

    def _find_first_image_module(self) -> nn.Module:
        """
        Find the first module that directly processes images.

        Searches for common patterns:
        1. Named modules: 'backbone', 'stem', 'conv1', 'patch_embed', 'model'
        2. First Conv2d or Linear layer
        3. First child module

        Returns:
            First module that likely receives image tensors, or None if not found
        """
        # Strategy 1: Check for common named attributes
        common_names = ["backbone", "stem", "conv1", "patch_embed", "features", "model"]
        for name in common_names:
            if hasattr(self.base_model, name):
                module = getattr(self.base_model, name)
                if isinstance(module, nn.Module):
                    return module

        # Strategy 2: Find first Conv2d or Linear
        for module in self.base_model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module is not self.base_model:
                return module

        # Strategy 3: First child
        try:
            first_child = next(self.base_model.children())
            if isinstance(first_child, nn.Module):
                return first_child
        except StopIteration:
            pass

        return None

    def _extract_norm_layers(self):
        """
        Find all normalization layers including FrozenBatchNorm.

        Extracts source statistics (running_mean/var) for alignment.
        """
        # Find Norms
        found = []
        for name, module in self.base_model.named_modules():
            module_type = type(module).__name__

            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) or "BatchNorm" in module_type:
                found.append((
                    name, "BN", module, module.running_mean.clone(), module.running_var.clone()
                ))
            elif isinstance(module, nn.LayerNorm):
                found.append((
                    name, "LN", module, torch.tensor(0.0), torch.tensor(1.0)
                ))

        # Wrapping (no filtering in V1, always use all layers)
        self._cascade_wrap(found)

        # Populate to CascadedNorm
        for _, _, module, running_mean, running_var in found:
            self.dist_norm.norm_layers.append(module)
            self.dist_norm.source_means.append(running_mean)
            self.dist_norm.source_vars.append(running_var)

    @staticmethod
    def _cascade_wrap(filtered: list[nn.Module]):
        class_cache = {}

        for name, module_type, module, running_mean, running_var in filtered:
            original_class = module.__class__

            if original_class not in class_cache:
                # Define wrapped forward
                if module_type == "BN":
                    def new_forward(_self, _input: torch.Tensor) -> torch.Tensor:
                        if _input.dim() == 4:  # (B, C, H, W)
                            dims = (0, 2, 3)
                        elif _input.dim() == 3:  # (B, C, L)
                            dims = (0, 2)
                        else:  # (B, C)
                            dims = (0,)
                        _self.current_mean = _input.mean(dim=dims)
                        _self.current_var = _input.var(dim=dims, unbiased=False)

                        return original_class.forward(_self, _input)
                elif module_type == "LN":
                    def new_forward(_self, _input: torch.Tensor) -> torch.Tensor:
                        if hasattr(module, "normalized_shape"):
                            dims = tuple(range(-len(module.normalized_shape), 0))
                            _self.current_mean = _input.mean(dim=dims)
                            _self.current_var = _input.var(dim=dims, unbiased=False)
                        else:
                            _self.current_mean = _input.mean()
                            _self.current_var = _input.var(unbiased=False)

                        return original_class.forward(_self, _input)

                # Create new class
                new_class = type("Cascaded"+original_class.__name__, (original_class,), {
                    "forward": new_forward
                })
                class_cache[original_class] = new_class
            else:  # from class cache
                new_class = class_cache[original_class]

            module.__class__ = new_class  # override class
            module.current_mean = torch.tensor(0.0)  # register stat variable
            module.current_var = torch.tensor(0.0)  # register stat variable

    def _reset_stats(self):
        """Initialize statistics tracking."""
        self._stats = {
            'losses': [],
            'alignment_losses': [],
            'transform_params': [],
            'config': {
                'num_layers': len(self.dist_norm.norm_layers),
            }
        }

    def reset(self, reset_stats=False):
        """Reset model to initial state."""
        self.dist_norm.load_state_dict(self.dist_norm_state)
        self.online(self.adapting)
        self.to(self.device)
        self.to(self.dtype)
        try:
            self.optimizer.zero_grad()
        except:
            pass
        if reset_stats:
            current_stats = self._stats
            self._reset_stats()
            return current_stats
        else:
            return None

    def forward(self, *args, **kwargs):
        # Zero gradients
        self.optimizer.zero_grad()

        # Forward
        result = self.base_model(*args, **kwargs)

        # Compute alignment loss (original V1 method)
        loss = self.dist_norm.compute_alignment_loss()
        self._stats['alignment_losses'].append(loss.item())

        # Backward and update
        loss.backward()
        self.optimizer.step()
        self._stats['losses'].append(loss.item())

        # Log transform parameters
        clip_low, clip_high, gamma = self.dist_norm.current_params
        self._stats['transform_params'].append({
            'clip_low': clip_low.item(),
            'clip_high': clip_high.item(),
            'gamma': gamma.item() if gamma.dim() == 0 else gamma.mean().item()
        })

        return result


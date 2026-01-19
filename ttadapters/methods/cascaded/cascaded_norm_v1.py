"""
CascadedNorm: Input Transformation for BN Statistics Alignment

Test-time adaptation that transforms input images to align with source BN statistics.

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
    1. Architecture-agnostic (works with any model with BN)
    2. BN layers stay completely frozen (use running_mean/var)
    3. Only transform parameters are learned (3 params)
    4. Single forward pass per step
    5. No source data needed (BN.running_mean/var contains source info)

Author: CascadedNorm Team
Date: 2026-01-13
Version: 2.0 (Input Transform Alignment)
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Literal

from ..base import AdaptationEngine, AdaptationConfig
from ...models.base import BaseModel


# =============================================================================
# Input Transformation Modules
# =============================================================================

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


class InputTransformController(nn.Module):
    """Learnable parameters for histogram stretching with gamma correction."""

    def __init__(self):
        super().__init__()
        self.clip_low = nn.Parameter(torch.tensor(2.0))
        self.clip_high = nn.Parameter(torch.tensor(98.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))  # Gamma correction

    def forward(self):
        """Get constrained parameters."""
        clip_low = torch.sigmoid(self.clip_low) * 10  # [0, 10]
        clip_high = 90 + torch.sigmoid(self.clip_high) * 10  # [90, 100]
        gamma = 0.5 + torch.sigmoid(self.gamma) * 1.5  # [0.5, 2.0]
        return clip_low, clip_high, gamma


# =============================================================================
# BN Statistics Management
# =============================================================================

class NormStatisticsExtractor:
    """Extract and cache BN/LN layer references and source statistics."""

    def __init__(self, model: nn.Module):
        self.norm_layers: List[nn.Module] = []
        self.norm_types: List[str] = []  # 'bn' or 'ln'
        self.source_means: List[torch.Tensor] = []
        self.source_vars: List[torch.Tensor] = []

        self._extract_norm_layers(model)

    def _extract_norm_layers(self, model: nn.Module):
        """Find all BN/LN layers."""
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d) or "BatchNorm2d" in module.__class__.__name__:
                self.norm_layers.append(module)
                self.norm_types.append('bn')
                # BN: Use running statistics (averaged over channels)
                self.source_means.append(module.running_mean.mean().clone())
                self.source_vars.append(module.running_var.mean().clone())
                print(f"  [BN] Found: {name} ({module.num_features} channels)")

            elif isinstance(module, nn.LayerNorm) or "LayerNorm" in module.__class__.__name__:
                self.norm_layers.append(module)
                self.norm_types.append('ln')
                # LN: Target normalized distribution (mean=0, var=1)
                self.source_means.append(torch.tensor(0.0))
                self.source_vars.append(torch.tensor(1.0))
                print(f"  [LN] Found: {name}")

    def compute_alignment_loss(self) -> torch.Tensor:
        """Compute alignment loss between batch and source statistics."""
        total_loss = torch.tensor(0.0, device=self.source_means[0].device)

        for i, (norm_layer, norm_type) in enumerate(zip(self.norm_layers, self.norm_types)):
            if not hasattr(norm_layer, '_batch_mean') or norm_layer._batch_mean is None:
                continue

            batch_mean = norm_layer._batch_mean
            batch_var = norm_layer._batch_var

            source_mean = self.source_means[i].to(batch_mean.device)
            source_var = self.source_vars[i].to(batch_var.device)

            # For BN with multiple channels, average to scalar
            if norm_type == 'bn' and batch_mean.ndim > 0:
                batch_mean = batch_mean.mean()
                batch_var = batch_var.mean()

            loss_mean = F.mse_loss(batch_mean, source_mean)
            loss_var = F.mse_loss(batch_var, source_var)

            total_loss = total_loss + loss_mean + loss_var

        return total_loss


class NormStatisticsHook:
    """Hook to collect batch statistics from BN/LN layers."""

    def __init__(self, norm_types: List[str]):
        """
        Args:
            norm_types: List of 'bn' or 'ln' for each norm layer
        """
        self.hooks = []
        self.norm_types = norm_types

    def register_hooks(self, norm_layers: List[nn.Module]):
        """Register forward hooks."""
        for i, norm_layer in enumerate(norm_layers):
            norm_type = self.norm_types[i]
            hook = norm_layer.register_forward_hook(
                lambda module, input, output, nt=norm_type: self._hook_fn(module, input, output, nt)
            )
            self.hooks.append(hook)

    def _hook_fn(self, module, input, output, norm_type):
        """Capture batch statistics."""
        x = input[0]

        if norm_type == 'bn':
            # BatchNorm: (B, C, H, W) -> channel-wise statistics
            batch_mean = x.mean(dim=[0, 2, 3])  # [C]
            batch_var = x.var(dim=[0, 2, 3], unbiased=False)  # [C]
        else:  # 'ln'
            # LayerNorm: global statistics
            batch_mean = x.mean()  # scalar
            batch_var = x.var(unbiased=False)  # scalar

        module._batch_mean = batch_mean
        module._batch_var = batch_var

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# =============================================================================
# Configuration and Engine
# =============================================================================

@dataclass
class CascadedNormConfig(AdaptationConfig):
    """Configuration for CascadedNorm."""

    adaptation_name: str = "CascadedNorm"
    temperature: float = 0.01
    param_regularization: float = 0.01
    optim: Literal["SGD", "Adam"] = "SGD"
    adapt_lr: float = 1e-3


class CascadedNormEngine(AdaptationEngine):
    """
    CascadedNorm: Input transformation for BN/LN alignment.

    Transforms input to match norm layer source statistics.
    Works with both BatchNorm and LayerNorm.
    """

    model_name: str = "CascadedNorm"

    def __init__(self, base_model: BaseModel, config: CascadedNormConfig):
        super().__init__(base_model, config)

        self.config = config

        # Input transformation
        self.transform_controller = InputTransformController()
        self.stretcher = DifferentiableHistogramStretcher(config.temperature)

        # Norm extraction (both BN and LN)
        print(f"[CascadedNorm] Extracting norm layers...")
        self.norm_extractor = NormStatisticsExtractor(self.base_model)
        print(f"[CascadedNorm] Found {len(self.norm_extractor.norm_layers)} norm layers "
              f"(BN: {self.norm_extractor.norm_types.count('bn')}, "
              f"LN: {self.norm_extractor.norm_types.count('ln')})")

        # Hook manager
        self.hook_manager = NormStatisticsHook(self.norm_extractor.norm_types)
        self.hook_manager.register_hooks(self.norm_extractor.norm_layers)

        # Stats
        self._stats = {'alignment_losses': [], 'transform_params': []}

    def online_parameters(self):
        """Only transformation parameters."""
        return self.transform_controller.parameters()

    def _transform_image(self, img):
        """Transform single image with gamma correction."""
        clip_low, clip_high, gamma = self.transform_controller()
        transformed = self.stretcher(img, clip_low, clip_high, gamma)
        return transformed, (clip_low, clip_high, gamma)

    def _transform_batch(self, imgs):
        """Transform batch."""
        transformed_list = []
        params_list = []

        for i in range(imgs.shape[0]):
            transformed, params = self._transform_image(imgs[i])
            transformed_list.append(transformed)
            params_list.append(params)

        return torch.stack(transformed_list, dim=0), params_list

    def _compute_regularization_loss(self):
        """L2 regularization."""
        reg_loss = torch.tensor(0.0, device=self._device)
        for param in self.transform_controller.parameters():
            reg_loss = reg_loss + param.pow(2).sum()
        return self.config.param_regularization * reg_loss

    def forward(self, batched_inputs):
        """Forward with transformation and alignment."""
        if not self.adapting:
            return self.base_model(batched_inputs)

        if isinstance(batched_inputs, torch.Tensor):
            return self._forward_tensor(batched_inputs)
        return self._forward_dict_list(batched_inputs)

    def _forward_tensor(self, imgs):
        """Handle tensor input."""
        imgs = imgs.to(self._device)

        original_scale = imgs.max() <= 1.0
        if original_scale:
            imgs = imgs * 255.0

        imgs_transformed, params_list = self._transform_batch(imgs)

        for params in params_list:
            self._stats['transform_params'].append(tuple(p.item() for p in params))

        model_input = imgs_transformed / 255.0 if original_scale else imgs_transformed
        outputs = self.base_model(model_input)

        alignment_loss = self.norm_extractor.compute_alignment_loss()
        reg_loss = self._compute_regularization_loss()
        total_loss = alignment_loss + reg_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._stats['alignment_losses'].append(total_loss.item())

        return outputs

    def _forward_dict_list(self, batched_inputs):
        """Handle list of dicts."""
        transformed_inputs = []

        for input_dict in batched_inputs:
            if 'image' not in input_dict:
                transformed_inputs.append(input_dict)
                continue

            img = input_dict['image'].to(self._device)
            original_scale = img.max() <= 1.0
            if original_scale:
                img = img * 255.0

            img_transformed, params = self._transform_image(img)
            self._stats['transform_params'].append(tuple(p.item() for p in params))

            new_input = input_dict.copy()
            new_input['image'] = img_transformed / 255.0 if original_scale else img_transformed
            transformed_inputs.append(new_input)

        outputs = self.base_model(transformed_inputs)

        alignment_loss = self.norm_extractor.compute_alignment_loss()
        reg_loss = self._compute_regularization_loss()
        total_loss = alignment_loss + reg_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._stats['alignment_losses'].append(total_loss.item())

        return outputs

    def reset(self, reset_stats=False):
        """Reset adaptation."""
        super().reset(reset_stats=reset_stats)
        self.transform_controller = InputTransformController().to(self._device)
        self._optimizer = None

    @property
    def stats(self):
        """Get statistics."""
        if not self._stats['alignment_losses']:
            return None

        params_array = np.array(self._stats['transform_params'])

        return {
            'num_steps': len(self._stats['alignment_losses']),
            'mean_loss': np.mean(self._stats['alignment_losses']),
            'final_loss': self._stats['alignment_losses'][-1],
            'mean_clip_low': np.mean(params_array[:, 0]),
            'mean_clip_high': np.mean(params_array[:, 1]),
            'mean_gamma': np.mean(params_array[:, 2]),
        }

    def to(self, *args, **kwargs):
        """Move to device."""
        result = super().to(*args, **kwargs)
        self.transform_controller = self.transform_controller.to(self._device)
        self.stretcher = self.stretcher.to(self._device)
        return result

    def __del__(self):
        """Cleanup hooks."""
        if hasattr(self, 'hook_manager'):
            self.hook_manager.remove_hooks()

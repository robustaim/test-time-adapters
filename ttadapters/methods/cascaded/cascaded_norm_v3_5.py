"""
CascadedNorm v3.5: Global Learnable Gating & Temperature

Based on v3 (Fixed 50%), this version helps to investigate if 
"per-sample" adaptation is really needed or if "global" adaptation 
(domain-specific constant) is sufficient.

Key Features:
- Global learnable `gating` (blending ratio)
- Global learnable `temperature` (histogram stretching sharpness)
- Global learnable `gamma`, `clip_low`, `clip_high` (from v3)
- No per-sample inference (unlike v2.5 or v4)
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
    """Differentiable histogram stretching with dynamic temperature."""

    def __init__(self, temperature: float = 0.01):
        super().__init__()
        self.default_temperature = temperature

    def soft_percentile(self, x: torch.Tensor, p: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        """Differentiable percentile approximation."""
        x_flat = x.flatten()
        n = x_flat.shape[0]
        p = p.to(x.device)
        
        # Ensure temperature is positive and reasonable
        t = temperature if temperature is not None else self.default_temperature

        idx = (p / 100.0) * (n - 1)
        indices = torch.arange(n, device=x.device, dtype=x.dtype)
        weights = F.softmax(-(indices - idx).abs() / (t * n), dim=0)

        sorted_x, _ = torch.sort(x_flat)
        return (weights * sorted_x).sum()

    def stretch_channel(self, channel, clip_low, clip_high, gamma, temperature):
        """Apply stretching to single channel with gamma correction."""
        low_val = self.soft_percentile(channel, clip_low, temperature)
        high_val = self.soft_percentile(channel, clip_high, temperature)

        scale = 50.0
        clipped = low_val + F.softplus((channel - low_val) * scale) / scale
        clipped = high_val - F.softplus((high_val - clipped) * scale) / scale

        range_val = high_val - low_val + 1e-6
        normalized = (clipped - low_val) / range_val

        gamma_corrected = torch.pow(normalized + 1e-6, gamma)

        return torch.clamp(gamma_corrected * 255.0, 0, 255)

    def forward(self, image, clip_low, clip_high, gamma, temperature):
        """Apply stretching to image with gamma correction."""
        C = image.shape[0]
        stretched = torch.zeros_like(image)

        for c in range(C):
            stretched[c] = self.stretch_channel(image[c], clip_low, clip_high, gamma, temperature)

        return stretched


class GammaTransform(nn.Module):
    """Learnable GLOBAL parameters for histogram stretching, temperature, and gating."""

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        # 1. Clip limits
        self.clip_low = nn.Parameter(torch.tensor(2.0))
        self.clip_high = nn.Parameter(torch.tensor(98.0))
        
        # 2. Gamma correction
        self.gamma = nn.Parameter(torch.tensor(1.0))
        
        # 3. Gating (Blending ratio)
        # Initialize to 0.0 (sigmoid(0.0) = 0.5) to matching v3 baseline
        self.gating_logit = nn.Parameter(torch.tensor(0.0))
        
        # 4. Temperature (Sharpness)
        # Initialize to log(0.01) to match v3 baseline
        self.log_temp = nn.Parameter(torch.tensor(np.log(config.temperature)))

        # Integrated stretcher
        self.stretcher = DifferentiableHistogramStretcher(config.temperature)

    def forward(self):
        """Get constrained parameters."""
        # 1. Clip
        clip_low = torch.sigmoid(self.clip_low) * 10  # [0, 10]
        clip_high = 90 + torch.sigmoid(self.clip_high) * 10  # [90, 100]
        
        # 2. Gamma
        gamma = 0.5 + torch.sigmoid(self.gamma) * 1.5  # [0.5, 2.0]
        
        # 3. Gating
        gating = torch.sigmoid(self.gating_logit)  # [0, 1]
        
        # 4. Temperature
        temperature = torch.exp(self.log_temp).clamp(1e-4, 0.1)  # [0.0001, 0.1]
        
        return clip_low, clip_high, gamma, gating, temperature


class CascadedNorm(nn.Module):
    """
    CascadedNorm v3.5: Manages transformation and norm layer statistics.
    """

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config

        # Transform controller with integrated stretcher
        self.transform_controller = GammaTransform(config)

        # Norm layer tracking
        self.norm_layers: List[nn.Module] = []
        self.norm_types: List[str] = []
        self.source_means: List[torch.Tensor] = []
        self.source_vars: List[torch.Tensor] = []

    def forward(self, img):
        """Transform single image with learned global parameters."""
        # Retrieve global parameters
        clip_low, clip_high, gamma, gating, temperature = self.transform_controller()
        
        # Transform (Histogram Stretching)
        transformed = self.transform_controller.stretcher(
            img, clip_low, clip_high, gamma, temperature
        )

        # Blending (Gated)
        output = gating * transformed + (1 - gating) * img

        return output, (clip_low, clip_high, gamma, gating, temperature)

    def compute_alignment_loss(self) -> torch.Tensor:
        """Compute alignment loss between batch and source statistics."""
        total_loss = torch.tensor(0.0, device=self.source_means[0].device)

        for i, (norm_layer, norm_type) in enumerate(zip(self.norm_layers, self.norm_types)):
            if not hasattr(norm_layer, 'current_mean') or norm_layer.current_mean is None:
                continue

            batch_mean = norm_layer.current_mean
            batch_var = norm_layer.current_var

            source_mean = self.source_means[i].to(batch_mean.device)
            source_var = self.source_vars[i].to(batch_var.device)

            # For BN with multiple channels, average to scalar
            if norm_type == 'BN' and batch_mean.ndim > 0:
                batch_mean = batch_mean.mean()
                batch_var = batch_var.mean()

            loss_mean = F.mse_loss(batch_mean, source_mean)
            loss_var = F.mse_loss(batch_var, source_var)

            total_loss = total_loss + loss_mean + loss_var

        return total_loss

    def online_parameters(self):
        """Get learnable parameters for optimization."""
        return self.transform_controller.parameters()


class CascadedNormEngine(AdaptationEngine):
    """
    CascadedNorm v3.5 Engine
    """
    model_name: str = "CascadedNormEngine"

    def __init__(self, base_model: BaseModel, config: CascadedNormConfig):
        self.cascaded_norm: CascadedNorm
        self.cascaded_norm_state: dict
        self.config = config

        super().__init__(base_model, config)

    def _pre_init(self):
        self.cascaded_norm = CascadedNorm(self.config)

    def _post_init(self):
        self.cascaded_norm.to(self.device)
        self.cascaded_norm_state = {key: value.cpu() for key, value in self.cascaded_norm.state_dict().items()}
        self._extract_norm_layers()
        self._stats = {'alignment_losses': [], 'transform_params': []}

    def _extract_norm_layers(self):
        print(f"[CascadedNorm] Extracting norm layers...")
        found = []
        for name, module in self.base_model.named_modules():
            module_type = type(module).__name__

            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) or "BatchNorm" in module_type:
                found.append((
                    name, "BN", module,
                    module.running_mean.mean().clone(),
                    module.running_var.mean().clone()
                ))
            elif isinstance(module, nn.LayerNorm) or "LayerNorm" in module_type:
                found.append((
                    name, "LN", module,
                    torch.tensor(0.0),
                    torch.tensor(1.0)
                ))
        
        filtered = self._filter_by_cascade_mode(found)
        self._cascade_wrap(filtered)
        
        for _, norm_type, module, running_mean, running_var in filtered:
            self.cascaded_norm.norm_layers.append(module)
            self.cascaded_norm.norm_types.append(norm_type)
            self.cascaded_norm.source_means.append(running_mean)
            self.cascaded_norm.source_vars.append(running_var)

        print(f"[CascadedNorm] Found {len(self.cascaded_norm.norm_layers)} norm layers.")

    def _filter_by_cascade_mode(self, norm_list):
        if not hasattr(self.config, 'cascade_mode'): return norm_list
        match self.config.cascade_mode:
            case "single": return [norm_list[0]]
            case "single_last": return [norm_list[-1]]
            case "selected": return [norm_list[i] for i in getattr(self.config, 'cascade_indices', [])]
            case _: return norm_list

    @staticmethod
    def _cascade_wrap(filtered: list[nn.Module]):
        class_cache = {}
        for name, module_type, module, running_mean, running_var in filtered:
            original_class = module.__class__
            if original_class not in class_cache:
                if module_type == "BN":
                    def new_forward(_self, _input: torch.Tensor) -> torch.Tensor:
                        if _input.dim() == 4: dims = (0, 2, 3)
                        elif _input.dim() == 3: dims = (0, 2)
                        else: dims = (0,)
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
                new_class = type("Cascaded"+original_class.__name__, (original_class,), {"forward": new_forward})
                class_cache[original_class] = new_class
            else:
                new_class = class_cache[original_class]
            module.__class__ = new_class
            module.current_mean = torch.tensor(0.0)
            module.current_var = torch.tensor(0.0)

    def online_parameters(self):
        return self.cascaded_norm.online_parameters()

    def _transform_batch(self, imgs):
        transformed_list = []
        params_list = []
        for i in range(imgs.shape[0]):
            transformed, params = self.cascaded_norm(imgs[i])
            transformed_list.append(transformed)
            params_list.append(params)
        return torch.stack(transformed_list, dim=0), params_list

    def _compute_regularization_loss(self):
        reg_loss = torch.tensor(0.0, device=self._device)
        for param in self.cascaded_norm.transform_controller.parameters():
            reg_loss = reg_loss + param.pow(2).sum()
        return self.config.param_regularization * reg_loss

    def forward(self, batched_inputs):
        if not self.adapting:
            return self.base_model(batched_inputs)
        
        if isinstance(batched_inputs, torch.Tensor):
            return self._forward_tensor(batched_inputs)
        return self._forward_dict_list(batched_inputs)

    def _forward_tensor(self, imgs):
        imgs = imgs.to(self._device)
        original_scale = imgs.max() <= 1.0
        if original_scale: imgs = imgs * 255.0

        imgs_transformed, params_list = self._transform_batch(imgs)
        for params in params_list:
            self._stats['transform_params'].append(tuple(p.item() for p in params))

        model_input = imgs_transformed / 255.0 if original_scale else imgs_transformed
        outputs = self.base_model(model_input)

        alignment_loss = self.cascaded_norm.compute_alignment_loss()
        reg_loss = self._compute_regularization_loss()
        total_loss = alignment_loss + reg_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._stats['alignment_losses'].append(total_loss.item())
        return outputs

    def _forward_dict_list(self, batched_inputs):
        transformed_inputs = []
        for input_dict in batched_inputs:
            if 'image' not in input_dict:
                transformed_inputs.append(input_dict)
                continue
            
            img = input_dict['image'].to(self._device)
            original_scale = img.max() <= 1.0
            if original_scale: img = img * 255.0

            img_transformed, params = self.cascaded_norm(img)
            self._stats['transform_params'].append(tuple(p.item() for p in params))

            new_input = input_dict.copy()
            new_input['image'] = img_transformed / 255.0 if original_scale else img_transformed
            transformed_inputs.append(new_input)

        outputs = self.base_model(transformed_inputs)

        alignment_loss = self.cascaded_norm.compute_alignment_loss()
        reg_loss = self._compute_regularization_loss()
        total_loss = alignment_loss + reg_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._stats['alignment_losses'].append(total_loss.item())
        return outputs

    def reset(self, reset_stats=False):
        self.cascaded_norm.load_state_dict(self.cascaded_norm_state)
        self.online(self.adapting)
        self.to(self.device)
        self.to(self.dtype)
        try: self.optimizer.zero_grad()
        except: pass
        if reset_stats: self._stats = {'alignment_losses': [], 'transform_params': []}

    @property
    def stats(self):
        if not self._stats['alignment_losses']: return None
        params_array = np.array(self._stats['transform_params'])
        return {
            'num_steps': len(self._stats['alignment_losses']),
            'mean_loss': np.mean(self._stats['alignment_losses']),
            'final_loss': self._stats['alignment_losses'][-1],
            'mean_clip_low': np.mean(params_array[:, 0]),
            'mean_clip_high': np.mean(params_array[:, 1]),
            'mean_gamma': np.mean(params_array[:, 2]),
            'mean_gating': np.mean(params_array[:, 3]),
            'mean_temp': np.mean(params_array[:, 4]),
        }

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.cascaded_norm = self.cascaded_norm.to(self._device)
        return result

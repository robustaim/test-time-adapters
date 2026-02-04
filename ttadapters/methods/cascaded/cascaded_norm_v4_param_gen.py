"""
CascadedNorm V4: Parameter Generation (Clean V3 Base)

Re-implementation of V4 (Context-Aware) strictly based on the stable V3 codebase.
Replaces scalar parameters with a lightweight Generator (Policy) and adds Consistency Loss.
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
class CascadedNormV4Config(AdaptationConfig):
    """Configuration for CascadedNorm V4."""
    adaptation_name: str = "CascadedNormEngine"
    adapt_lr: float = 1e-4  # Lower LR for generator stability
    
    # Transformation
    temperature: float = 0.01
    saturation_limit: float = 100.0
    
    # Anchor / Consistency
    anchor_loss_weight: float = 10.0  # Increased 1.0 -> 10.0 to prevent drift/over-adaptation

    # Generator
    hidden_dim: int = 32
    
    # Optimizer Reset (Optional, inherited from V3 logic)
    optimizer_reset_interval: int = 30


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


class ParameterGenerator(nn.Module):
    """
    Lightweight CNN to predict transformation parameters from input image.
    Input: (B, C, H, W) -> Output: (B, 2) [noise_delta, gamma_delta]
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)), # Resize to small fixed size
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        # Zero init for head to start as Identity
        nn.init.constant_(self.head[-1].weight, 0.0)
        nn.init.constant_(self.head[-1].bias, 0.0)

    def forward(self, img):
        if img.dim() == 3: img = img.unsqueeze(0)
        if img.max() > 1.0: img = img / 255.0  # Normalize for stability
        
        return self.head(self.encoder(img))


class GammaTransform(nn.Module):
    """Learnable parameters (via Generator) for noise reduction."""
    noise_floor_init = 0.0  # Init to 0 (No clipping by default)
    gamma_init = 1.0        # Init to 1 (No gamma correction by default)

    def __init__(self, config: CascadedNormV4Config):
        super().__init__()
        self.saturation_limit = torch.tensor(config.saturation_limit, requires_grad=False)
        
        # Replaced scalar params with Generator
        self.generator = ParameterGenerator(config.hidden_dim)
        self.stretcher = DifferentiableHistogramStretcher(config.temperature)
        
        self.current_delta = None

    def forward(self, img):
        """Get constrained parameters and apply transform."""
        # Generator predicts raw logit, we apply Tanh scaling
        delta = self.generator(img)
        self.current_delta = delta
        
        avg_delta = delta.mean(dim=0) # (2,)
        
        # Tanh Scaling to prevent explosion
        # Delta 0 -> Noise +/- 50.0 (Wide enough to cover V3's 17.0)
        # Delta 1 -> Gamma +/- 0.9  (Wide enough to cover V3's 0.5)
        noise_delta = torch.tanh(avg_delta[0]) * 50.0 
        gamma_delta = torch.tanh(avg_delta[1]) * 0.9
        
        noise_floor = (self.noise_floor_init + noise_delta).clamp(0.0, 50.0) # Expanded Clamp
        gamma = (self.gamma_init + gamma_delta).clamp(0.1, 3.0) # Expanded Clamp
        
        if self.generator.training:
             print(f"[V4 Debug] Noise: {noise_floor.item():.2f}, Gamma: {gamma.item():.2f}")

        transformed = self.stretcher(img, noise_floor, self.saturation_limit, gamma)

        return transformed, (noise_floor, self.saturation_limit, gamma)

    def get_anchor_loss(self):
        """Penalty for deviating from identity (Delta=0)."""
        if self.current_delta is None:
            return torch.tensor(0.0)
        return self.current_delta.pow(2).mean()


class CascadedNorm(nn.Module):
    """
    CascadedNorm: Manages transformation and norm layer statistics.
    """

    def __init__(self, config: CascadedNormV4Config):
        super().__init__()
        self.config = config
        self.transform_controller = GammaTransform(config)
        self.norm_layers: List[nn.Module] = []
        self.norm_types: List[str] = []
        self.source_means: List[torch.Tensor] = []
        self.source_vars: List[torch.Tensor] = []

    def forward(self, img):
        transformed, params = self.transform_controller(img)
        # Residual Form
        output = 0.5 * transformed + 0.5 * img
        return output, params

    def compute_alignment_loss(self) -> torch.Tensor:
        """Compute alignment loss + Anchor Loss."""
        total_loss = torch.tensor(0.0, device=self.source_means[0].device if self.source_means else 'cpu')

        for i, (norm_layer, norm_type) in enumerate(zip(self.norm_layers, self.norm_types)):
            if not hasattr(norm_layer, 'current_mean') or norm_layer.current_mean is None:
                continue

            batch_mean = norm_layer.current_mean
            batch_var = norm_layer.current_var
            source_mean = self.source_means[i].to(batch_mean.device)
            source_var = self.source_vars[i].to(batch_var.device)

            if batch_mean.numel() > 1: batch_mean = batch_mean.mean()
            if batch_var.numel() > 1: batch_var = batch_var.mean()

            total_loss = total_loss + F.mse_loss(batch_mean, source_mean) + F.mse_loss(batch_var, source_var)

        # Anchor Loss
        if self.config.anchor_loss_weight > 0:
            anchor_loss = self.transform_controller.get_anchor_loss()
            total_loss = total_loss + anchor_loss * self.config.anchor_loss_weight

        return total_loss

    def online_parameters(self):
        return self.transform_controller.generator.parameters()


class CascadedNormEngine(AdaptationEngine):
    """
    CascadedNorm Engine (V3 Base).
    """
    model_name: str = "CascadedNormEngine"

    def __init__(self, base_model: BaseModel, config: CascadedNormV4Config):
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
        self.step_counter = 0

    def _maybe_reset_optimizer(self):
        if self.config.optimizer_reset_interval <= 0: return
        if self.step_counter > 0 and self.step_counter % self.config.optimizer_reset_interval == 0:
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    state = self.optimizer.state[p]
                    if 'exp_avg' in state: state['exp_avg'].zero_()

    def _extract_norm_layers(self):
        print(f"[CascadedNorm] Extracting norm layers...")
        found = []
        for name, module in self.base_model.named_modules():
            module_type = type(module).__name__
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) or "BatchNorm" in module_type:
                found.append((name, "BN", module, module.running_mean.mean().clone(), module.running_var.mean().clone()))
            elif isinstance(module, nn.LayerNorm) or "LayerNorm" in module_type:
                found.append((name, "LN", module, torch.tensor(0.0), torch.tensor(1.0)))
        
        filtered = self._filter_by_cascade_mode(found)
        self._cascade_wrap(filtered)
        
        for _, norm_type, module, running_mean, running_var in filtered:
            self.cascaded_norm.norm_layers.append(module)
            self.cascaded_norm.norm_types.append(norm_type)
            self.cascaded_norm.source_means.append(running_mean)
            self.cascaded_norm.source_vars.append(running_var)
            
        print(f"[CascadedNorm] Found {len(self.cascaded_norm.norm_layers)} norm layers")

    def _filter_by_cascade_mode(self, norm_list):
        if not hasattr(self.config, 'cascade_mode'): return norm_list
        match self.config.cascade_mode:
            case "single": return [norm_list[0]]
            case "single_last": return [norm_list[-1]]
            case "selected": return [norm_list[i] for i in getattr(self.config, 'cascade_indices', [])]
            case _: return norm_list

    @staticmethod
    def _create_wrapped_class(original_class, module_type):
        if module_type == "BN":
            def forward(_self, _input: torch.Tensor) -> torch.Tensor:
                if _input.dim() == 4: dims = (0, 2, 3)
                elif _input.dim() == 3: dims = (0, 2)
                else: dims = (0,)
                _self.current_mean = _input.mean(dim=dims)
                _self.current_var = _input.var(dim=dims, unbiased=False)
                return original_class.forward(_self, _input)
        elif module_type == "LN":
            def forward(_self, _input: torch.Tensor) -> torch.Tensor:
                if hasattr(_self, "normalized_shape"):
                    dims = tuple(range(-len(_self.normalized_shape), 0))
                    _self.current_mean = _input.mean(dim=dims)
                    _self.current_var = _input.var(dim=dims, unbiased=False)
                else:
                    _self.current_mean = _input.mean()
                    _self.current_var = _input.var(unbiased=False)
                return original_class.forward(_self, _input)
        else: return original_class
        
        return type(f"Cascaded{original_class.__name__}", (original_class,), {"forward": forward})

    @staticmethod
    def _cascade_wrap(filtered: list[nn.Module]):
        class_cache = {}
        for name, module_type, module, running_mean, running_var in filtered:
            original_class = module.__class__
            if original_class not in class_cache:
                new_class = CascadedNormEngine._create_wrapped_class(original_class, module_type)
                class_cache[original_class] = new_class
            else: new_class = class_cache[original_class]
            
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

    def forward(self, batched_inputs=None, **kwargs):
        is_kwargs = False
        if batched_inputs is None and kwargs:
            batched_inputs = kwargs
            is_kwargs = True

        if not self.adapting:
            if is_kwargs: return self.base_model(**batched_inputs)
            return self.base_model(batched_inputs)

        if isinstance(batched_inputs, torch.Tensor):
            return self._forward_tensor(batched_inputs)
        elif isinstance(batched_inputs, dict):
            return self._forward_dict(batched_inputs, unpack_args=is_kwargs)
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
        
        self.optimizer.zero_grad()
        alignment_loss.backward()
        self.optimizer.step()
        
        self.step_counter += 1
        self._maybe_reset_optimizer()
        self._stats['alignment_losses'].append(alignment_loss.item())

        return outputs

    def _forward_dict(self, input_dict, unpack_args=False):
        if 'pixel_values' in input_dict: img_key = 'pixel_values'
        elif 'img' in input_dict: img_key = 'img'
        else:
            if unpack_args: return self.base_model(**input_dict)
            return self.base_model(input_dict)

        imgs = input_dict[img_key].to(self._device)
        original_scale = imgs.max() <= 1.0
        if original_scale: imgs = imgs * 255.0

        imgs_transformed, params_list = self._transform_batch(imgs)

        for params in params_list:
            self._stats['transform_params'].append(tuple(p.item() for p in params))

        model_input = imgs_transformed / 255.0 if original_scale else imgs_transformed
        new_input = input_dict.copy()
        new_input[img_key] = model_input
        
        if unpack_args: outputs = self.base_model(**new_input)
        else: outputs = self.base_model(new_input)

        alignment_loss = self.cascaded_norm.compute_alignment_loss()
        
        self.optimizer.zero_grad()
        alignment_loss.backward()
        self.optimizer.step()
        
        self.step_counter += 1
        self._maybe_reset_optimizer()
        self._stats['alignment_losses'].append(alignment_loss.item())

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
        
        self.optimizer.zero_grad()
        alignment_loss.backward()
        self.optimizer.step()
        
        self.step_counter += 1
        self._maybe_reset_optimizer()
        self._stats['alignment_losses'].append(alignment_loss.item())

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
            'mean_noise_floor': np.mean(params_array[:, 0]),
            'mean_saturation_limit': np.mean(params_array[:, 1]),
            'mean_gamma': np.mean(params_array[:, 2]),
        }

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.cascaded_norm = self.cascaded_norm.to(self._device)
        return result

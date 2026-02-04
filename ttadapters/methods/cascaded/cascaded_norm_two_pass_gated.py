"""
CascadedNorm Two-Pass Gated: Adapt-then-Infer with Learnable Gating

Extends the Two-Pass (Reset) strategy by replacing the fixed residual connection
with a learnable Gating Network. This network decides "where" and "how much"
to apply the transformation, adapting uniquely to each sample.

Features:
1.  **Two-Pass Adaptation**: Adapt -> Infer -> Reset (Per-Sample Isolation).
2.  **Learnable Gating**:
    - 'pixel': Spatial Attention Map (B, 1, H, W). Good for local fog/shadows.
    - 'scalar': Global Mixture Weight (B, 1, 1, 1). Good for overall intensity.
3.  **Optimized**: Gating networks are lightweight to minimize adaptation overhead.
"""

from typing import List, Literal
from dataclasses import dataclass
import copy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ..base import AdaptationEngine, AdaptationConfig
from ...models.base import BaseModel


@dataclass
class CascadedNormGatedConfig(AdaptationConfig):
    """Configuration for CascadedNorm Two-Pass Gated."""
    adaptation_name: str = "CascadedNormEngine"
    adapt_lr: float = 1e-2
    
    # Transformation Params
    temperature: float = 0.01
    saturation_limit: float = 100.0
    param_regularization: float = 0.0
    
    # Two-Pass Params
    reset_every_batch: bool = True
    num_adaptation_steps: int = 1
    max_layers: int = 5
    
    # Gating Params
    gating_type: Literal['pixel', 'scalar'] = 'pixel'  # 'pixel' or 'scalar'
    gating_lr_mult: float = 10.0  # Multiplier for gating network LR (needs faster update than scalars)


class DifferentiableHistogramStretcher(nn.Module):
    """Differentiable histogram stretching (Standard V3)."""

    def __init__(self, temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature

    def soft_percentile(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x_flat = x.flatten()
        n = x_flat.shape[0]
        p = p.to(x.device)
        idx = (p / 100.0) * (n - 1)
        indices = torch.arange(n, device=x.device, dtype=x.dtype)
        weights = F.softmax(-(indices - idx).abs() / (self.temperature * n), dim=0)
        sorted_x, _ = torch.sort(x_flat)
        return (weights * sorted_x).sum()

    def stretch_channel(self, channel, clip_low, clip_high, gamma):
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
        C = image.shape[0]
        stretched = torch.zeros_like(image)
        for c in range(C):
            stretched[c] = self.stretch_channel(image[c], clip_low, clip_high, gamma)
        return stretched


class GatingNetwork(nn.Module):
    """
    Lightweight Gating Network.
    Decides the mixing ratio: Alpha * Transformed + (1-Alpha) * Original.
    """
    def __init__(self, mode: str = 'pixel'):
        super().__init__()
        self.mode = mode
        
        if mode == 'pixel':
            # Spatial Attention: (C, H, W) -> (1, H, W)
            # Simple 1x1 conv to mix channels and decide importance per pixel
            # We use a small reception field (3x3) to see local context
            self.net = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 1, kernel_size=1)
            )
        else:
            # Scalar Weight: (C, H, W) -> (1, 1, 1)
            # Global Average Pooling -> FC
            self.net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(3, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            )
        
        # Initialize to output ~0.0 (Sigmoid(0.0) = 0.5)
        # We start with a neutral 50:50 mixing.
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        # Force the final layer to produce close to 0
        if self.mode == 'pixel':
            nn.init.constant_(self.net[-1].bias, 0.0)
            nn.init.normal_(self.net[-1].weight, 0, 0.001)
        else:
            nn.init.constant_(self.net[-1].bias, 0.0)
            nn.init.normal_(self.net[-1].weight, 0, 0.001)

    def forward(self, x):
        # x is (B, 3, H, W) normalized or raw
        # If raw (0-255), normalize roughly for stability
        if x.max() > 1.0:
            x = x / 255.0
            
        out = self.net(x) # (B, 1, H, W) or (B, 1)
        
        if self.mode == 'scalar':
            out = out.view(-1, 1, 1, 1) # (B, 1, 1, 1)
            
        return torch.sigmoid(out) # Range 0~1


class GammaTransform(nn.Module):
    """Learnable parameters + Gating."""
    noise_floor_init = 2.0
    gamma_init = 1.0

    def __init__(self, config: CascadedNormGatedConfig):
        super().__init__()
        self.saturation_limit = torch.tensor(config.saturation_limit, requires_grad=False)
        self.noise_floor = nn.Parameter(torch.tensor(self.noise_floor_init))
        self.gamma = nn.Parameter(torch.tensor(self.gamma_init))

        self.stretcher = DifferentiableHistogramStretcher(config.temperature)
        self.gating_net = GatingNetwork(config.gating_type)

    def forward(self, img):
        noise_floor = self.noise_floor.clamp(min=0.0, max=20.0)
        gamma = self.gamma.clamp(min=0.5, max=2.0)

        # 1. Transform
        transformed_img = self.stretcher(img, noise_floor, self.saturation_limit, gamma)
        
        # 2. Compute Gating Alpha
        # Use detached image for gating or gradient flow?
        # We want to learn gating too, so gradient flow is needed.
        alpha = self.gating_net(img) # (1, H, W) or (1, 1, 1) per batch item logic handled in loop
        
        return transformed_img, alpha, (noise_floor, self.saturation_limit, gamma)

    def get_regularization_loss(self):
        return (self.noise_floor - self.noise_floor_init).pow(2) + (self.gamma - self.gamma_init).pow(2)


class CascadedNorm(nn.Module):
    def __init__(self, config: CascadedNormGatedConfig):
        super().__init__()
        self.config = config
        self.transform_controller = GammaTransform(config)
        self.norm_layers: List[nn.Module] = []
        self.norm_types: List[str] = []
        self.source_means: List[torch.Tensor] = []
        self.source_vars: List[torch.Tensor] = []

    def forward(self, img):
        # Transformed, Alpha, Params
        transformed, alpha, params = self.transform_controller(img)
        
        # Gated Mixing
        # Output = Alpha * Transformed + (1 - Alpha) * Original
        output = alpha * transformed + (1 - alpha) * img
        
        return output, params, alpha

    def compute_alignment_loss(self) -> torch.Tensor:
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

        if self.config.param_regularization > 0:
            reg_loss = self.transform_controller.get_regularization_loss()
            total_loss = total_loss + reg_loss * self.config.param_regularization

        return total_loss

    def online_parameters(self):
        # Differential LR for Gating vs Scalars
        return [
            {
                'params': [self.transform_controller.noise_floor, self.transform_controller.gamma],
                'lr': self.config.adapt_lr
            },
            {
                'params': self.transform_controller.gating_net.parameters(),
                'lr': self.config.adapt_lr * self.config.gating_lr_mult  # Gating needs faster adaptation
            }
        ]


class CascadedNormEngine(AdaptationEngine):
    model_name: str = "CascadedNormEngine"

    def __init__(self, base_model: BaseModel, config: CascadedNormGatedConfig):
        self.cascaded_norm: CascadedNorm
        self.cascaded_norm_state: dict
        self.config = config
        super().__init__(base_model, config)

    def _pre_init(self):
        self.cascaded_norm = CascadedNorm(self.config)

    def _post_init(self):
        self.cascaded_norm.to(self.device)
        self.cascaded_norm_state = {key: value.cpu().clone() for key, value in self.cascaded_norm.state_dict().items()}
        self._extract_norm_layers()
        self._stats = {'alignment_losses': [], 'transform_params': [], 'mean_alpha': []}
    
    def _extract_norm_layers(self):
        print(f"[CascadedNorm] Extracting norm layers (First {self.config.max_layers} only)...")
        found = []
        for name, module in self.base_model.named_modules():
            module_type = type(module).__name__
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) or "BatchNorm" in module_type:
                found.append((name, "BN", module, module.running_mean.mean().clone(), module.running_var.mean().clone()))
            elif isinstance(module, nn.LayerNorm) or "LayerNorm" in module_type:
                found.append((name, "LN", module, torch.tensor(0.0), torch.tensor(1.0)))
        
        filtered = self._filter_by_cascade_mode(found)
        if len(filtered) > self.config.max_layers:
            filtered = filtered[:self.config.max_layers]
            print(f"[CascadedNorm] Limiting to first {len(filtered)} layers for speed optimization.")

        self._cascade_wrap(filtered)

        for _, norm_type, module, running_mean, running_var in filtered:
            self.cascaded_norm.norm_layers.append(module)
            self.cascaded_norm.norm_types.append(norm_type)
            self.cascaded_norm.source_means.append(running_mean)
            self.cascaded_norm.source_vars.append(running_var)
            
        print(f"[CascadedNorm] Final tracked layers: {len(self.cascaded_norm.norm_layers)}")

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
                        if _self.training or hasattr(_self, 'force_stat_capture'):
                             _self.current_mean = _input.mean(dim=dims)
                             _self.current_var = _input.var(dim=dims, unbiased=False)
                        return original_class.forward(_self, _input)
                elif module_type == "LN":
                    def new_forward(_self, _input: torch.Tensor) -> torch.Tensor:
                        if hasattr(module, "normalized_shape"):
                            dims = tuple(range(-len(module.normalized_shape), 0))
                        else:
                            dims = None
                        if _self.training or hasattr(_self, 'force_stat_capture'):
                            if dims:
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
            module.force_stat_capture = True

    def online_parameters(self):
        return self.cascaded_norm.online_parameters()

    def _transform_batch(self, imgs):
        transformed_list = []
        params_list = []
        alpha_list = []
        
        for i in range(imgs.shape[0]):
            transformed, params, alpha = self.cascaded_norm(imgs[i])
            transformed_list.append(transformed)
            params_list.append(params)
            alpha_list.append(alpha.mean().item())
            
        return torch.stack(transformed_list, dim=0), params_list, alpha_list

    def forward(self, batched_inputs):
        if not self.adapting: return self.base_model(batched_inputs)

        # --- Step 0: Snapshot State ---
        initial_state = None
        if self.config.reset_every_batch:
            initial_state = {k: v.clone() for k, v in self.cascaded_norm.state_dict().items()}
        
        # Prepare inputs
        is_tensor = isinstance(batched_inputs, torch.Tensor)
        if is_tensor:
            imgs_ref = batched_inputs.to(self.device)
        else:
            imgs_ref = torch.stack([x['image'] for x in batched_inputs]).to(self.device)
        
        original_scale = imgs_ref.max() <= 1.0
        if original_scale: imgs_ref = imgs_ref * 255.0

        # --- Step 1: Adaptation Loop ---
        torch.set_grad_enabled(True)
        self.cascaded_norm.train()
        
        for step in range(self.config.num_adaptation_steps):
            imgs_transformed, params_list, alpha_list = self._transform_batch(imgs_ref)
            model_input = imgs_transformed / 255.0 if original_scale else imgs_transformed
            
            if is_tensor: _ = self.base_model(model_input)
            else:
                _inputs = self._pack_dict_list(batched_inputs, model_input, original_scale)
                _ = self.base_model(_inputs)
                
            loss = self.cascaded_norm.compute_alignment_loss()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self._stats['alignment_losses'].append(loss.item())

        # --- Step 2: Inference Pass ---
        torch.set_grad_enabled(False)
        self.cascaded_norm.eval()
        
        imgs_transformed, params_list, alpha_list = self._transform_batch(imgs_ref)
        model_input = imgs_transformed / 255.0 if original_scale else imgs_transformed
        
        if is_tensor: outputs = self.base_model(model_input)
        else:
            _inputs = self._pack_dict_list(batched_inputs, model_input, original_scale)
            outputs = self.base_model(_inputs)
            
        for params in params_list:
            self._stats['transform_params'].append(tuple(p.item() for p in params))
        self._stats['mean_alpha'].extend(alpha_list)

        # --- Step 3: Restore State ---
        if self.config.reset_every_batch and initial_state is not None:
            self.cascaded_norm.load_state_dict(initial_state)
            self.optimizer.state.clear()
            
        torch.set_grad_enabled(True) 
        return outputs

    def _pack_dict_list(self, original_inputs, transformed_imgs, original_scale):
        new_inputs = []
        for i, input_dict in enumerate(original_inputs):
            new_input = input_dict.copy()
            img = transformed_imgs[i]
            new_input['image'] = img
            new_inputs.append(new_input)
        return new_inputs

    def reset(self, reset_stats=False):
        self.cascaded_norm.load_state_dict(self.cascaded_norm_state)
        self.online(self.adapting)
        self.to(self.device)
        self.to(self.dtype)
        try:
            self.optimizer.zero_grad()
            self.optimizer.state.clear()
        except: pass
        if reset_stats:
            self._stats = {'alignment_losses': [], 'transform_params': [], 'mean_alpha': []}

    @property
    def stats(self):
        if not self._stats['alignment_losses']: return None
        params_array = np.array(self._stats['transform_params'])
        return {
            'num_steps': len(self._stats['alignment_losses']),
            'mean_loss': np.mean(self._stats['alignment_losses']),
            'final_loss': self._stats['alignment_losses'][-1] if self._stats['alignment_losses'] else 0,
            'mean_noise_floor': np.mean(params_array[:, 0]) if len(params_array) > 0 else 0,
            'mean_alpha': np.mean(self._stats['mean_alpha']) if self._stats['mean_alpha'] else 0,
        }

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.cascaded_norm = self.cascaded_norm.to(self._device)
        return result

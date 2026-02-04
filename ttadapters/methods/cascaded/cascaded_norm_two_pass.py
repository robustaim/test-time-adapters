"""
CascadedNorm Two-Pass: Test-Time Training (TTT) Version based on V3

Implements the "Adapt-then-Infer" strategy (Two-Pass Forward) to eliminate adaptation lag.
To prevent parameter drift across domains, this version supports 'Per-Sample' adaptation 
(resetting parameters after each inference).

Key Changes from V3:
1.  **Two-Pass Forward**:
    - Pass 1: Forward -> Loss -> Update Parameters
    - Pass 2: Forward (with updated params) -> Prediction
2.  **Early Layer Restriction**:
    - Only aligns statistics for the first 5 Normalization layers to improve speed.
3.  **Per-Sample Option**:
    - Can reset parameters to initialization after every batch to prevent drift.
"""

from typing import List
from dataclasses import dataclass
import copy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ..base import AdaptationEngine, AdaptationConfig
from ...models.base import BaseModel


@dataclass
class CascadedNormTwoPassConfig(AdaptationConfig):
    """Configuration for CascadedNorm Two-Pass."""
    adaptation_name: str = "CascadedNormEngine"
    adapt_lr: float = 1e-2  # Higher LR for one-shot adaptation
    
    # V3 Params
    temperature: float = 0.01
    saturation_limit: float = 100.0
    param_regularization: float = 0.0
    
    # Two-Pass Specific
    reset_every_batch: bool = True  # If True, resets params after every batch (Per-Sample Mode)
    num_adaptation_steps: int = 1   # How many gradient steps to take before inference
    max_layers: int = 5             # Limit alignment to first N layers for speed


class DifferentiableHistogramStretcher(nn.Module):
    """Differentiable histogram stretching (Same as V3)."""

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


class GammaTransform(nn.Module):
    """Learnable parameters (Same as V3)."""
    noise_floor_init = 2.0
    gamma_init = 1.0

    def __init__(self, config: CascadedNormTwoPassConfig):
        super().__init__()
        self.saturation_limit = torch.tensor(config.saturation_limit, requires_grad=False)
        self.noise_floor = nn.Parameter(torch.tensor(self.noise_floor_init))
        self.gamma = nn.Parameter(torch.tensor(self.gamma_init))

        self.stretcher = DifferentiableHistogramStretcher(config.temperature)

    def forward(self, img):
        noise_floor = self.noise_floor.clamp(min=0.0, max=20.0)
        gamma = self.gamma.clamp(min=0.5, max=2.0)

        transformed = self.stretcher(img, noise_floor, self.saturation_limit, gamma)
        return transformed, (noise_floor, self.saturation_limit, gamma)

    def get_regularization_loss(self):
        return (self.noise_floor - self.noise_floor_init).pow(2) + (self.gamma - self.gamma_init).pow(2)


class CascadedNorm(nn.Module):
    """CascadedNorm Manager (Same as V3)."""

    def __init__(self, config: CascadedNormTwoPassConfig):
        super().__init__()
        self.config = config
        self.transform_controller = GammaTransform(config)
        self.norm_layers: List[nn.Module] = []
        self.norm_types: List[str] = []
        self.source_means: List[torch.Tensor] = []
        self.source_vars: List[torch.Tensor] = []

    def forward(self, img):
        transformed, params = self.transform_controller(img)
        output = 0.5 * transformed + 0.5 * img
        return output, params

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
        return self.transform_controller.parameters()


class CascadedNormEngine(AdaptationEngine):
    """
    CascadedNorm Two-Pass Engine.
    Implements Adapt-then-Infer logic.
    """
    model_name: str = "CascadedNormEngine"

    def __init__(self, base_model: BaseModel, config: CascadedNormTwoPassConfig):
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
        self._stats = {'alignment_losses': [], 'transform_params': []}
    
    def _extract_norm_layers(self):
        print(f"[CascadedNorm] Extracting norm layers (First {self.config.max_layers} only)...")
        found = []
        for name, module in self.base_model.named_modules():
            module_type = type(module).__name__
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) or "BatchNorm" in module_type:
                found.append((name, "BN", module, module.running_mean.mean().clone(), module.running_var.mean().clone()))
            elif isinstance(module, nn.LayerNorm) or "LayerNorm" in module_type:
                found.append((name, "LN", module, torch.tensor(0.0), torch.tensor(1.0)))
        
        # 1. First, apply cascade mode filtering if specified (usually 'all' or default)
        filtered = self._filter_by_cascade_mode(found)
        
        # 2. **CRITICAL**: Limit to first N layers as requested for speed
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
        """Same wrapping logic as V3."""
        class_cache = {}
        for name, module_type, module, running_mean, running_var in filtered:
            original_class = module.__class__
            if original_class not in class_cache:
                if module_type == "BN":
                    def new_forward(_self, _input: torch.Tensor) -> torch.Tensor:
                        if _input.dim() == 4: dims = (0, 2, 3)
                        elif _input.dim() == 3: dims = (0, 2)
                        else: dims = (0,)
                        # Capture stats
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
            module.force_stat_capture = True # Enable capture

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

    def forward(self, batched_inputs):
        """
        Two-Pass Forward Logic:
        1. Snapshot state (if Per-Sample mode)
        2. Optimization Loop (Pass 1..N): Forward -> Loss -> Update
        3. Inference Pass: Forward with updated params
        4. Restore state (if Per-Sample mode)
        """
        if not self.adapting:
            return self.base_model(batched_inputs)

        # --- Step 0: Snapshot State (for restoration) ---
        initial_state = None
        if self.config.reset_every_batch:
            # We copy the state dict to restore later. 
            # Note: We must also consider optimizer state if we want true isolation,
            # but usually for simple SGD/Adam with 1 step, resetting weights is enough 
            # if we don't care about momentum history carrying over (which we don't for independent samples).
            # Actually, we should probably reset optimizer too or just accept momentum reset.
            initial_state = {k: v.clone() for k, v in self.cascaded_norm.state_dict().items()}
        
        # Prepare inputs
        is_tensor = isinstance(batched_inputs, torch.Tensor)
        if is_tensor:
            imgs_ref = batched_inputs.to(self.device)
        else:
            # Handle list of dicts
            imgs_ref = torch.stack([x['image'] for x in batched_inputs]).to(self.device)
        
        original_scale = imgs_ref.max() <= 1.0
        if original_scale:
            imgs_ref = imgs_ref * 255.0

        # --- Step 1: Adaptation Loop (Pass 1) ---
        # Enable gradients for adaptation
        torch.set_grad_enabled(True)
        self.cascaded_norm.train()
        
        for step in range(self.config.num_adaptation_steps):
            # 1.1 Transform
            imgs_transformed, params_list = self._transform_batch(imgs_ref)
            model_input = imgs_transformed / 255.0 if original_scale else imgs_transformed
            
            # 1.2 Forward Backbone (to hit Norm layers)
            # We only need to run until the last tracked norm layer to save time?
            # Creating a partial forward is hard without hooks. 
            # We run full forward but we could optimize this later.
            if is_tensor:
                _ = self.base_model(model_input)
            else:
                # Reconstruct dict list
                _inputs = self._pack_dict_list(batched_inputs, model_input, original_scale)
                _ = self.base_model(_inputs)
                
            # 1.3 Compute Loss
            loss = self.cascaded_norm.compute_alignment_loss()
            
            # 1.4 Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Record stats
            self._stats['alignment_losses'].append(loss.item())

        # --- Step 2: Inference Pass (Pass 2) ---
        # Disable gradients for final inference (unless needed for something else)
        torch.set_grad_enabled(False)
        self.cascaded_norm.eval()
        
        # 2.1 Transform with NEW parameters
        imgs_transformed, params_list = self._transform_batch(imgs_ref)
        model_input = imgs_transformed / 255.0 if original_scale else imgs_transformed
        
        # 2.2 Final Forward
        if is_tensor:
            outputs = self.base_model(model_input)
        else:
            _inputs = self._pack_dict_list(batched_inputs, model_input, original_scale)
            outputs = self.base_model(_inputs)
            
        # Log final params
        for params in params_list:
            self._stats['transform_params'].append(tuple(p.item() for p in params))

        # --- Step 3: Restore State (Per-Sample Mode) ---
        if self.config.reset_every_batch and initial_state is not None:
            self.cascaded_norm.load_state_dict(initial_state)
            # Also ideally reset optimizer buffer, but standard Optimizer doesn't have easy reset.
            # Re-creating optimizer is expensive. 
            # For 1-step One-Pass, momentum matters less. 
            # If we want pure isolation, we should clear optimizer state.
            self.optimizer.state.clear()
            
        # Restore Grad
        torch.set_grad_enabled(True) 
            
        return outputs

    def _pack_dict_list(self, original_inputs, transformed_imgs, original_scale):
        new_inputs = []
        for i, input_dict in enumerate(original_inputs):
            new_input = input_dict.copy()
            img = transformed_imgs[i]
            # No need to div 255 here, it was done before passing to this function
            new_input['image'] = img
            new_inputs.append(new_input)
        return new_inputs

    def reset(self, reset_stats=False):
        """Reset model to GLOBAL initial state (from Pre-Init)."""
        self.cascaded_norm.load_state_dict(self.cascaded_norm_state)
        self.online(self.adapting)
        self.to(self.device)
        self.to(self.dtype)
        try:
            self.optimizer.zero_grad()
            self.optimizer.state.clear()
        except:
            pass
        if reset_stats:
            self._stats = {'alignment_losses': [], 'transform_params': []}

    @property
    def stats(self):
        if not self._stats['alignment_losses']: return None
        params_array = np.array(self._stats['transform_params'])
        return {
            'num_steps': len(self._stats['alignment_losses']),
            'mean_loss': np.mean(self._stats['alignment_losses']),
            'final_loss': self._stats['alignment_losses'][-1] if self._stats['alignment_losses'] else 0,
            'mean_noise_floor': np.mean(params_array[:, 0]) if len(params_array) > 0 else 0,
            'mean_saturation_limit': np.mean(params_array[:, 1]) if len(params_array) > 0 else 0,
            'mean_gamma': np.mean(params_array[:, 2]) if len(params_array) > 0 else 0,
        }

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.cascaded_norm = self.cascaded_norm.to(self._device)
        return result

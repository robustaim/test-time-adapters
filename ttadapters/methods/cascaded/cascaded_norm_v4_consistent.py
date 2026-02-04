"""
CascadedNorm V4 Consistent: Context-Aware Adaptation with Identity Anchor

Solves the "Drift vs. Plasticity" dilemma by learning a POLICY (Generator) instead of a STATE (Scalar).

Key Mechanisms:
1.  **Input-Dependent Generator**: 
    - Predicts (noise, gamma) based on the image content.
    - Solves "Vanishing Gradient" by being at the input layer (direct gradient access).
    - Solves "Drift" by allowing distinct reactions to Night vs. Clear images.

2.  **Identity Anchor (Consistency Preservation)**:
    - Prevents "Policy Collapse" (forgetting how to handle Clear images after Night adaptation).
    - Enforces strong regularization on Generator weights to stay close to "Identity Mapping".
    - Unlike V3's scalar regularization (which forces params to be 0), this forces the FUNCTION to remain broad.
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
class CascadedNormV4Config(AdaptationConfig):
    """Configuration for CascadedNorm V4 Consistent."""
    adaptation_name: str = "CascadedNormEngine"
    adapt_lr: float = 1e-3  # Lower LR for stable policy learning
    
    # Transformation
    temperature: float = 0.01
    saturation_limit: float = 100.0  # Fixed based on V3 finding
    
    # Regularization (The "Anchor")
    # Forces generator weights to behave like Identity function initially and resist drift
    weight_decay: float = 1e-4      # Standard weight decay
    anchor_loss_weight: float = 1.0 # Force outputs to be close to Identity for "safe" inputs
    
    # Generator Arch
    hidden_dim: int = 32


class DifferentiableHistogramStretcher(nn.Module):
    """Differentiable histogram stretching."""
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


class ParameterGenerator(nn.Module):
    """
    Lightweight CNN to predict transformation parameters from input image.
    Input: (B, C, H, W)
    Output: (B, 2) -> [noise_floor, gamma]
    """
    def __init__(self, hidden_dim=32):
        super().__init__()
        # Input 256x256 -> Downsample significantly
        # We only need "Global Atmosphere" info (is it dark? foggy?), not fine details.
        
        self.encoder = nn.Sequential(
            # 1. Extreme Downsampling (Average Pool) to remove noise and get atmosphere
            nn.AdaptiveAvgPool2d((16, 16)), # (B, 3, 16, 16)
            
            # 2. Lightweight Conv
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # (B, 32, 8, 8)
            
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # (B, 32, 1, 1)
            nn.Flatten()
        )
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # [noise_delta, gamma_delta]
        )
        
        # Initialize to Identity (Delta = 0)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
        # Force last layer to start at 0
        nn.init.constant_(self.head[-1].weight, 0.0)
        nn.init.constant_(self.head[-1].bias, 0.0)

    def forward(self, img):
        # Ensure batch dimension
        if img.dim() == 3:
            img = img.unsqueeze(0)
            
        if img.max() > 1.0:
            img = img / 255.0 # Normalize for stability
            
        feat = self.encoder(img)
        delta = self.head(feat)
        return delta


class GammaTransform(nn.Module):
    """Context-Aware Transform Controller."""
    noise_floor_init = 2.0
    gamma_init = 1.0

    def __init__(self, config: CascadedNormV4Config):
        super().__init__()
        self.saturation_limit = torch.tensor(config.saturation_limit, requires_grad=False)
        self.generator = ParameterGenerator(config.hidden_dim)
        self.stretcher = DifferentiableHistogramStretcher(config.temperature)
        
        self.current_delta = None

    def forward(self, img):
        # Predict deltas
        # delta: (B, 2)
        delta = self.generator(img)
        self.current_delta = delta
        
        # Apply deltas to anchors
        # We use mean delta for batch processing in Stretcher (current limitation of Stretcher)
        # But for V4 to work best, we should ideally stretch each image differently.
        # But Stretcher is slow in loop, let's use mean for now or optimize later.
        # Since Batch=1 usually, mean is fine.
        avg_delta = delta.mean(dim=0)
        
        noise_floor = (self.noise_floor_init + avg_delta[0]).clamp(0.0, 48.0)
        gamma = (self.gamma_init + avg_delta[1]).clamp(0.1, 3.0)
        
        transformed = self.stretcher(img, noise_floor, self.saturation_limit, gamma)
        
        return transformed, (noise_floor, self.saturation_limit, gamma)

    def get_anchor_loss(self):
        """Penalty for deviating from identity (Delta=0)."""
        if self.current_delta is None:
            return torch.tensor(0.0)
        # L2 norm of predicted deltas
        # "Unless necessary, don't change anything"
        return self.current_delta.pow(2).mean()


class CascadedNorm(nn.Module):
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

            # Align using MSE
            total_loss = total_loss + F.mse_loss(batch_mean, source_mean) + F.mse_loss(batch_var, source_var)

        # Anchor Loss (Consistency)
        if self.config.anchor_loss_weight > 0:
            anchor_loss = self.transform_controller.get_anchor_loss()
            total_loss = total_loss + anchor_loss * self.config.anchor_loss_weight

        return total_loss

    def online_parameters(self):
        return self.transform_controller.generator.parameters()


class CascadedNormEngine(AdaptationEngine):
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
        self.cascaded_norm_state = {key: value.cpu().clone() for key, value in self.cascaded_norm.state_dict().items()}
        self._extract_norm_layers()
        self._stats = {'alignment_losses': [], 'transform_params': []}
    
    def _extract_norm_layers(self):
        # We need Early layers for sensitivity!
        # And we rely on Generator's stability to prevent drift.
        # So we can safely use ALL layers or EARLY layers.
        print(f"[CascadedNorm] Extracting norm layers...")
        found = []
        for name, module in self.base_model.named_modules():
            module_type = type(module).__name__
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) or "BatchNorm" in module_type:
                found.append((name, "BN", module, module.running_mean.mean().clone(), module.running_var.mean().clone()))
            elif isinstance(module, nn.LayerNorm) or "LayerNorm" in module_type:
                found.append((name, "LN", module, torch.tensor(0.0), torch.tensor(1.0)))
        
        # Filter (User logic)
        filtered = self._filter_by_cascade_mode(found)
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
            case _: return norm_list  # Default to ALL (V4 is robust enough)

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
        for i in range(imgs.shape[0]):
            transformed, params = self.cascaded_norm(imgs[i])
            transformed_list.append(transformed)
            params_list.append(params)
        return torch.stack(transformed_list, dim=0), params_list

    def forward(self, batched_inputs):
        """Continuous Adaptation (No Reset)."""
        if not self.adapting: return self.base_model(batched_inputs)

        # Prepare inputs
        is_tensor = isinstance(batched_inputs, torch.Tensor)
        if is_tensor:
            imgs_ref = batched_inputs.to(self.device)
        else:
            imgs_ref = torch.stack([x['image'] for x in batched_inputs]).to(self.device)
        
        original_scale = imgs_ref.max() <= 1.0
        if original_scale: imgs_ref = imgs_ref * 255.0

        # --- Adaptation ---
        torch.set_grad_enabled(True)
        self.cascaded_norm.train()
        
        # 1. Forward
        imgs_transformed, params_list = self._transform_batch(imgs_ref)
        model_input = imgs_transformed / 255.0 if original_scale else imgs_transformed
        
        if is_tensor:
            _ = self.base_model(model_input)
        else:
            _inputs = self._pack_dict_list(batched_inputs, model_input, original_scale)
            _ = self.base_model(_inputs)
            
        # 2. Loss & Update
        loss = self.cascaded_norm.compute_alignment_loss()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self._stats['alignment_losses'].append(loss.item())
        
        # --- Inference (with updated policy) ---
        torch.set_grad_enabled(False)
        self.cascaded_norm.eval()
        
        imgs_transformed, params_list = self._transform_batch(imgs_ref)
        model_input = imgs_transformed / 255.0 if original_scale else imgs_transformed
        
        if is_tensor:
            outputs = self.base_model(model_input)
        else:
            _inputs = self._pack_dict_list(batched_inputs, model_input, original_scale)
            outputs = self.base_model(_inputs)
            
        for params in params_list:
            self._stats['transform_params'].append(tuple(p.item() for p in params))

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
        """Reset model to initial state."""
        self.cascaded_norm.load_state_dict(self.cascaded_norm_state)
        self.online(self.adapting)
        self.to(self.device)
        self.to(self.dtype)
        try:
            self.optimizer.zero_grad()
            self.optimizer.state.clear()
        except: pass
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
            'mean_gamma': np.mean(params_array[:, 2]) if len(params_array) > 0 else 0,
        }

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.cascaded_norm = self.cascaded_norm.to(self._device)
        return result

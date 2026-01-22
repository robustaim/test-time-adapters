"""
CascadedNorm V4: Context-Aware Input Transformation

Solves the "Domain Dilemma" (Night requires clip_low=12, Clear requires clip_low=2)
by introducing an Input-Dependent Parameter Generator.

Key Innovation:
    Instead of learning scaling parameters (scalars), we learn a lightweight Function P(x; θ)
    that looks at the image 'x' and outputs the optimal parameters.
    
    P(x) -> (noise_floor, gamma)

    - If input is Noisy/Dark -> Generator outputs High Noise Floor, Low Gamma.
    - If input is Clear/Bright -> Generator outputs Low Noise Floor, High Gamma.
    
    This prevents "Drift" because the model learns a POLICY, not a STATE.

Architecture:
    [Input 8x8] -> [Conv/Encoder] -> [Features] -> [MLP] -> [noise_floor, gamma]
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

    param_regularization: float = 0.01  # Regularize generator outputs towards anchor
    temperature: float = 0.01
    saturation_limit: float = 100.0  # Unlock gradient flow (Fixed)


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


class ContextAwareParameterGenerator(nn.Module):
    """
    Lightweight Network to generate transformation parameters from input image.
    Input: (C, H, W) Image
    Output: (noise_floor, gamma)
    """
    def __init__(self):
        super().__init__()
        # 1. Feature Extractor (Simple visual statistics)
        # Input: 3x8x8 (downsampled) -> 24x4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # (B, 16, 1, 1)
        )
        
        # 2. Parameter Heads
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 2) # [noise_delta, gamma_delta]
        )

        # Initialize to Identity-like behavior
        # Output should be close to 0 initially
        nn.init.normal_(self.head[-1].weight, 0, 0.001)
        nn.init.zeros_(self.head[-1].bias)
        
        # Anchors
        self.anchor_noise = 2.0
        self.anchor_gamma = 1.0

    def forward(self, img):
        if img.dim() == 3:
            img = img.unsqueeze(0) # (1, C, H, W)

        # Downsample for speed
        img_tiny = F.interpolate(img, size=(8, 8), mode='bilinear', align_corners=False)
        
        feats = self.encoder(img_tiny)
        deltas = self.head(feats) # (B, 2)
        
        # Add anchors
        noise_floor = self.anchor_noise + deltas[:, 0]
        gamma = self.anchor_gamma + deltas[:, 1]
        
        return noise_floor, gamma


class GammaTransform(nn.Module):
    """Context-Aware Transform Controller."""

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.saturation_limit = torch.tensor(config.saturation_limit, requires_grad=False)
        self.generator = ContextAwareParameterGenerator()
        self.stretcher = DifferentiableHistogramStretcher(config.temperature)
        
        # Storage for current batch params (for stats/loss)
        self.current_noise = None
        self.current_gamma = None

    def forward(self, img):
        """Generate parameters from image and apply transform."""
        # Generate specific params for THIS image
        batch_noise, batch_gamma = self.generator(img)
        
        # Average primarily for stats/regularization logic consistency, 
        # but technically we could stretch each image individually if we vectorized stretcher.
        # DifferentiableHistogramStretcher currently processes (C, H, W) one by one in loop.
        # For batch processing, let's use the MEAN parameters for the whole batch to align with original architecture
        # (Original V1 learned ONE set of params for the batch).
        # To make it truly instance-aware, we'd need to modify stretcher to handle batch vectors.
        # For now, let's treat the batch as a "context block".
        
        noise_floor = batch_noise.mean()
        gamma = batch_gamma.mean()
        
        # Constraints
        noise_floor = noise_floor.clamp(min=0.0, max=48.0)
        gamma = gamma.clamp(min=0.1, max=5.0)
        
        self.current_noise = noise_floor
        self.current_gamma = gamma

        transformed = self.stretcher(img, noise_floor, self.saturation_limit, gamma)
        
        return transformed, (noise_floor, self.saturation_limit, gamma)

    def get_regularization_loss(self):
        """Penalize deviation from anchors (Stability)."""
        if self.current_noise is None: 
            return torch.tensor(0.0)
            
        loss = (self.current_noise - 2.0).pow(2) + \
               (self.current_gamma - 1.0).pow(2)
        return loss


class CascadedNorm(nn.Module):
    """
    CascadedNorm: Manages transformation and norm layer statistics.
    """

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config

        self.transform_controller = GammaTransform(config)

        self.norm_layers: List[nn.Module] = []
        self.norm_types: List[str] = []
        self.source_means: List[torch.Tensor] = []
        self.source_vars: List[torch.Tensor] = []

    def forward(self, img):
        # Image-wise forward is handled by controller
        transformed, params = self.transform_controller(img)
        output = 0.5 * transformed + 0.5 * img
        return output, params

    def compute_alignment_loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.source_means[0].device)

        for i, (norm_layer, norm_type) in enumerate(zip(self.norm_layers, self.norm_types)):
            if not hasattr(norm_layer, 'current_mean') or norm_layer.current_mean is None:
                continue

            batch_mean = norm_layer.current_mean
            batch_var = norm_layer.current_var
            source_mean = self.source_means[i].to(batch_mean.device)
            source_var = self.source_vars[i].to(batch_var.device)

            if norm_type == 'BN' and batch_mean.ndim > 0:
                batch_mean = batch_mean.mean()
                batch_var = batch_var.mean()

            loss_mean = F.mse_loss(batch_mean, source_mean)
            loss_var = F.mse_loss(batch_var, source_var)
            total_loss = total_loss + loss_mean + loss_var

        if self.config.param_regularization > 0:
            reg_loss = self.transform_controller.get_regularization_loss()
            total_loss = total_loss + reg_loss * self.config.param_regularization

        return total_loss
    
    def online_parameters(self):
        # Optimize the generator weights
        return self.transform_controller.generator.parameters()


class CascadedNormEngine(AdaptationEngine):
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

    def forward(self, batched_inputs):
        if not self.adapting: return self.base_model(batched_inputs)
        if isinstance(batched_inputs, torch.Tensor): return self._forward_tensor(batched_inputs)
        return self._forward_dict_list(batched_inputs)

    def _forward_tensor(self, imgs):
        imgs = imgs.to(self._device)
        original_scale = imgs.max() <= 1.0
        if original_scale: imgs = imgs * 255.0
        imgs_transformed, params_list = self._transform_batch(imgs)
        for params in params_list: self._stats['transform_params'].append(tuple(p.item() for p in params))
        model_input = imgs_transformed / 255.0 if original_scale else imgs_transformed
        outputs = self.base_model(model_input)
        alignment_loss = self.cascaded_norm.compute_alignment_loss()
        total_loss = alignment_loss
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
        total_loss = alignment_loss
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
            'mean_noise_floor': np.mean(params_array[:, 0]),
            'mean_saturation_limit': np.mean(params_array[:, 1]),
            'mean_gamma': np.mean(params_array[:, 2]),
        }

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.cascaded_norm = self.cascaded_norm.to(self._device)
        return result

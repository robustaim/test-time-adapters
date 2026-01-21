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
    Learnable parameters for adaptive gamma with delta prediction from stats.
    
    Architecture:
        - Gamma = Global Base + Predicted Delta (from Image Stats)
        - Temperature = Fixed (Config)
        - Gating = Global Base (0.5) + Residual (Learnable)
    """

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        # Global Parameters (Anchors)
        # Init with inverse_sigmoid for correct starting values
        # clip_low: Target 2.0 (in 0-10 range) -> sigmoid(x)=0.2 -> x = -1.386
        self.clip_low = nn.Parameter(torch.tensor(-1.386))
        
        # clip_high: Target 98.0 (in 90-100 range) -> sigmoid(x)=0.8 -> x = 1.386
        # Previous init (98.0) caused sigmoid saturation (grad=0) and stuck at 100.
        self.clip_high = nn.Parameter(torch.tensor(1.386))
        
        self.gamma_base = nn.Parameter(torch.tensor(0.0))  # Target 1.0 -> sigmoid(x)=0.33 -> x=-0.69 (if range 0.5-2.0). 
                                                           # Wait, gamma formula is 0.5 + sig(x)*1.5. 
                                                           # Target 1.0 => 0.5 + 0.5 = 1.0 => sig(x)=0.333 => x=-0.693.
                                                           # If prev init was 1.0: 0.5 + 0.73*1.5 = 1.6.  
                                                           # Let's align gamma too? User didn't ask but good practice.
                                                           # User code had self.gamma_base = 1.0. 
                                                           # Let's stick to user's Gamma init if not broken, but fix Clips primarily.
                                                           # Actually, I'll just fix Clips 2.0/98.0.
                                                           
        self.gamma_base = nn.Parameter(torch.tensor(0.0)) # Let's set Gamma Base to start at ~1.25 (Center) or 1.0?
                                                          # v3 init was 1.0 -> 0.5 + 0.73*1.5 = 1.6. 
                                                          # Let's stick to 1.0 if user liked v3 start. 
                                                          # Wait, I am replacing lines 110-112. I should carefully modify gamma too if needed.
                                                          # User's v4 currently has gamma_base = 1.0. 
                                                          # Leave Gamma as is to minimize variables changed, focus on Fixing Clips.
                                                          
        self.gamma_base = nn.Parameter(torch.tensor(1.0))
        
        # Gamma Delta Predictor (MLP: 6 -> 32 -> 1)
        # Input: Mean(3) + Std(3) = 6
        # Gamma Delta Predictor (Linear: 6 -> 1)
        # Simplify to Linear to prevent overfitting/instability
        self.gamma_delta = nn.Linear(6, 1)
        
        # Initialize Delta Predictor to 0 (Identity Start)
        nn.init.constant_(self.gamma_delta.weight, 0.0)
        nn.init.constant_(self.gamma_delta.bias, 0.0)
        
        # Integrated stretcher
        self.stretcher = DifferentiableHistogramStretcher(config.temperature)

    def forward(self, img):
        """
        Args:
            img: (B, C, H, W) image tensor
        Returns:
            transformed: (B, C, H, W)
            params: tuple for logging
        """
        # 1. Global Parameters
        clip_low = torch.sigmoid(self.clip_low) * 10  # [0, 10]
        clip_high = 90 + torch.sigmoid(self.clip_high) * 10  # [90, 100]
        
        # 2. Compute Gamma (Global + Delta)
        # Global Base: Anchored at v3 optimum (init 1.25 range [0.5, 2.0])
        gamma_base = 0.5 + torch.sigmoid(self.gamma_base) * 1.5 
        
        # Delta Prediction: From Image Stats (Mean, Std)
        if img.dim() == 3:
            C, H, W = img.shape
            B = 1
            flat = img.view(1, C, -1)
        else:
            B, C, H, W = img.shape
            flat = img.view(B, C, -1)

        stats = torch.cat([flat.mean(dim=2), flat.std(dim=2)], dim=1) / 255.0  # (B, 6) Normalized
        
        gamma_delta = self.gamma_delta(stats) # (B, 1)
        
        # Final Gamma: Base + Residual (tanh scaled to +/- 0.5)
        raw_gamma = gamma_base + 0.5 * torch.tanh(gamma_delta)
        gamma = raw_gamma.clamp(0.5, 3.0)  # Safety Clamp (Critical for Stability)
        
        # 3. Transform
        transformed = self.stretcher(img, clip_low, clip_high, gamma)
        output = 0.5 * transformed + 0.5 * img  # residual form

        return output, (clip_low, clip_high, gamma)


class CascadedNorm(nn.Module):
    """
    CascadedNorm: Manages transformation and norm layer statistics.
    
    Integrates GammaTransform controller and tracks normalization layers.
    """

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config
        
        # Transform controller with integrated stretcher
        self.transform_controller = GammaTransform(config)
        
        # Norm layer tracking (will be populated by Engine)
        self.norm_layers: List[nn.Module] = []
        self.norm_types: List[str] = []  # 'bn' or 'ln'
        self.source_means: List[torch.Tensor] = []
        self.source_vars: List[torch.Tensor] = []

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
    CascadedNorm: Input transformation for BN/LN alignment.

    Transforms input to match norm layer source statistics.
    Works with both BatchNorm and LayerNorm.
    """
    model_name: str = "CascadedNormEngine"

    def __init__(self, base_model: BaseModel, config: CascadedNormConfig):
        self.cascaded_norm: CascadedNorm  # will be initialized in _pre_init()
        self.cascaded_norm_state: dict
        self.config = config

        super().__init__(base_model, config)

    def _pre_init(self):
        # Transformation modules
        self.cascaded_norm = CascadedNorm(self.config)

    def _post_init(self):
        self.cascaded_norm.to(self.device)
        self.cascaded_norm_state = {key: value.cpu() for key, value in self.cascaded_norm.state_dict().items()}

        # Extract norm layers and wrap them
        self._extract_norm_layers()

        # Stats
        self._stats = {'alignment_losses': [], 'transform_params': []}

    def _extract_norm_layers(self):
        """
        Find all normalization layers including FrozenBatchNorm.

        Extracts source statistics (running_mean/var) for alignment.
        """
        print(f"[CascadedNorm] Extracting norm layers...")
        found = []
        for name, module in self.base_model.named_modules():
            module_type = type(module).__name__

            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) or "BatchNorm" in module_type:
                # BN: Use running statistics (averaged over channels)
                found.append((
                    name, "BN", module,
                    module.running_mean.mean().clone(), # Scalar source mean
                    module.running_var.mean().clone()   # Scalar source var
                ))
            elif isinstance(module, nn.LayerNorm) or "LayerNorm" in module_type:
                 # LN: Target normalized distribution (mean=0, var=1)
                found.append((
                    name, "LN", module,
                    torch.tensor(0.0),
                    torch.tensor(1.0)
                ))
        
        # Filtering & Wrapping
        filtered = self._filter_by_cascade_mode(found)
        self._cascade_wrap(filtered)
        
        # Populate to CascadedNorm
        for _, norm_type, module, running_mean, running_var in filtered:
            self.cascaded_norm.norm_layers.append(module)
            self.cascaded_norm.norm_types.append(norm_type)
            self.cascaded_norm.source_means.append(running_mean)
            self.cascaded_norm.source_vars.append(running_var)

        print(f"[CascadedNorm] Found {len(self.cascaded_norm.norm_layers)} norm layers "
              f"(BN: {self.cascaded_norm.norm_types.count('BN')}, "
              f"LN: {self.cascaded_norm.norm_types.count('LN')})")

    def _filter_by_cascade_mode(self, norm_list):
        """Filter normalization layers based on cascade mode."""
        if not hasattr(self.config, 'cascade_mode'):
             return norm_list
             
        match self.config.cascade_mode:
            case "single":
                return [norm_list[0]]
            case "single_last":
                return [norm_list[-1]]
            case "selected":
                return [norm_list[i] for i in getattr(self.config, 'cascade_indices', [])]
            case _:  # all
                return norm_list

    @staticmethod
    def _cascade_wrap(filtered: list[nn.Module]):
        """Wrap norm layer forward methods to capture batch statistics."""
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

    def online_parameters(self):
        """Only transformation parameters."""
        return self.cascaded_norm.online_parameters()

    def _transform_image(self, img):
        """Transform single image with memory-based adaptive gating."""
        transformed, params = self.cascaded_norm.transform_controller(img)
        return transformed, params

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
        for param in self.cascaded_norm.transform_controller.parameters():
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

        alignment_loss = self.cascaded_norm.compute_alignment_loss()
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

        alignment_loss = self.cascaded_norm.compute_alignment_loss()
        reg_loss = self._compute_regularization_loss()
        
        total_loss = alignment_loss + reg_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._stats['alignment_losses'].append(total_loss.item())

        return outputs

    def reset(self, reset_stats=False):
        """Reset model to initial state."""
        self.cascaded_norm.load_state_dict(self.cascaded_norm_state)
        self.online(self.adapting)
        self.to(self.device)
        self.to(self.dtype)
        try:
            self.optimizer.zero_grad()
        except:
            pass
        if reset_stats:
            self._stats = {'alignment_losses': [], 'transform_params': []}

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
        self.cascaded_norm = self.cascaded_norm.to(self._device)
        return result

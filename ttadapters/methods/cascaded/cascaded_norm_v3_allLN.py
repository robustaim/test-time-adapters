"""
CascadedNorm All-LN: Drift-Free Input Transformation

Converts all BatchNorm layers to LayerNorm to eliminate parameter drift.

Key Innovation:
    1. Replace BN with LN (domain-agnostic normalization)
    2. Initialize LN's gamma/beta from BN's running_mean/var (preserve source info)
    3. All layers align to (mean=0, var=1) target (consistent across domains)
    
Why This Solves Drift:
    - BN: Domain-specific target (Clear's running stats) → Drift when switching domains
    - LN: Domain-agnostic target (0, 1) → Stable equilibrium across all domains
    
Pipeline:
    [Input] → [Transform T(θ)] → [Model with LN (BN→LN converted)] → [Output]
                    ↑
              Update via (0, 1) alignment loss (all layers)
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

    temperature: float = 0.01
    saturation_limit: float = 100.0
    param_regularization: float = 0.1


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
    """Learnable parameters for noise reduction with gamma correction."""
    noise_floor_init = 2.0
    gamma_init = 1.0

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.saturation_limit = torch.tensor(config.saturation_limit, requires_grad=False)
        self.noise_floor = nn.Parameter(torch.tensor(self.noise_floor_init))
        self.gamma = nn.Parameter(torch.tensor(self.gamma_init))

        self.stretcher = DifferentiableHistogramStretcher(config.temperature)

    def forward(self, img):
        """Get constrained parameters and apply transform."""
        noise_floor = self.noise_floor.clamp(min=0.0, max=20.0)  # pass range 20~100
        gamma = self.gamma.clamp(min=0.5, max=2.0)  # gamma range 0.5~2.0

        transformed = self.stretcher(img, noise_floor, self.saturation_limit, gamma)

        return transformed, (noise_floor, self.saturation_limit, gamma)

    def get_regularization_loss(self):
        """Compute regularization loss relative to initialization anchors."""
        return (self.noise_floor - self.noise_floor_init).pow(2) + (self.gamma - self.gamma_init).pow(2)


class CascadedAnchor(nn.Module):
    """
    Unified normalization anchor for drift-free adaptation.
    
    Preserves input→output relationship (like LN) by using batch stats + learned affine.
    """
    
    def __init__(self, original_module, normalized_shape, is_from_bn=False):
        """
        Args:
            original_module: Original BN/LN module
            normalized_shape: Shape for normalization
            is_from_bn: Whether converting from BN (True) or LN (False)
        """
        super().__init__()
        
        self.normalized_shape = normalized_shape
        self.is_from_bn = is_from_bn
        self.eps = 1e-5
        
        # Preserve learned affine for consistent input→output relationship
        if is_from_bn:
            # BN: Preserve learned gamma/beta
            # This maintains "BN input → BN output" relationship
            if hasattr(original_module, 'weight') and original_module.weight is not None:
                self.weight = nn.Parameter(original_module.weight.clone())
                self.bias = nn.Parameter(original_module.bias.clone())
            else:
                # No affine
                num_features = original_module.num_features
                self.weight = nn.Parameter(torch.ones(num_features))
                self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            # LN: Keep learned parameters
            self.weight = nn.Parameter(original_module.weight.clone())
            self.bias = nn.Parameter(original_module.bias.clone())
        
        # Stats tracking
        self.register_buffer('current_mean', torch.tensor(0.0))
        self.register_buffer('current_var', torch.tensor(0.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization using BATCH statistics (LN-style).
        
        Preserves BN's learned gamma/beta but uses batch stats for domain-agnostic behavior.
        """
        if x.dim() == 4:  # (B, C, H, W) - from BN2d
            # Measure input stats
            dims = (0, 2, 3)
            self.current_mean = x.mean(dim=dims)
            self.current_var = x.var(dim=dims, unbiased=False)
            
            # Normalize using BATCH statistics (key difference from BN!)
            batch_mean = x.mean(dim=dims, keepdim=True)
            batch_var = x.var(dim=dims, keepdim=True, unbiased=False)
            normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            
            # Apply BN's LEARNED gamma/beta
            gamma = self.weight.view(1, -1, 1, 1)
            beta = self.bias.view(1, -1, 1, 1)
            
            return normalized * gamma + beta
            
        elif x.dim() == 3:  # (B, C, L) - sequence
            dims = (0, 2)
            self.current_mean = x.mean(dim=dims)
            self.current_var = x.var(dim=dims, unbiased=False)
            
            batch_mean = x.mean(dim=dims, keepdim=True)
            batch_var = x.var(dim=dims, keepdim=True, unbiased=False)
            normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            
            gamma = self.weight.view(1, -1, 1)
            beta = self.bias.view(1, -1, 1)
            
            return normalized * gamma + beta
            
        else:
            # Standard LN behavior
            if hasattr(self, 'normalized_shape') and self.normalized_shape:
                dims = tuple(range(-len(self.normalized_shape), 0))
            else:
                dims = None
            
            if dims:
                self.current_mean = x.mean(dim=dims)
                self.current_var = x.var(dim=dims, unbiased=False)
                batch_mean = x.mean(dim=dims, keepdim=True)
                batch_var = x.var(dim=dims, keepdim=True, unbiased=False)
            else:
                self.current_mean = x.mean()
                self.current_var = x.var(unbiased=False)
                batch_mean = x.mean(keepdim=True)
                batch_var = x.var(keepdim=True, unbiased=False)
                
            normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            return normalized * self.weight + self.bias



class CascadedNorm(nn.Module):
    """
    CascadedNorm All-LN: Manages transformation and CascadedAnchors.
    
    All norm layers are replaced with CascadedAnchors targeting (mean=0, var=1).
    """

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config

        # Transform controller
        self.transform_controller = GammaTransform(config)

        # CascadedAnchors tracking (populated by Engine)
        self.norm_layers: List[CascadedAnchor] = []
        self.norm_types: List[str] = []  # 'BN→LN' or 'LN' (for logging)
        self.source_means: List[torch.Tensor] = []  # All (0.0) for compatibility
        self.source_vars: List[torch.Tensor] = []   # All (1.0) for compatibility

    def forward(self, img):
        transformed, params = self.transform_controller(img)

        # Residual Form
        output = 0.5 * transformed + 0.5 * img

        return output, params

    def compute_alignment_loss(self) -> torch.Tensor:
        """
        Compute alignment loss for all CascadedAnchors.
        
        All anchors target (mean=0, var=1) for drift-free adaptation.
        """
        if len(self.norm_layers) == 0:
            return torch.tensor(0.0)
        
        total_loss = torch.tensor(0.0, device=self.source_means[0].device)
        
        for anchor in self.norm_layers:
            # All anchors are CascadedAnchor instances with current_mean/var
            batch_mean = anchor.current_mean
            batch_var = anchor.current_var
            
            # Reduce to scalar if needed
            if batch_mean.numel() > 1:
                batch_mean = batch_mean.mean()
            if batch_var.numel() > 1:
                batch_var = batch_var.mean()
            
            # Target: (0, 1) - domain-agnostic
            target_mean = torch.tensor(0.0, device=batch_mean.device)
            target_var = torch.tensor(1.0, device=batch_var.device)
            
            # MSE loss
            loss_mean = F.mse_loss(batch_mean, target_mean)
            loss_var = F.mse_loss(batch_var, target_var)
            
            total_loss = total_loss + loss_mean + loss_var
        
        # Add parameter regularization
        if self.config.param_regularization > 0:
            reg_loss = self.transform_controller.get_regularization_loss()
            total_loss = total_loss + reg_loss * self.config.param_regularization
        
        return total_loss

    def online_parameters(self):
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

    def _replace_with_anchor(self, module, parent_module, attr_name):
        """
        Replace BN or LN with CascadedAnchor.
        
        Args:
            module: Original BN/LN module
            parent_module: Parent module
            attr_name: Attribute name in parent
            
        Returns:
            (anchor, layer_type)
        """
        module_type = type(module).__name__
        
        # Determine normalized_shape and conversion type
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) or "BatchNorm" in module_type:
            # BatchNorm → CascadedAnchor
            if hasattr(module, 'num_features'):
                normalized_shape = (module.num_features,)
            else:
                normalized_shape = module.running_mean.shape
            
            is_from_bn = True
            layer_type = "BN→LN"
            
        elif isinstance(module, nn.LayerNorm) or "LayerNorm" in module_type:
            # LayerNorm → CascadedAnchor
            normalized_shape = module.normalized_shape
            is_from_bn = False
            layer_type = "LN"
            
        else:
            raise ValueError(f"Unsupported module type: {module_type}")
        
        # Create CascadedAnchor with ORIGINAL module
        anchor = CascadedAnchor(
            original_module=module,
            normalized_shape=normalized_shape,
            is_from_bn=is_from_bn
        )
        
        # Move to same device
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) or "BatchNorm" in module_type:
            anchor = anchor.to(module.running_mean.device)
        else:
            anchor = anchor.to(next(module.parameters()).device)
        
        # Replace in parent
        if parent_module is not None and attr_name is not None:
            setattr(parent_module, attr_name, anchor)
        
        return anchor, layer_type

    def _extract_norm_layers(self):
        """
        Replace all BN/LN with CascadedAnchor for unified (0, 1) alignment.
        """
        print(f"[CascadedNorm AllLN] Replacing norm layers with CascadedAnchor...")
        found = []
        conversion_log = []
        
        # Collect all norm layers with parent info
        module_list = list(self.base_model.named_modules())
        
        for name, module in module_list:
            module_type = type(module).__name__
            
            # Skip if not a norm layer
            if not (isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)) or 
                    "BatchNorm" in module_type or "LayerNorm" in module_type):
                continue
            
            # Find parent module
            parent_module = None
            attr_name = None
            if '.' in name:
                parent_name = name.rsplit('.', 1)[0]
                attr_name = name.rsplit('.', 1)[1]
                for pname, pmodule in module_list:
                    if pname == parent_name:
                        parent_module = pmodule
                        break
            else:
                parent_module = self.base_model
                attr_name = name
            
            # Replace with CascadedAnchor
            try:
                anchor, layer_type = self._replace_with_anchor(
                    module, parent_module, attr_name
                )
                
                # Add to tracking
                found.append((
                    name, layer_type, anchor,
                    torch.tensor(0.0),  # Target: mean = 0
                    torch.tensor(1.0)   # Target: var = 1
                ))
                
                # Log with parameter info
                if layer_type == "BN→LN":
                    gamma_mean = anchor.weight.mean().item()
                    beta_mean = anchor.bias.mean().item()
                    conversion_log.append(f"  [{layer_type}] {name}: γ={gamma_mean:.2f}, β={beta_mean:.2f} (learned)")
                else:
                    conversion_log.append(f"  [{layer_type}] {name}")
                
            except Exception as e:
                print(f"  [WARNING] Failed to convert {name}: {e}")
        
        # Log conversions
        print(f"\n[Conversions]")
        for log in conversion_log[:10]:  # Show first 10
            print(log)
        if len(conversion_log) > 10:
            print(f"  ... and {len(conversion_log) - 10} more")

        
        # Filtering
        filtered = self._filter_by_cascade_mode(found)
        
        # Populate to CascadedNorm
        for _, norm_type, anchor, source_mean, source_var in filtered:
            self.cascaded_norm.norm_layers.append(anchor)
            self.cascaded_norm.norm_types.append(norm_type)
            self.cascaded_norm.source_means.append(source_mean)
            self.cascaded_norm.source_vars.append(source_var)
        
        # Summary
        bn_to_ln = sum(1 for nt in self.cascaded_norm.norm_types if nt == "BN→LN")
        ln_count = sum(1 for nt in self.cascaded_norm.norm_types if nt == "LN")
        
        print(f"\n[Summary]")
        print(f"  Total anchors: {len(self.cascaded_norm.norm_layers)}")
        print(f"  BN→LN: {bn_to_ln}, LN: {ln_count}")
        print(f"  All layers target (μ=0, σ²=1) for drift-free adaptation")


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


    def online_parameters(self):
        """Only transformation parameters."""
        return self.cascaded_norm.online_parameters()

    def _transform_batch(self, imgs):
        """Transform batch."""
        transformed_list = []
        params_list = []

        for i in range(imgs.shape[0]):
            transformed, params = self.cascaded_norm(imgs[i])
            transformed_list.append(transformed)
            params_list.append(params)

        return torch.stack(transformed_list, dim=0), params_list

    def forward(self, batched_inputs=None, **kwargs):
        """Forward with transformation and alignment."""
        # Handle case where inputs are passed as kwargs (e.g. model(**batch))
        is_kwargs = False
        if batched_inputs is None and kwargs:
            batched_inputs = kwargs
            is_kwargs = True

        if not self.adapting:
            if is_kwargs:
                return self.base_model(**batched_inputs)
            return self.base_model(batched_inputs)

        if isinstance(batched_inputs, torch.Tensor):
            return self._forward_tensor(batched_inputs)
        elif isinstance(batched_inputs, dict):
            return self._forward_dict(batched_inputs, unpack_args=is_kwargs)
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
        total_loss = alignment_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._stats['alignment_losses'].append(total_loss.item())

        return outputs

    def _forward_dict(self, input_dict, unpack_args=False):
        """Handle dictionary input (RT-DETR, YOLO)."""
        if 'pixel_values' in input_dict:
            img_key = 'pixel_values'
        elif 'img' in input_dict:
            img_key = 'img'
        else:
            if unpack_args:
                return self.base_model(**input_dict)
            return self.base_model(input_dict)

        imgs = input_dict[img_key].to(self._device)
        
        original_scale = imgs.max() <= 1.0
        if original_scale:
            imgs = imgs * 255.0

        imgs_transformed, params_list = self._transform_batch(imgs)

        for params in params_list:
            self._stats['transform_params'].append(tuple(p.item() for p in params))

        model_input = imgs_transformed / 255.0 if original_scale else imgs_transformed
        
        new_input = input_dict.copy()
        new_input[img_key] = model_input
        
        if unpack_args:
            outputs = self.base_model(**new_input)
        else:
            outputs = self.base_model(new_input)

        alignment_loss = self.cascaded_norm.compute_alignment_loss()
        total_loss = alignment_loss

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
            'mean_noise_floor': np.mean(params_array[:, 0]),
            'mean_saturation_limit': np.mean(params_array[:, 1]),
            'mean_gamma': np.mean(params_array[:, 2]),
        }

    def to(self, *args, **kwargs):
        """Move to device."""
        result = super().to(*args, **kwargs)
        self.cascaded_norm = self.cascaded_norm.to(self._device)
        return result

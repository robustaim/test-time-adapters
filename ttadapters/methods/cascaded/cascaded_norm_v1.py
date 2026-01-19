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
    """Learnable parameters for histogram stretching with gamma correction."""

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.clip_low = nn.Parameter(torch.tensor(2.0))
        self.clip_high = nn.Parameter(torch.tensor(98.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))  # Gamma correction
        
        # Integrated stretcher
        self.stretcher = DifferentiableHistogramStretcher(config.temperature)

    def forward(self):
        """Get constrained parameters."""
        clip_low = torch.sigmoid(self.clip_low) * 10  # [0, 10]
        clip_high = 90 + torch.sigmoid(self.clip_high) * 10  # [90, 100]
        gamma = 0.5 + torch.sigmoid(self.gamma) * 1.5  # [0.5, 2.0]
        return clip_low, clip_high, gamma


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

    def extract_norm_layers(self, model: nn.Module, cascade_wrap_fn):
        """Find all BN/LN layers and wrap them."""
        found = []
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d) or "BatchNorm2d" in module.__class__.__name__:
                # BN: Use running statistics (averaged over channels)
                found.append((
                    name, "BN", module,
                    module.running_mean.mean().clone(),
                    module.running_var.mean().clone()
                ))
                print(f"  [BN] Found: {name} ({module.num_features} channels)")

            elif isinstance(module, nn.LayerNorm) or "LayerNorm" in module.__class__.__name__:
                # LN: Target normalized distribution (mean=0, var=1)
                found.append((
                    name, "LN", module,
                    torch.tensor(0.0),
                    torch.tensor(1.0)
                ))
                print(f"  [LN] Found: {name}")
        
        # Wrap layers
        cascade_wrap_fn(found)
        
        # Populate norm layer info
        for _, _, module, running_mean, running_var in found:
            self.norm_layers.append(module)
            self.norm_types.append(_)
            self.source_means.append(running_mean)
            self.source_vars.append(running_var)

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

        # Inject dist_norm into base model
        self._inject_dist_norm()

        # Extract norm layers and wrap them
        print(f"[CascadedNorm] Extracting norm layers...")
        self.cascaded_norm.extract_norm_layers(self.base_model, self._cascade_wrap)
        print(f"[CascadedNorm] Found {len(self.cascaded_norm.norm_layers)} norm layers "
              f"(BN: {self.cascaded_norm.norm_types.count('BN')}, "
              f"LN: {self.cascaded_norm.norm_types.count('LN')})")

        # Stats
        self._stats = {'alignment_losses': [], 'transform_params': []}

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
                if self.adapting:
                    # Handle tensor input
                    if isinstance(x, torch.Tensor) and x.ndim == 4:
                        # V1: Scale to 255, transform each image, scale back
                        original_scale = x.max() <= 1.0
                        if original_scale:
                            x = x * 255.0
                        
                        # Transform batch (process each image individually)
                        transformed_list = []
                        for i in range(x.shape[0]):
                            clip_low, clip_high, gamma = self.cascaded_norm.transform_controller()
                            transformed = self.cascaded_norm.transform_controller.stretcher(x[i], clip_low, clip_high, gamma)
                            transformed_list.append(transformed)
                        x = torch.stack(transformed_list, dim=0)
                        
                        if original_scale:
                            x = x / 255.0
                    
                    # Handle dict_list input (Detectron2 format)
                    elif isinstance(x, list) and len(x) > 0 and isinstance(x[0], dict):
                        transformed_inputs = []
                        for input_dict in x:
                            if 'image' not in input_dict:
                                transformed_inputs.append(input_dict)
                                continue
                            
                            img = input_dict['image'].to(self._device)
                            original_scale = img.max() <= 1.0
                            if original_scale:
                                img = img * 255.0
                            
                            # Transform single image
                            clip_low, clip_high, gamma = self.cascaded_norm.transform_controller()
                            img_transformed = self.cascaded_norm.transform_controller.stretcher(img, clip_low, clip_high, gamma)
                            
                            new_input = input_dict.copy()
                            new_input['image'] = img_transformed / 255.0 if original_scale else img_transformed
                            transformed_inputs.append(new_input)
                        x = transformed_inputs
                
                return original_forward(x, *args, **kwargs)
            self.base_model.forward = preprocessing_forward
            return

        # Wrap the first module's forward
        original_forward = first_module.forward
        
        # Counter for debugging
        self._transform_call_count = 0

        def preprocessing_forward(x, *args, **kwargs):
            # Apply dist_norm to image input with 255 scaling only when adapting
            # At this point, x is always a tensor (dict extraction happened earlier)
            if self.adapting and isinstance(x, torch.Tensor) and x.ndim == 4:
                self._transform_call_count += 1
                print(f"[DEBUG] Transform #{self._transform_call_count}, shape: {x.shape}")
                original_scale = x.max() <= 1.0
                if original_scale:
                    x = x * 255.0
                
                # Transform batch (process each image individually)
                transformed_list = []
                for i in range(x.shape[0]):
                    clip_low, clip_high, gamma = self.cascaded_norm.transform_controller()
                    transformed = self.cascaded_norm.transform_controller.stretcher(x[i], clip_low, clip_high, gamma)
                    transformed_list.append(transformed)
                x = torch.stack(transformed_list, dim=0)
                
                if original_scale:
                    x = x / 255.0
                print(f"[DEBUG] Transform done, params: {clip_low.item():.2f}, {clip_high.item():.2f}, {gamma.item():.2f}")
            
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

    def online_parameters(self):
        """Only transformation parameters."""
        return self.cascaded_norm.online_parameters()

    def _compute_regularization_loss(self):
        """L2 regularization."""
        reg_loss = torch.tensor(0.0, device=self._device)
        for param in self.cascaded_norm.transform_controller.parameters():
            reg_loss = reg_loss + param.pow(2).sum()
        return self.config.param_regularization * reg_loss

    def forward(self, *args, **kwargs):
        """Forward pass with adaptation."""
        if not self.adapting:
            return self.base_model(*args, **kwargs)

        # Reset transform counter for this forward call
        self._transform_call_count = 0
        print(f"\n[FORWARD START]")

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward (transformation happens in _inject_dist_norm wrapper)
        result = self.base_model(*args, **kwargs)

        print(f"[FORWARD END] Total transformations in this forward: {self._transform_call_count}\n")

        # Compute alignment loss
        alignment_loss = self.cascaded_norm.compute_alignment_loss()
        reg_loss = self._compute_regularization_loss()
        total_loss = alignment_loss + reg_loss

        self._stats['alignment_losses'].append(total_loss.item())

        # Backward and update
        total_loss.backward()
        self.optimizer.step()

        # Log transform parameters
        clip_low, clip_high, gamma = self.cascaded_norm.transform_controller()
        self._stats['transform_params'].append({
            'clip_low': clip_low.item(),
            'clip_high': clip_high.item(),
            'gamma': gamma.item() if gamma.dim() == 0 else gamma.mean().item()
        })

        return result

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

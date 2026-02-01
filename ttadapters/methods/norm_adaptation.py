from typing import Optional, List, Literal
from dataclasses import dataclass
import copy

import torch
from torch import nn

from .base import AdaptationEngine, AdaptationConfig
from ..models.base import BaseModel, ModelProvider

try:
    from detectron2.layers import FrozenBatchNorm2d
except ImportError:
    FrozenBatchNorm2d = None

try:
    from transformers.models.rt_detr.modeling_rt_detr import RTDetrFrozenBatchNorm2d
except ImportError:
    RTDetrFrozenBatchNorm2d = None


@dataclass
class NormAdaptationConfig(AdaptationConfig):
    adaptation_name: str = "NormAdaptationEngine"
    adaptation_layers: str = "backbone+encoder"
    source_sum: int = 128


class NormAdaptationEngine(AdaptationEngine):
    """
    Norm Adaptation Engine (also known as TENT-like or Source-Weighted Norm).
    
    Refines Batch Normalization statistics during test time by blending 
    current batch statistics with running statistics based on a 'source_sum' heuristic.
    """
    model_name: str = "NormAdaptationEngine"
    
    def __init__(self, base_model: BaseModel, config: NormAdaptationConfig):
        self.config: NormAdaptationConfig = config
        self.wrapped_layers = []
        
        super().__init__(base_model, config)

    def _post_init(self):
        self._apply_norm_adaptation()
    
    def online(self, mode=True):
        self.adapting = mode
        if mode:
            self.base_model.eval()
            self._set_norm_mode(True)
        else:
             self._set_norm_mode(False)
             
    def _set_norm_mode(self, active: bool):
        for module in self.wrapped_layers:
            module.norm_active = active

    def _apply_norm_adaptation(self):
        # Identify layers
        candidates = []
        for name, module in self.base_model.named_modules():
             if self._is_norm_layer(module):
                candidates.append((name, module))
        
        filtered = []
        for name, module in candidates:
            should_adapt = False
            
            is_decoder = 'decoder' in name.lower()
            is_encoder = 'encoder' in name.lower() and not is_decoder
            is_backbone = 'backbone' in name.lower() or 'bottom_up' in name.lower()
            
            if self.config.adaptation_layers == "backbone":
                if is_backbone:
                     should_adapt = True
            elif self.config.adaptation_layers == "encoder":
                if is_encoder:
                    should_adapt = True
            elif self.config.adaptation_layers == "backbone+encoder":
                if not is_decoder:
                    should_adapt = True
            else:
                if not is_decoder:
                    should_adapt = True
            
            if should_adapt:
                filtered.append(module)
                
        # Wrap layers
        for module in filtered:
            self._wrap_layer(module)
            self.wrapped_layers.append(module)
            
        if self.config.verbose:
            print(f"[NormAdapt] Wrapped {len(self.wrapped_layers)} layers for adaptation.")

    def _is_norm_layer(self, module: nn.Module) -> bool:
        if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            return True
        if FrozenBatchNorm2d and isinstance(module, FrozenBatchNorm2d):
            return True
        if RTDetrFrozenBatchNorm2d and isinstance(module, RTDetrFrozenBatchNorm2d):
            return True
        return False

    def _wrap_layer(self, module):
        module.norm_active = False
        module.source_sum = self.config.source_sum
        
        if not hasattr(module, 'original_running_mean') and hasattr(module, 'running_mean'):
             module.original_running_mean = module.running_mean.clone()
             module.original_running_var = module.running_var.clone()
        
        original_forward = module.forward
        
        def norm_forward(_self, x):
            if hasattr(_self, 'norm_active') and _self.norm_active:
                # Calculate alpha
                alpha = x.shape[0] / (_self.source_sum + x.shape[0])
                
                # Calculate current batch stats
                if x.dim() == 4:
                        reduce_dims = (0, 2, 3)
                else:
                        reduce_dims = (0,) # Fallback for 2D/3D (maybe (0, 2)?)

                if isinstance(_self, nn.LayerNorm):
                     nb_dims = len(_self.normalized_shape)
                     reduce_dims = tuple(range(0, x.dim() - nb_dims))

                batch_mean = x.mean(dim=reduce_dims)
                batch_var = x.var(dim=reduce_dims, unbiased=True) # Unbiased usually

                # Update running stats mechanism
                # The logic in baseline is:
                # running_mean = (1 - alpha) * self.running_mean + alpha * x.mean(...)
                # But it USES this valid "running_mean" to normalize.
                
                running_mean = _self.running_mean
                running_var = _self.running_var
                
                # Check if we should update or just use linear combination
                # Baseline says:
                # running_mean = (1 - alpha) * self.running_mean + alpha * x.mean(dim=[0,2,3])
                # This suggests it computes a TEMPORARY running_mean for THIS batch?
                # "scale = self.weight * (running_var + self.eps).rsqrt()"
                # "bias = self.bias - running_mean * scale"
                # So it uses the blended statistic to normalize the CURRENT batch.
                # It does NOT seem to permanently update self.running_mean?
                # "self.running_mean" is used in the equation.
                # It doesn't say "self.running_mean = ...".
                
                # Re-reading baseline line 363:
                # running_mean = (1 - alpha) * self.running_mean + alpha * x.mean(dim=[0,2,3])
                # Yes, local variable.
                
                # BUT, does it persist? TENT usually updates the model.
                # If it's a local variable, it's just "Source-Weighted Inference".
                # The name in baseline is just "NORM".
                # I will implement it as local blending as per code.
                
                # Compute blended stats
                # If LayerNorm, it usually doesn't have running_mean/var. 
                # Baseline line 378: "elif isinstance(self, nn.LayerNorm): return nn.functional.layer_norm..."
                # Use standard LN for LayerNorm. Why? LayerNorm computes stats per sample/instantiation anyway.
                # So Norm Adaptation only applies to BN.
                
                if isinstance(_self, (nn.BatchNorm2d, FrozenBatchNorm2d)) or (RTDetrFrozenBatchNorm2d and isinstance(_self, RTDetrFrozenBatchNorm2d)):
                     # Use existing running stats if available
                     if not hasattr(_self, 'running_mean') or _self.running_mean is None:
                         # Fallback to standard
                         return original_forward(x)
                         
                     rm = _self.running_mean
                     rv = _self.running_var
                     
                     blended_mean = (1 - alpha) * rm + alpha * batch_mean
                     blended_var = (1 - alpha) * rv + alpha * batch_var
                     
                     eps = getattr(_self, 'eps', 1e-5)
                     
                     # Manual Normalization
                     scale = _self.weight * (blended_var + eps).rsqrt()
                     bias = _self.bias - blended_mean * scale
                     
                     scale = scale.reshape(1, -1, 1, 1)
                     bias = bias.reshape(1, -1, 1, 1)
                     
                     return x * scale + bias
                     
                else:
                     return original_forward(x)

            # Else standard forward
            return original_forward(x)
        
        module.forward = norm_forward.__get__(module, module.__class__)

    def forward(self, *args, **kwargs):
        # Adaptation happens inside layer forwards
        return self.base_model(*args, **kwargs)

    def reset(self, reset_stats=False):
        # Stats are not permanently updated in this logic (local variable), so just resetting config might be enough.
        # But if we did update stats, we would restore here.
        super().reset(reset_stats)


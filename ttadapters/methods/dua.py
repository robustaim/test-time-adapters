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
class DUAConfig(AdaptationConfig):
    adaptation_name: str = "DUAEngine"
    adaptation_layers: str = "backbone+encoder"
    min_momentum_constant: float = 0.0001
    decay_factor: float = 0.94
    mom_pre: float = 0.01


class DUAEngine(AdaptationEngine):
    """
    DUA (Dynamic Update Adaptation) Engine.
    
    Dynamically updates Batch Normalization statistics with a decaying momentum.
    """
    model_name: str = "DUAEngine"
    
    def __init__(self, base_model: BaseModel, config: DUAConfig):
        self.config: DUAConfig = config
        self.wrapped_layers = []
        self.original_forward_methods = {} # Backup
        
        super().__init__(base_model, config)

    def _post_init(self):
        self._apply_dua_adaptation()
    
    def online(self, mode=True):
        self.adapting = mode
        # DUA requires model to be in EVAL mode generally, but we manually update stats?
        # Original code: self.model.eval(), but wrapped forward updates running_mean/var.
        if mode:
            self.base_model.eval()
            self._set_dual_mode(True)
        else:
             self._set_dual_mode(False)
             
    def _set_dual_mode(self, active: bool):
        for module in self.wrapped_layers:
            module.dua_active = active

    def _apply_dua_adaptation(self):
        # Identify layers
        candidates = []
        for name, module in self.base_model.named_modules():
             if self._is_norm_layer(module):
                candidates.append((name, module))
        
        # Filter based on adaptation_layers config
        is_rtdetr = self.model_provider == ModelProvider.HuggingFace 
        
        filtered = []
        for name, module in candidates:
            should_adapt = False
            
            is_decoder = 'decoder' in name.lower()
            is_encoder = 'encoder' in name.lower() and not is_decoder
            
            # Helper for RTDETR vs others
            # Original code had specific checks for RTDetrFrozenBatchNorm2d in backbone
            if self.config.adaptation_layers == "backbone":
                if 'backbone' in name.lower() or 'bottom_up' in name.lower():
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
            print(f"[DUA] Wrapped {len(self.wrapped_layers)} layers for adaptation.")

    def _is_norm_layer(self, module: nn.Module) -> bool:
        if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
            return True
        if FrozenBatchNorm2d and isinstance(module, FrozenBatchNorm2d):
            return True
        if RTDetrFrozenBatchNorm2d and isinstance(module, RTDetrFrozenBatchNorm2d):
            return True
        return False

    def _wrap_layer(self, module):
        # We need to replace the forward method or wrap the class
        # Monkey patching forward is easier but "messy". 
        # CascadedNorm creates new classes. Let's do that for cleanliness if possible or just monkey patch like baseline.
        # Given "messy" complaint, let's try to be clean. But simpler to monkey patch instance method for DUA state.
        
        # Initialize DUA state
        module.dua_active = False       
        module.min_momentum_constant = self.config.min_momentum_constant
        module.decay_factor = self.config.decay_factor
        module.mom_pre = self.config.mom_pre
        
        if not hasattr(module, 'original_running_mean') and hasattr(module, 'running_mean'):
             module.original_running_mean = module.running_mean.clone()
             module.original_running_var = module.running_var.clone()
        
        # Keep original forward
        # self.original_forward_methods[module] = module.forward # This stores the bound method?
        # Actually simplest is to define a new forward function and bind it.
        
        original_forward = module.forward
        
        def dua_forward(_self, x):
            if hasattr(_self, 'dua_active') and _self.dua_active:
                with torch.no_grad():
                    current_momentum = _self.mom_pre + _self.min_momentum_constant
                    
                    if x.dim() == 4:
                         reduce_dims = (0, 2, 3)
                    else:
                         # Handle LayerNorm or others
                         if isinstance(_self, nn.LayerNorm):
                             nb_dims = len(_self.normalized_shape)
                             reduce_dims = tuple(range(0, x.dim() - nb_dims))
                         else:
                             reduce_dims = (0,)

                    batch_mean = x.mean(dim=reduce_dims)
                    batch_var = x.var(dim=reduce_dims, unbiased=True)
                    
                    # Update running stats
                    # For LN, running_mean/var might not exist?
                    # DUA baseline checks:
                    # if hasattr(self, 'running_mean'):
                    
                    if hasattr(_self, 'running_mean') and _self.running_mean is not None:
                        # Ensure shapes match (LayerNorm running mean might be missing or shape issue)
                        # BatchNorm: running_mean is (C). batch_mean is (C).
                         _self.running_mean.mul_(1 - current_momentum).add_(batch_mean, alpha=current_momentum)
                         _self.running_var.mul_(1 - current_momentum).add_(batch_var, alpha=current_momentum)
                    
                    _self.mom_pre *= _self.decay_factor

            # Standard forward
            # Need to handle FrozenBatchNorm behaviors if needed
            # For FrozenBatchNorm2d, it uses buffer stats usually.
            # DUA implementation explicitly calculates scale/bias using running_var/mean manually in the forward
            # because FrozenBatchNorm might ignore them or use fixed parameters?
            # Standard FrozenBatchNorm2d (Detectron) uses weight/bias and running_mean/var? 
            # Detectron FrozenBatchNorm2d: "Batch normalization statistics are FROZEN."
            # So if we update running_mean, it might not use it if we call original_forward?
            # Let's check DUA baseline implementation again.
            # It reimplements the normalization math!
            
            # Helper for manual norm
            def manual_norm(__self, __x):
                # We need __self.running_mean/var to be used.
                if isinstance(__self, (nn.BatchNorm2d, FrozenBatchNorm2d)) or (RTDetrFrozenBatchNorm2d and isinstance(__self, RTDetrFrozenBatchNorm2d)):
                     eps = getattr(__self, 'eps', 1e-5)
                     scale = __self.weight * (__self.running_var + eps).rsqrt()
                     bias = __self.bias - __self.running_mean * scale
                     scale = scale.reshape(1, -1, 1, 1)
                     bias = bias.reshape(1, -1, 1, 1)
                     return __x * scale + bias
                elif isinstance(__self, nn.LayerNorm):
                     return nn.functional.layer_norm(__x, __self.normalized_shape, __self.weight, __self.bias, __self.eps)
                return __x # Fallback
            
            # If DUA is active, we want to use the updated stats.
            # If we just call original_forward on FrozenBN, it might not use the updated stats?
            # Detectron FrozenBN source: "It uses computed mean and variance from the batch to update running stats?" NO. "Frozen" means it uses provided stats (track_running_stats=False usually or eval mode).
            # If we update the stats in place, and then clean BN uses them, it works.
            # BUT FrozenBN often might not register running_mean as a buffer that affects output in the same way?
            # Detectron2 FrozenBatchNorm2d simply mimics BN but with parameters frozen. It DOES use running_mean/var in forward.
            # Wait, Detectron2 FrozenBatchNorm2d documentation says "It extracts the statistics from valid batches... NO."
            # "running_mean and running_var are loaded from the checkpoint and never updated."
            # But if we update them manually, does `forward` use them?
            # `F.batch_norm(x, self.running_mean, self.running_var, ...)`
            # Yes it should.
            
            # HOWEVER, DUA baseline reimplemented the math manually (lines 536+ in baseline.py).
            # "scale = self.weight * ... "
            # This suggests they wanted to be absolutely sure or avoid some underlying behavior.
            # I will follow baseline and use manual calculation if DUA is active or if we successfully updated stats.
            
            if hasattr(_self, 'dua_active') and _self.dua_active:
                 return manual_norm(_self, x)
            else:
                 # If original_forward is bound method, it works?
                 # original_forward is `module.forward`.
                 return original_forward(x)
        
        # Bind new forward
        module.forward = dua_forward.__get__(module, module.__class__)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def reset(self, reset_stats=False):
        # Restore original running means
        for module in self.wrapped_layers:
             if hasattr(module, 'original_running_mean'):
                 module.running_mean.copy_(module.original_running_mean)
                 module.running_var.copy_(module.original_running_var)
             module.mom_pre = self.config.mom_pre
             
        # Reset stats in base
        super().reset(reset_stats)


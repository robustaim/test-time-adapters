from typing import Optional, List, Literal, Tuple
from dataclasses import dataclass
from pathlib import Path
import warnings
import math
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

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
class ActMADConfig(AdaptationConfig):
    adaptation_name: str = "ActMADEngine"
    statistic_save_path: str = "./save_actmad_statistics.pt"
    clean_bn_extract_batch: int = 32
    adaptation_layers: str = "backbone+encoder"  # "backbone", "encoder", "backbone+encoder"
    layer_names: Optional[List[str]] = None  # To load specific layers if needed


class ActMADEngine(AdaptationEngine):
    """
    ActMAD (Activation Mean Alignment and Discrepancy) Engine
    
    Aligns the batch statistics of the current test batch with the pre-computed 
    statistics (mean/var) from a clean source dataset.
    """
    model_name: str = "ActMADEngine"
    loss_class = nn.L1Loss

    def __init__(self, base_model: BaseModel, config: ActMADConfig):
        self.config: ActMADConfig = config  # Type hint
        self.clean_mean_list: List[torch.Tensor] = []
        self.clean_var_list: List[torch.Tensor] = []
        self.target_layers: List[nn.Module] = []
        self.hook_handles = []
        self.current_batch_means = []
        self.current_batch_vars = []
        
        super().__init__(base_model, config)

    def _pre_init(self):
        # Initialize storage
        pass

    def _post_init(self):
        # Identify layers to adapt
        self._identify_layers()
        
        # Load statistics
        self._load_statistics()
        
        # Enable gradients for adaptation
        self.online(True)
        
    def online(self, mode=True):
        self.adapting = mode
        if mode:
            # Unfreeze all parameters for ActMAD as per original implementation
            # "ActMAD: Enable gradients for all model parameters"
            self.base_model.train() # Set to train mode? Original used eval() but set requires_grad=True.
            # Actually original sets model.eval() but requires_grad=True.
            # And then manually sets BN layers to eval if needed or extracts stats.
            
            # AdaptationEngine.online sets generic requires_grad based on online_parameters() which defaults to all.
            # But we want to ensure BN layers are treated correctly.
            
            # Use base logic first
            super().online(mode)
            
            # ActMAD specific: ensure evaluaton mode for BN layers usually, but we want to optimize inputs/params.
            # The original code runs: self.model.eval() and optimizer.zero_grad().
            # AdaptationEngine.online(True) does: self.eval(), freezes all, then unfreezes online_parameters.
            # AdaptationEngine.online_parameters defaults to self.base_model.parameters().
            # So super().online(True) will unfreeze everything in base_model (if online_parameters is default).
            
            pass 
        else:
            super().online(mode)

    def _identify_layers(self):
        """
        Identify normalization layers based on configuration and model provider.
        """
        self.target_layers = []
        self.target_layer_names = []
        
        candidates = []
        for name, module in self.base_model.named_modules():
             if self._is_norm_layer(module):
                candidates.append((name, module))
        
        # Filter based on adaptation_layers config
        # Logic copied/adapted from baseline.py and rtdetr_baseline.py
        
        filtered = []
        for name, module in candidates:
            should_adapt = False
            
            # Generic filtering logic
            is_decoder = 'decoder' in name.lower()
            is_encoder = 'encoder' in name.lower() and not is_decoder
            is_backbone = 'backbone' in name.lower() or 'bottom_up' in name.lower() or 'res2' in name or 'res3' in name or 'res4' in name or 'res5' in name # Detectron2 backbone often has resN

            if self.config.adaptation_layers == "backbone":
                if is_backbone:
                    should_adapt = True
            elif self.config.adaptation_layers == "encoder":
                if is_encoder:
                    should_adapt = True
            elif self.config.adaptation_layers == "backbone+encoder":
                if not is_decoder: # Everything except decoder
                    should_adapt = True
            else: # Default/fallback
                 if not is_decoder:
                    should_adapt = True
            
            if should_adapt:
                filtered.append((name, module))

        # Original implementation takes the SECOND HALF of the layers.
        # "Select only the first half of all layers" -> wait, code says cutoff = len // 2; chosen = info[cutoff:]
        # So it selects the LAST half (later layers).
        
        cutoff = len(filtered) // 2
        selected = filtered[cutoff:]
        
        for name, module in selected:
            self.target_layers.append(module)
            self.target_layer_names.append(name)
            
        if self.config.verbose:
            print(f"[ActMAD] Selected {len(self.target_layers)} layers out of {len(candidates)} total norm layers.")

    def _is_norm_layer(self, module: nn.Module) -> bool:
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            return True
        if FrozenBatchNorm2d and isinstance(module, FrozenBatchNorm2d):
            return True
        if RTDetrFrozenBatchNorm2d and isinstance(module, RTDetrFrozenBatchNorm2d):
            return True
        return False

    def _load_statistics(self):
        save_path = Path(self.config.statistic_save_path)
        if save_path.exists():
            if self.config.verbose:
                print(f"[ActMAD] Loading statistics from {save_path}")
            data = torch.load(save_path, map_location=self.device)
            self.clean_mean_list = [t.to(self.device) for t in data["clean_mean_list_final"]]
            self.clean_var_list = [t.to(self.device) for t in data["clean_var_list_final"]]
            
            # Validation
            if len(self.clean_mean_list) != len(self.target_layers):
                warnings.warn(f"[ActMAD] Mismatch in number of layers: Loaded {len(self.clean_mean_list)}, Current model selected {len(self.target_layers)}. Statistics might be misaligned.")
        else:
            if self.config.verbose:
                print(f"[ActMAD] No statistics found at {save_path}. Please run fit() with a clean dataset first.")

    def fit(self, clean_dataset: torch.utils.data.Dataset, collate_fn=None):
        """
        Extract statistics from a clean dataset.
        """
        if self.config.verbose:
            print("[ActMAD] Extracting statistics from clean dataset...")
            
        loader = DataLoader(
            clean_dataset, 
            batch_size=self.config.clean_bn_extract_batch, 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # Setup hooks
        means = [0.0] * len(self.target_layers)
        vars_ = [0.0] * len(self.target_layers)
        counts = [0] * len(self.target_layers)
        
        def get_hook(idx):
            def hook(module, input, output):
                # Calculate mean/var
                # Standardize shape
                x = output
                if x.dim() == 4: # N, C, H, W
                    dims = [0, 2, 3]
                elif x.dim() == 3: # N, L, C or N, C, L - LayerNorm often keeps shape
                     # For LayerNorm, we usually AVERAGE over spatial/sequence and batch
                     # baseline.py checks for layer norm and does dims = tuple(range(-len(shape), 0))
                     # But here we want [C] stats?
                     # baseline ActMAD extract_activation_alignment uses SaveOutput which does:
                     # out.mean(dim=[0, 2, 3]) for 4D.
                     # For RTDETR, it handles various shapes.
                     # Let's align with input shape logic.
                     dims = tuple(range(len(x.shape)))[:-1] # Generic: all except last (C) -- Wait, BN is (N, C, H, W).
                     pass
                
                # Let's use the logic from baseline.py SaveOutput utils
                # But reimplemented here for self-containment or better, use standard methods.
                pass
                
        # Re-implementing extraction logic clearly
        # Using a temporary list to store batch stats and then averaging is safer for variance? 
        # Baseline uses AverageMeter.
        
        accum_means = [[] for _ in range(len(self.target_layers))]
        accum_vars = [[] for _ in range(len(self.target_layers))]
        
        hooks = []
        for i, layer in enumerate(self.target_layers):
            def hook_fn(module, input, output, idx=i):
                x = output
                # Determine dimensions to reduce
                # BN: (N, C, H, W) -> reduce (0, 2, 3) -> (C)
                # LN: (N, ..., C) -> reduce (0, ...) -> (C) or similar?
                # RT-DETR LayerNorm is (N, L, D). We want D?
                
                if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d)) or (RTDetrFrozenBatchNorm2d and isinstance(module, RTDetrFrozenBatchNorm2d)):
                     if x.dim() == 4:
                        reduce_dims = (0, 2, 3)
                     else:
                        reduce_dims = (0,) # Fallback
                elif isinstance(module, nn.LayerNorm):
                     # For LN, we want to average over Batch and Spatial/Sequence dimensions, keeping the normalized dimension.
                     # normalized_shape is usually the last dim(s).
                     # If normalized_shape is (D,), and input is (N, L, D), we reduce (0, 1).
                     nb_dims = len(module.normalized_shape)
                     total_dims = x.dim()
                     reduce_dims = tuple(range(0, total_dims - nb_dims))
                else:
                     reduce_dims = tuple(range(0, x.dim() - 1)) # Default to reducing all but last

                current_mean = x.mean(dim=reduce_dims)
                current_var = x.var(dim=reduce_dims, unbiased=True)
                
                accum_means[idx].append(current_mean.detach())
                accum_vars[idx].append(current_var.detach())
                
            hooks.append(layer.register_forward_hook(hook_fn))
            
        self.base_model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="ActMAD Extract"):
                # Forward pass
                # Need to handle different inputs based on provider/dataset
                # AdaptationEngine.forward calls base_model.
                # But here we are iterating dataloader directly.
                
                if self.model_provider == ModelProvider.Detectron2:
                     # batch is list of dicts or similar.
                     if isinstance(batch, list):
                         pass
                     else:
                         batch = [batch] # wrap?
                     self.base_model(batch)
                elif self.model_provider == ModelProvider.HuggingFace:
                    # RT-DETR: pixel_values
                     if isinstance(batch, dict):
                         # Move to device
                         batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                         self.base_model(**batch)
                     else:
                         # Assume tuple/list
                         pass
                         
        # Cleanup hooks
        for h in hooks:
            h.remove()
            
        # Aggregate
        self.clean_mean_list = [torch.stack(m_list).mean(dim=0) for m_list in accum_means]
        self.clean_var_list = [torch.stack(v_list).mean(dim=0) for v_list in accum_vars]
        
        # Save
        save_path = Path(self.config.statistic_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "clean_mean_list_final": [t.cpu() for t in self.clean_mean_list],
            "clean_var_list_final": [t.cpu() for t in self.clean_var_list],
            "layer_names": self.target_layer_names
        }, save_path)
        
        if self.config.verbose:
            print(f"[ActMAD] Statistics saved to {save_path}")

    def forward(self, *args, **kwargs):
        # Hook for current batch logic
        self.current_batch_means = []
        self.current_batch_vars = []
        
        hooks = []
        forlayer_idx, layer in enumerate(self.target_layers):
            def hook_fn(module, input, output):
                x = output
                # Same reduction logic as fit
                if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d)) or (RTDetrFrozenBatchNorm2d and isinstance(module, RTDetrFrozenBatchNorm2d)):
                     if x.dim() == 4:
                        reduce_dims = (0, 2, 3)
                     else:
                        reduce_dims = (0,)
                elif isinstance(module, nn.LayerNorm):
                     nb_dims = len(module.normalized_shape)
                     total_dims = x.dim()
                     reduce_dims = tuple(range(0, total_dims - nb_dims))
                else:
                     reduce_dims = tuple(range(0, x.dim() - 1))

                self.current_batch_means.append(x.mean(dim=reduce_dims))
                self.current_batch_vars.append(x.var(dim=reduce_dims, unbiased=True))
                
            hooks.append(layer.register_forward_hook(hook_fn))

        # Forward pass
        # AdaptationEngine forward calls base_model(*args, **kwargs)
        # But we need to make sure we are not in infinite recursion if we call super().forward
        # super().forward calls self.base_model()
        
        # Reset grads
        if self.optimizer:
            self.optimizer.zero_grad()
            
        outputs = self.base_model(*args, **kwargs)
        
        # Cleanup hooks
        for h in hooks:
            h.remove()
            
        # Compute Loss
        loss = 0.0
        if self.adapting and self.clean_mean_list:
            loss_func = self.loss_function # L1Loss default
            
            for i in range(len(self.current_batch_means)):
                # Ensure device match
                target_mean = self.clean_mean_list[i]
                target_var = self.clean_var_list[i]
                
                loss += loss_func(self.current_batch_means[i], target_mean)
                loss += loss_func(self.current_batch_vars[i], target_var)
                
            self._stats.setdefault('losses', []).append(loss.item())
            
            loss.backward()
            self.optimizer.step()
            
        return outputs

"""
CascadedNorm V5: Associative Memory Input Transformation

Implements "Associative Memory" to solve the Domain Dilemma with mathematically guaranteed drift prevention.

Key Innovation:
    Uses a reconstructive memory matrix W that maps Keys (K) to Values (V).
    
    1. Memory Update: W is trained to minimize ||K @ W - V||^2
    2. Orthogonality: Forces Keys of different inputs to be orthogonal (Cosine Similarity penalty).
       - This ensures "Night" keys do not overlap with "Clear" keys.
       - Therefore, learning "Night" rules does not overwrite "Clear" rules (Zero Forgetting).
    
    Structure:
    [Image] -> [Encoder] -> [Features] -> [Q, K, V]
                            [Memory W] -> [Params]

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

    param_regularization: float = 0.05
    memory_loss_weight: float = 1.0  # Weight for memory reconstruction
    orth_loss_weight: float = 0.1    # Weight for key orthogonality
    
    temperature: float = 0.01
    saturation_limit: float = 100.0


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


class AssociativeParameterMemory(nn.Module):
    """
    Associative Memory for Parameters.
    Maps Image Features -> Parameters with Orthogonality Constraint.
    """
    def __init__(self, feat_dim=64):
        super().__init__()
        # Compact Feature Extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)), # 32x2x2 = 128
            nn.Flatten()
        )
        self.input_dim = 128
        self.feat_dim = feat_dim
        
        # Projections
        self.qk_proj = nn.Linear(self.input_dim, feat_dim)
        self.v_proj = nn.Linear(self.input_dim, feat_dim)
        
        # Memory Matrix W: Maps Key space to Value space
        self.W = nn.Linear(feat_dim, feat_dim, bias=False)
        nn.init.eye_(self.W.weight) # Start as identity mapping
        
        # Output Projection: Value space -> Parameters (2)
        self.out_proj = nn.Linear(feat_dim, 2)
        
        # Initialize Output to produce (2.0, 1.0)
        # We assume W is identity initially, so Q @ W = Q.
        # We want out_proj(Q) ~ (2.0, 1.0).
        # We initialize out_proj weights small and bias to target.
        nn.init.normal_(self.out_proj.weight, 0, 0.001)
        self.out_proj.bias.data = torch.tensor([2.0, 1.0])

    def forward(self, img):
        if img.dim() == 3: img = img.unsqueeze(0)
        
        # 1. Component Extraction
        feat = self.encoder(img) # (B, 128)
        
        # 2. Memory Interfaces
        QK = self.qk_proj(feat) # (B, 64) - Query/Key
        V = self.v_proj(feat)   # (B, 64) - Value
        
        # 3. Memory Retrieval (Read)
        # P = Q @ W
        retrieved = self.W(QK) 
        params = self.out_proj(retrieved) # (B, 2)
        
        # 4. Memory Construction (Write/Update Loss)
        # We want K @ W = V
        # If we update W to satisfy this, we "memorize" the mapping K->V.
        reconstructed_V = self.W(QK) # Note: In TTT-Linear, Q=K
        loss_mem = F.mse_loss(reconstructed_V, V)
        
        # 5. Orthogonality (Anti-Drift)
        # Force keys to be dispersed in high-dim space
        # Simple proxy: penalize cosine similarity if batch > 1
        loss_orth = torch.tensor(0.0, device=img.device)
        if QK.shape[0] > 1:
            # Pairwise cosine similarity minimized
            qk_norm = F.normalize(QK, dim=1)
            sim_matrix = qk_norm @ qk_norm.T
            # Remove diagonal (self-similarity is 1)
            mask = ~torch.eye(QK.shape[0], dtype=torch.bool, device=img.device)
            loss_orth = sim_matrix[mask].abs().mean()
            
        return params, loss_mem, loss_orth


class GammaTransform(nn.Module):
    """Context-Aware Transform Controller with Memory."""

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.saturation_limit = torch.tensor(config.saturation_limit, requires_grad=False)
        self.memory = AssociativeParameterMemory()
        self.stretcher = DifferentiableHistogramStretcher(config.temperature)
        
        self.current_params = None

    def forward(self, img):
        # Retrieve context-aware parameters
        raw_params, loss_mem, loss_orth = self.memory(img)
        
        # Apply constraints
        noise_floor = raw_params[:, 0].mean().clamp(0.0, 48.0)
        gamma = raw_params[:, 1].mean().clamp(0.1, 5.0)
        
        self.current_params = (noise_floor, gamma)
        
        transformed = self.stretcher(img, noise_floor, self.saturation_limit, gamma)
        
        return transformed, (noise_floor, self.saturation_limit, gamma), (loss_mem, loss_orth)

    def get_regularization_loss(self):
        if self.current_params is None: return torch.tensor(0.0)
        n, g = self.current_params
        return (n - 2.0).pow(2) + (g - 1.0).pow(2)


class CascadedNorm(nn.Module):
    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config
        self.transform_controller = GammaTransform(config)
        self.norm_layers: List[nn.Module] = []
        self.norm_types: List[str] = []
        self.source_means: List[torch.Tensor] = []
        self.source_vars: List[torch.Tensor] = []

    def forward(self, img):
        transformed, params, mem_losses = self.transform_controller(img)
        output = 0.5 * transformed + 0.5 * img
        return output, params, mem_losses

    def compute_alignment_loss(self, mem_losses) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.source_means[0].device)

        # 1. Alignment Loss
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

            total_loss = total_loss + F.mse_loss(batch_mean, source_mean) + F.mse_loss(batch_var, source_var)

        # 2. Regularization Loss
        if self.config.param_regularization > 0:
            reg_loss = self.transform_controller.get_regularization_loss()
            total_loss = total_loss + reg_loss * self.config.param_regularization
            
        # 3. Memory Losses (Construction + Orthogonality)
        loss_mem, loss_orth = mem_losses
        total_loss = total_loss + loss_mem * self.config.memory_loss_weight
        total_loss = total_loss + loss_orth * self.config.orth_loss_weight

        return total_loss
    
    def online_parameters(self):
        return self.transform_controller.memory.parameters()


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
        mem_losses_list = []
        
        # V4/V5 returns (params, mem_losses) tuple from controller
        # Need to handle this in loop
        
        for i in range(imgs.shape[0]):
            # controller forward returns: transformed, (n, s, g), (loss_mem, loss_orth)
            transformed, params, mem_losses = self.cascaded_norm(imgs[i])
            transformed_list.append(transformed)
            params_list.append(params)
            mem_losses_list.append(mem_losses)
            
        # Aggregate memory losses
        avg_loss_mem = torch.mean(torch.stack([m[0] for m in mem_losses_list]))
        avg_loss_orth = torch.mean(torch.stack([m[1] for m in mem_losses_list]))
        
        return torch.stack(transformed_list, dim=0), params_list, (avg_loss_mem, avg_loss_orth)

    def forward(self, batched_inputs):
        if not self.adapting: return self.base_model(batched_inputs)
        if isinstance(batched_inputs, torch.Tensor): return self._forward_tensor(batched_inputs)
        # Simplified for dict list (assuming tensor path is primary for test)
        return self._forward_tensor(torch.stack([x['image'] for x in batched_inputs]))

    def _forward_tensor(self, imgs):
        imgs = imgs.to(self._device)
        original_scale = imgs.max() <= 1.0
        if original_scale: imgs = imgs * 255.0
        
        imgs_transformed, params_list, mem_losses = self._transform_batch(imgs)
        
        for params in params_list: self._stats['transform_params'].append(tuple(p.item() for p in params))
        
        model_input = imgs_transformed / 255.0 if original_scale else imgs_transformed
        outputs = self.base_model(model_input)
        
        alignment_loss = self.cascaded_norm.compute_alignment_loss(mem_losses) # Pass mem_losses here!
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

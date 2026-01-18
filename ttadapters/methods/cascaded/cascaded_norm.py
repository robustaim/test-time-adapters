""" CascadedNorm: Input Transformation for BN Statistics Alignment """

from typing import Optional, List, Literal
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ..base import AdaptationEngine, AdaptationConfig
from ...models.base import BaseModel


@dataclass
class CascadedNormConfig(AdaptationConfig):
    """
    Configuration for CascadedNorm adaptation.

    Adaptation Methods:
    ------------------
    1. Gamma Clamp (default)
        - Standard gamma correction with percentile-based clipping
        - use_percentile=False: Simple clipping
        - use_percentile=True: Percentile-adaptive clipping

    2. LUT (Look-Up Table)
        - LUT-based transformation
        - lut_monotonic=True: Enforce monotonic constraint
        - lut_ema=True: Use exponential moving average
        - lut_reg=True: Add smoothness regularization

    Cascade Modes:
    -------------
    1. 'all': Use all Normalization layers (default)
    2. 'single': Use only last Normalization layer (fastest)
    3. 'selected': Use specified Normalization layers by indices

    Examples:
    --------
    # Basic gamma clamp
    config = CascadedNormConfig(
        adaptation_method='gamma_clamp',
        cascade_mode='single'
    )

    # Gamma clamp with percentile
    config = CascadedNormConfig(
        adaptation_method='gamma_clamp',
        use_percentile=True,
        cascade_mode='all'
    )

    # LUT with all options
    config = CascadedNormConfig(
        adaptation_method='lut',
        lut_monotonic=True,
        lut_ema=True,
        lut_reg=True,
        cascade_mode='selected',
        cascade_indices=[0, 2, 4]
    )
    """
    adaptation_name: str = "CascadedNormEngine"

    # ==================== Adaptation Method ====================
    adaptation_method: Literal["gamma_clamp", "lut"] = "gamma_clamp"

    # Gamma Clamp options
    use_percentile: bool = False
    percentile_value: float = 95.0

    # LUT options
    lut_monotonic: bool = False  # Enforce monotonic constraint
    lut_ema: bool = False        # Use exponential moving average
    lut_reg: bool = False        # Add regularization loss
    lut_size: int = 256          # LUT table size (8bit)
    lut_ema_momentum: float = 0.9
    lut_reg_weight: float = 0.01

    # ==================== Cascade Mode ====================
    cascade_mode: Literal["all", "single", "single_last", "selected"] = "all"
    cascade_indices: Optional[List[int]] = None

    # ==================== Transformation ====================
    temperature: float = 0.01  # For differentiable percentile


class DifferentiableHistogramStretcher(nn.Module):
    """
    Vectorized differentiable histogram stretching for high-precision statistics.
    Optimized for resolutions like 800x1280 (~1M pixels).
    """

    def __init__(self, temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature

    def soft_percentile(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Vectorized differentiable percentile approximation using all pixels.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).
            p (torch.Tensor): Target percentile value (scalar).

        Returns:
            torch.Tensor: Calculated percentiles of shape (B, C).
        """
        B, C, H, W = x.shape
        # Flatten spatial dimensions to (B, C, 1024000)
        x_flat = x.view(B, C, -1)
        N = x_flat.shape[-1]

        # Calculate target index in the sorted array
        idx = (p / 100.0) * (N - 1)

        # Softmax-based weights around the target index
        # This handles the differentiability of the percentile selection
        indices = torch.arange(N, device=x.device, dtype=x.dtype)
        weights = F.softmax(-(indices - idx).abs() / (self.temperature * N), dim=-1)
        weights = weights.view(1, 1, -1)

        # Sort all pixels along the flattened spatial dimension
        # Modern GPUs handle 1M elements efficiently with CUDA sort
        sorted_x, _ = torch.sort(x_flat, dim=-1)

        # Weighted sum returns the approximate percentile value per channel: (B, C)
        return (weights * sorted_x).sum(dim=-1)

    def forward(self, image: torch.Tensor, clip_low: torch.Tensor, clip_high: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """
        Apply vectorized stretching and gamma correction to the entire batch.
        """
        # Ensure 4D shape (B, C, H, W) even for single image input
        if image.dim() == 3:
            image = image.unsqueeze(0)

        B, C, H, W = image.shape

        # 1. Compute precise percentiles for all channels simultaneously
        low_val = self.soft_percentile(image, clip_low)
        high_val = self.soft_percentile(image, clip_high)

        # 2. Reshape for broadcasting to the spatial dimensions
        low_val_bc = low_val.view(B, C, 1, 1)
        high_val_bc = high_val.view(B, C, 1, 1)

        # 3. Differentiable Clipping using Softplus to keep gradients alive
        scale = 50.0
        clipped = low_val_bc + F.softplus((image - low_val_bc) * scale) / scale
        clipped = high_val_bc - F.softplus((high_val_bc - clipped) * scale) / scale

        # 4. Normalization to [0, 1] and Gamma Correction
        range_val = high_val_bc - low_val_bc + 1e-6
        normalized = (clipped - low_val_bc) / range_val
        gamma_corrected = torch.pow(normalized + 1e-6, gamma)

        # 5. Scale back to [0, 255]
        return torch.clamp(gamma_corrected * 255.0, 0, 255)


class GammaTransform(nn.Module):
    """Learnable gamma parameters for CascadedNorm"""

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config

        self.clip_low = nn.Parameter(torch.tensor(2.0))
        self.clip_high = nn.Parameter(torch.tensor(98.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))

        # Percentile tracking
        if self.config.use_percentile:
            self.register_buffer('percentile_history', torch.zeros(100, 3))  # RGB
            self.register_buffer('history_idx', torch.tensor(0, dtype=torch.long))

        self.stretcher = DifferentiableHistogramStretcher(config.temperature)

    def forward(self):
        """Forward for gamma clamp."""
        clip_low = torch.sigmoid(self.clip_low) * 10
        clip_high = 90 + torch.sigmoid(self.clip_high) * 10
        gamma = 0.5 + torch.sigmoid(self.gamma) * 1.5

        # Apply percentile
        if self.config.use_percentile and self.training:
            gamma = self._apply_percentile_adaptation(gamma)

        return clip_low, clip_high, gamma

    def _apply_percentile_adaptation(self, gamma):
        """Apply percentile-based adaptation."""
        idx = self.history_idx.item()
        self.percentile_history[idx] = gamma.detach()
        self.history_idx.copy_((idx + 1) % 100)

        valid_size = min(idx + 1, 100)
        if valid_size > 10:
            valid_history = self.percentile_history[:valid_size]
            gamma = torch.quantile(valid_history,
                self.config.percentile_value / 100.0, dim=0)

        return gamma


class LUTTransform(nn.Module):
    """Learnable look-up tables for CascadedNorm"""

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config

        self.register_buffer('lut_table',
            torch.linspace(0, 255, self.config.lut_size))
        self.lut_values = nn.Parameter(
            torch.linspace(0, 255, self.config.lut_size)
        )

        # EMA
        if self.config.lut_ema:
            self.register_buffer('lut_ema_values',
                torch.linspace(0, 255, self.config.lut_size))

    def forward(self):
        """Forward for LUT."""
        # Enforce monotonic
        if self.config.lut_monotonic and self.training:
            self._enforce_monotonic()

        # Return EMA or current values
        if self.config.lut_ema and self.training:
            return self.lut_ema_values
        return self.lut_values


class CascadedNorm(nn.Module):
    """
    CascadedNorm: Cascaded Input Distribution Normalization
        by Statistical Norm Layer Alignment

    Learns to normalize input pixel distribution
    with cascaded layer-wise supervision.

    The term "normalization" refers to distribution alignment
    between source and target domains.
    This differs from layer normalization methods (BN, LN)
    which perform statistical normalization of activations.

    Args:
        config: CascadedNormConfig with learning parameters
    """

    def __init__(self, config: CascadedNormConfig):
        super().__init__()

        self.config = config
        self.adaptation_method = config.adaptation_method

        if config.adaptation_method == 'gamma_clamp':
            self.transform_controller = GammaTransform(config)
        elif config.adaptation_method == 'lut':
            self.transform_controller = LUTTransform(config)

        self.norm_layers: nn.ModuleList[nn.Module] = nn.ModuleList()
        self.source_means: List[torch.Tensor] = []
        self.source_vars: List[torch.Tensor] = []



    def hook_fn(module, input, output):
        # Skip if cache is valid
        if self._cache_valid and idx in self._stats_cache:
            self.batch_stats[idx] = self._stats_cache[idx]
            return

        x = input[0]

        # Compute statistics based on norm type
        if norm_type in ['bn', 'frozen_bn']:
            if x.dim() == 4:  # (B, C, H, W)
                dims = (0, 2, 3)
            elif x.dim() == 3:  # (B, C, L)
                dims = (0, 2)
            else:  # (B, C)
                dims = (0,)
            mean = x.mean(dim=dims)
            var = x.var(dim=dims, unbiased=False)

        elif norm_type == 'ln':
            if hasattr(module, 'normalized_shape'):
                dims = tuple(range(-len(module.normalized_shape), 0))
                mean = x.mean(dim=dims)
                var = x.var(dim=dims, unbiased=False)
            else:
                mean = x.mean()
                var = x.var(unbiased=False)

        # Cache and store
        stats = {'mean': mean, 'var': var}
        self.batch_stats[idx] = stats
        self._stats_cache[idx] = stats



    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.config.adaptation_method == 'gamma_clamp':
            clip_low, clip_high, gamma = self.transform_controller()
            return self.stretcher(img, clip_low, clip_high, gamma)
        elif self.config.adaptation_method == 'lut':
            lut_values = self.transform_controller()
            # Apply LUT
            img_normalized = (img / 255.0 * (len(lut_values) - 1))
            img_int = img_normalized.long().clamp(0, len(lut_values) - 1)
            return lut_values[img_int]

    def _enforce_monotonic(self):
        """Enforce monotonic constraint on LUT."""
        with torch.no_grad():
            for i in range(1, len(self.lut_values)):
                if self.lut_values[i] < self.lut_values[i-1]:
                    self.lut_values[i].copy_(self.lut_values[i-1])

    def get_lut_regularization(self):
        """Compute LUT regularization loss."""
        if not self.config.lut_reg or self.adaptation_method != 'lut':
            return torch.tensor(0.0, device=self.lut_values.device)

        # Smoothness regularization
        diff2 = self.lut_values[2:] - 2 * self.lut_values[1:-1] + self.lut_values[:-2]
        return self.config.lut_reg_weight * (diff2 ** 2).mean()

    def update_lut_ema(self):
        """Update EMA for LUT."""
        if self.config.lut_ema and self.training and self.adaptation_method == 'lut':
            with torch.no_grad():
                self.lut_ema_values.copy_(
                    self.config.lut_ema_momentum * self.lut_ema_values +
                    (1 - self.config.lut_ema_momentum) * self.lut_values
                )



    @property
    def loss(self) -> torch.Tensor:
        """Compute BN statistics alignment loss (optimized)."""
        # Collect all statistics in batched tensors
        batch_means = []
        batch_vars = []
        source_means = []
        source_vars = []

        for stats, src_mean, src_var in zip(
                self.hook_manager.batch_stats,
                self.norm_extractor.source_means,
                self.norm_extractor.source_vars
        ):
            if stats['mean'] is not None:
                # Reduce to scalar for efficiency
                batch_means.append(stats['mean'].mean())
                batch_vars.append(stats['var'].mean())
                source_means.append(src_mean.mean())
                source_vars.append(src_var.mean())

        # Batched computation (single graph node)
        if len(batch_means) > 0:
            batch_means = torch.stack(batch_means)
            batch_vars = torch.stack(batch_vars)
            source_means = torch.stack(source_means).to(batch_means.device)
            source_vars = torch.stack(source_vars).to(batch_vars.device)

            mean_loss = F.mse_loss(batch_means, source_means)
            var_loss = F.mse_loss(batch_vars, source_vars)

            return mean_loss + var_loss
        
        

        # Update EMA if using LUT
        if self.config.lut_ema:
            self.transform_controller.update_lut_ema()

        return torch.tensor(0.0, device=self._device)




class CascadedNormEngine(AdaptationEngine):
    """
    Cascaded Norm Adaptation Engine
    Injects CascadedNorm module into the base model.

    Supports:
    - nn.BatchNorm2d / nn.BatchNorm1d / FrozenBatchNorm2d
    - nn.LayerNorm

    Args:
        base_model: Pre-trained model
        config: CascadedNormConfig with learning parameters

    Example:
        >>> adaptive_model = CascadedNormEngine(base_model, config)
        >>> otuput = adaptive_model(batch)
    """
    model_name: str = "CascadedNormEngine"

    def __init__(self, base_model: BaseModel, config: CascadedNormConfig):
        self.dist_norm = None  # will be overridden by _pre_init()
        self.dist_norm_state = None
        self.config = config

        super().__init__(base_model, config)

    def _pre_init(self):
        # Transformation modules
        self.dist_norm = CascadedNorm(self.config)

    def _post_init(self):
        self.dist_norm.to(self.device)
        self.dist_norm.to(self.dtype)
        self.dist_norm_state = {key: value.requires_grad for key, value in self.dist_norm.items()}

        # Extract norm layers
        if config.verbose:
            print(f"\n[CascadedNorm] Extracting norm layers...")
            print(f"  Cascade mode: {config.cascade_mode}")
            if config.cascade_indices:
                print(f"  Cascade indices: {config.cascade_indices}")

        if config.verbose:
            print(f"\n[CascadedNorm] Summary:")
            print(f"  Total layers: {len(self.norm_extractor.norm_layers)}")
            print(f"  BN: {self.norm_extractor.norm_types.count('bn')}")
            print(f"  FrozenBN: {self.norm_extractor.norm_types.count('frozen_bn')}")
            print(f"  LN: {self.norm_extractor.norm_types.count('ln')}")

    def _extract_norm_layers(self, model: nn.Module):
        """Find all normalization layers including FrozenBatchNorm."""
        for name, module in model.named_modules():
            module_type = type(module).__name__

            # Regular BatchNorm
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                self.norm_layers.append(module)
                self.norm_types.append('bn')
                self.source_means.append(module.running_mean.clone())
                self.source_vars.append(module.running_var.clone())
                if self.config.verbose:
                    print(f"  [BN] {name}: {module.num_features} channels")

            # LayerNorm
            elif isinstance(module, nn.LayerNorm):
                self.norm_layers.append(module)
                self.norm_types.append('ln')
                # LN targets normalized distribution
                self.source_means.append(torch.tensor(0.0))
                self.source_vars.append(torch.tensor(1.0))
                if self.config.verbose:
                    print(f"  [LN] {name}")

            # FrozenBatchNorm (Detectron2)
            elif 'FrozenBatchNorm' in module_type or \
                    ('Norm' in module_type and
                     hasattr(module, 'weight') and
                     hasattr(module, 'bias') and
                     not isinstance(module, nn.LayerNorm)):
                self.norm_layers.append(module)
                self.norm_types.append('frozen_bn')
                # FrozenBN stores stats in weight/bias
                self.source_means.append(module.weight.clone())
                self.source_vars.append(module.bias.clone())
                if self.config.verbose:
                    print(f"  [FrozenBN] {name}")

    def _filter_by_cascade_mode(self):
        """Filter layers based on cascade mode."""
        if self.config.cascade_mode == 'all':
            pass  # Use all

        elif self.config.cascade_mode == 'single':
            # Use only last layer
            if len(self.norm_layers) > 0:
                self.norm_layers = [self.norm_layers[-1]]
                self.norm_types = [self.norm_types[-1]]
                self.source_means = [self.source_means[-1]]
                self.source_vars = [self.source_vars[-1]]

        elif self.config.cascade_mode == 'selected':
            # Use selected indices
            if self.config.cascade_indices is not None:
                indices = self.config.cascade_indices
                self.norm_layers = [self.norm_layers[i] for i in indices
                                    if i < len(self.norm_layers)]
                self.norm_types = [self.norm_types[i] for i in indices
                                   if i < len(self.norm_types)]
                self.source_means = [self.source_means[i] for i in indices
                                     if i < len(self.source_means)]
                self.source_vars = [self.source_vars[i] for i in indices
                                    if i < len(self.source_vars)]

    def online_parameters(self):
        return self.dist_norm.online_parameters()

    def _reset_stats(self):
        # Statistics tracking
        self._stats = {
            'losses': [],
            'transform_params': [],
            'config': {
                'adaptation_method': self.adaptation_method,
                'cascade_mode': self.config.cascade_mode,
                'num_layers': len(self.norm_extractor.norm_layers),
            }
        }

    def reset(self, reset_stats=False):
        self.dist_norm.load_state_dict(self.dist_norm_state)
        self.online(self.adapting)
        self.to(self.device)
        self.to(self.dtype)
        try:
            self.optimizer.zero_grad()
        except:
            pass
        if reset_stats:
            current_stats = self._stats
            self._reset_stats()
            return current_stats
        else:
            return None

    def forward(self, *args, **kwargs):
        self.optimizer.zero_grad()

        # Forward through model
        result = self.base_model(*args, **kwargs)

        # Compute losses (batched for efficiency)
        loss = self.dist_norm.loss
        if self.

        # Backward and update
        loss.backward()
        self.optimizer.step()

        # Log statistics
        self.stats['losses'].append(loss.item())

        # Log transform parameters
        if self.config.adaptation_method == 'gamma_clamp':
            clip_low, clip_high, gamma = self.transform_controller()
            self.stats['transform_params'].append({
                'clip_low': clip_low.item(),
                'clip_high': clip_high.item(),
                'gamma': gamma.mean().item() if gamma.dim() > 0 else gamma.item()
            })

        return result

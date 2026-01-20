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

from typing import Optional, List, Literal
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ..base import AdaptationEngine, AdaptationConfig
from ...models.base import BaseModel, ModelProvider


@dataclass
class CascadedNormConfig(AdaptationConfig):
    """
    Configuration for CascadedNorm adaptation.

    Adaptation Methods:
    ------------------
    1. Gamma Clamp (default)
        - Gamma correction with histogram stretching
        - use_percentile=False: Direct pixel value clipping (default)
          * clip_low/high are interpreted as pixel values (0-255)
          * Example: clip_low=10, clip_high=245
        - use_percentile=True: Percentile-based clipping
          * clip_low/high are interpreted as percentile values (0-100)
          * Example: clip_low=2 means 2nd percentile, clip_high=98 means 98th percentile

    2. LUT (Look-Up Table)
        - LUT-based transformation
        - lut_monotonic=True: Enforce monotonic constraint
        - lut_ema=True: Use exponential moving average
        - lut_reg=True: Add smoothness regularization

    Cascade Modes:
    -------------
    1. 'all': Use all Normalization layers (default)
    2. 'single': Use only first Normalization layer (fastest)
    3. 'single_last': Use only last Normalization layer (fastest)
    4. 'selected': Use specified Normalization layers by indices

    Examples:
    --------
    # Basic gamma clamp (pixel value mode)
    config = CascadedNormConfig(
        adaptation_method='gamma_clamp',
        use_percentile=False,  # clip_low/high as pixel values
        cascade_mode='single'
    )

    # Gamma clamp with percentile
    config = CascadedNormConfig(
        adaptation_method='gamma_clamp',
        use_percentile=True,  # clip_low/high as percentile values
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
    adapt_lr: float = 1e-3

    # Gamma Clamp options
    use_percentile: bool = False  # False: pixel values, True: percentile values

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
    temperature: float = 0.01  # For differentiable percentile (only when use_percentile=True)

    def __post_init__(self):
        # Cascade mode validation
        if self.cascade_mode == "selected":
            if self.cascade_indices is None or len(self.cascade_indices) == 0:
                raise ValueError("cascade_indices must be provided when cascade_mode='selected'")

        # LUT validation
        if self.adaptation_method == "lut":
            if self.lut_size < 16 or self.lut_size > 1024:
                raise ValueError(f"lut_size must be in [16, 1024], got {self.lut_size}")


class DifferentiableHistogramStretcher(nn.Module):
    """
    Differentiable histogram stretching with channel-wise processing.

    Supports two modes:
    1. Percentile mode (use_percentile=True):
        - clip_low/high are percentile values (0-100)
        - Computes actual pixel values via soft_percentile per channel
    2. Clamp mode (use_percentile=False, default):
        - clip_low/high are direct pixel values (0-255)
        - Uses values directly without percentile computation

    Key: Each RGB channel is processed independently to preserve color balance.
    """

    def __init__(self, temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature

    def soft_percentile_batch(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Vectorized differentiable percentile for multiple channels.

        Args:
            x: Input channels (B, C, H, W)
            p: Target percentile (0-100, scalar)

        Returns:
            torch.Tensor: Percentile values per channel (B, C)
        """
        B, C, H, W = x.shape
        N = H * W

        # Reshape to (B, C, H*W) for per-channel sorting
        x_flat = x.reshape(B, C, -1)  # (B, C, N)

        # Compute percentile index
        idx = (p / 100.0) * (N - 1)
        indices = torch.arange(N, device=x.device, dtype=x.dtype)  # (N,)

        # Compute weights: (N,) broadcast to (B, C, N)
        weights = F.softmax(
            -(indices.unsqueeze(0).unsqueeze(0) - idx).abs() / (self.temperature * N), 
            dim=-1
        )  # (1, 1, N) -> (B, C, N) after broadcast

        # Sort per channel
        sorted_x, _ = torch.sort(x_flat, dim=-1)  # (B, C, N)

        # Weighted sum per channel
        return (weights * sorted_x).sum(dim=-1)  # (B, C)

    def forward(
        self, image: torch.Tensor,
        clip_low: torch.Tensor, clip_high: torch.Tensor, gamma: torch.Tensor,
        use_percentile: bool = False
    ) -> torch.Tensor:
        """
        Apply channel-wise stretching to image(s) with vectorized operations.

        Args:
            image: Input image (C, H, W) or (B, C, H, W)
            clip_low: Lower clipping bound
            clip_high: Upper clipping bound
            gamma: Gamma correction factor
            use_percentile: Whether bounds are percentiles or pixel values

        Returns:
            Stretched image with same shape as input
        """
        # Handle batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        B, C, H, W = image.shape

        # Compute channel-wise clipping values
        if use_percentile:
            low_vals = self.soft_percentile_batch(image, clip_low)    # (B, C)
            high_vals = self.soft_percentile_batch(image, clip_high)  # (B, C)
            # Reshape for broadcasting: (B, C, 1, 1)
            low_vals = low_vals.view(B, C, 1, 1)
            high_vals = high_vals.view(B, C, 1, 1)
        else:
            # Scalar values broadcast to all channels
            low_vals = clip_low
            high_vals = clip_high

        # Vectorized soft clipping (broadcasts across all dimensions)
        scale = 50.0
        clipped = low_vals + F.softplus((image - low_vals) * scale) / scale
        clipped = high_vals - F.softplus((high_vals - clipped) * scale) / scale

        # Vectorized normalization and gamma correction
        range_vals = high_vals - low_vals + 1e-6
        normalized = (clipped - low_vals) / range_vals
        gamma_corrected = torch.pow(normalized + 1e-6, gamma)

        stretched = torch.clamp(gamma_corrected * 255.0, 0, 255)

        if squeeze_output:
            stretched = stretched.squeeze(0)

        return stretched


class GammaTransform(nn.Module):
    """
    Learnable gamma parameters for CascadedNorm.

    Manages learnable parameters for histogram stretching:
    - clip_low: Lower clipping bound
    - clip_high: Upper clipping bound
    - gamma: Gamma correction factor

    Parameter ranges depend on use_percentile mode:
    - use_percentile=False (Clamp mode):
      * clip_low: [0, 127.5] (pixel values)
      * clip_high: [127.5, 255] (pixel values)
    - use_percentile=True (Percentile mode):
      * clip_low: [0, 10] (percentile values)
      * clip_high: [90, 100] (percentile values)
    - gamma: [0.5, 2.0] (both modes)
    """

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config

        # Learnable parameters (raw, will be constrained in forward)
        self.clip_low = nn.Parameter(torch.tensor(2.0))
        self.clip_high = nn.Parameter(torch.tensor(98.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))

        self.stretcher = DifferentiableHistogramStretcher(config.temperature)

    def forward(self):
        """
        Get constrained parameters based on use_percentile mode.

        Returns:
            tuple: (clip_low, clip_high, gamma)
        """
        if self.config.use_percentile:
            # Percentile mode: output percentile values (0-100)
            clip_low = torch.sigmoid(self.clip_low) * 10        # [0, 10] percentile
            clip_high = 90 + torch.sigmoid(self.clip_high) * 10 # [90, 100] percentile
        else:
            # Clamp mode: output pixel values (0-255)
            clip_low = torch.sigmoid(self.clip_low) * 127.5                    # [0, 127.5] pixel
            clip_high = 127.5 + torch.sigmoid(self.clip_high) * 127.5          # [127.5, 255] pixel

        # Gamma correction factor (same for both modes)
        gamma = 0.5 + torch.sigmoid(self.gamma) * 1.5  # [0.5, 2.0]

        return clip_low, clip_high, gamma


class LUTTransform(nn.Module):
    """
    Learnable look-up tables for CascadedNorm.

    Provides pixel-level transformation through learned LUT.
    Supports monotonic constraint, EMA, and smoothness regularization.
    """

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config

        # LUT table (fixed grid points) and values (learnable)
        self.register_buffer("lut_table", torch.linspace(0, 255, self.config.lut_size))
        self.lut_values = nn.Parameter(
            torch.linspace(0, 255, self.config.lut_size)
        )

        # EMA buffer
        if self.config.lut_ema:
            self.register_buffer("lut_ema_values", torch.linspace(0, 255, self.config.lut_size))

    def forward(self):
        """Get LUT values (with optional monotonic constraint and EMA)."""
        # Enforce monotonic constraint if enabled
        if self.config.lut_monotonic and self.training:
            self._enforce_monotonic()

        # Return EMA or current values
        if self.config.lut_ema and self.training:
            return self.lut_ema_values
        return self.lut_values

    def _enforce_monotonic(self):
        """Enforce monotonic constraint on LUT (in-place)."""
        with torch.no_grad():
            for i in range(1, len(self.lut_values)):
                if self.lut_values[i] < self.lut_values[i-1]:
                    self.lut_values[i].copy_(self.lut_values[i-1])

    def reg_loss(self) -> torch.Tensor | None:
        """Compute smoothness regularization loss."""
        if not self.config.lut_reg:
            return torch.tensor(0.0, device=self.lut_values.device)

        # Second-order smoothness: penalize large second derivatives
        diff2 = self.lut_values[2:] - 2 * self.lut_values[1:-1] + self.lut_values[:-2]
        return self.config.lut_reg_weight * (diff2 ** 2).mean()

    def update_ema(self):
        """Update EMA values (call after optimizer.step())."""
        if self.config.lut_ema and self.training:
            with torch.no_grad():
                self.lut_ema_values.copy_(
                    self.config.lut_ema_momentum * self.lut_ema_values
                    + (1 - self.config.lut_ema_momentum) * self.lut_values
                )


class CascadedNorm(nn.Module):
    """
    CascadedNorm: Cascaded Input Distribution Normalization
        by Statistical Norm Layer Alignment

    Learns to normalize input pixel distribution through adaptive
    transformation with cascaded layer-wise supervision.

    The term "normalization" refers to distribution alignment
    between source and target domains, achieved through:
    - Gamma Clamp: Histogram stretching with gamma correction
    - LUT: Learned look-up table transformation

    This differs from layer normalization methods (BatchNorm, LayerNorm)
    which perform statistical normalization of activations.

    Args:
        config: CascadedNormConfig with learning parameters
    """

    def __init__(self, config: CascadedNormConfig):
        super().__init__()

        self.config = config
        self.adaptation_method = config.adaptation_method
        self.current_params = (torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0))

        # Initialize transform controller based on method
        if config.adaptation_method == "lut":
            self.transform_controller = LUTTransform(config)
            self.forward = self.forward_lut
        else:
            self.transform_controller = GammaTransform(config)
            self.forward = self.forward_gamma_clamp
            self.reg_loss = lambda: None  # No regularization for gamma clamp
            self.update_ema = lambda: None  # No EMA for gamma clamp

        # Norm layer tracking (will be populated by Engine)
        self.norm_layers: nn.ModuleList[nn.Module] = nn.ModuleList()
        self.source_means: List[torch.Tensor] = []
        self.source_vars: List[torch.Tensor] = []

    def forward_gamma_clamp(self, img: torch.Tensor) -> torch.Tensor:
        self.current_params = self.transform_controller()
        return self.transform_controller.stretcher(
            img, *self.current_params, use_percentile=self.config.use_percentile  # Pass mode flag
        )

    def forward_lut(self, img: torch.Tensor) -> torch.Tensor:
        lut_values = self.transform_controller()
        # Apply LUT transformation
        img_normalized = (img / 255.0 * (len(lut_values) - 1))
        img_int = img_normalized.long().clamp(0, len(lut_values) - 1)
        return lut_values[img_int]

    def online_parameters(self):
        """Get learnable parameters for optimization."""
        return self.transform_controller.parameters()

    def reg_loss(self) -> torch.Tensor | None:
        """Compute regularization loss (only for LUT with lut_reg=True)."""
        return self.transform_controller.reg_loss()

    def update_ema(self):
        """Update EMA (only for LUT with lut_ema=True)."""
        self.transform_controller.update_ema()

    def diff(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute BN statistics alignment (optimized batched computation).

        Aligns batch statistics with source statistics across all monitored
        normalization layers.
        """
        current_means, current_vars, source_means, source_vars = [], [], [], []

        # Query target statistics
        for layer, src_mean, src_var in zip(self.norm_layers, self.source_means, self.source_vars):
            mean = layer.current_mean
            var = layer.current_var

            # Reduce to scalar for consistent stacking
            if mean.numel() > 1:
                mean = mean.mean()
            if var.numel() > 1:
                var = var.mean()
            if src_mean.numel() > 1:
                src_mean = src_mean.mean()
            if src_var.numel() > 1:
                src_var = src_var.mean()

            current_means.append(mean)
            current_vars.append(var)
            source_means.append(src_mean)
            source_vars.append(src_var)

        # Batched computation (single graph node for efficiency)
        if len(current_means) > 0:
            current_means = torch.stack(current_means)
            current_vars = torch.stack(current_vars)
            source_means = torch.stack([
                m.mean() if m.numel() > 1 else m
                for m in self.source_means
            ]).to(current_means.device)
            source_vars = torch.stack([
                v.mean() if v.numel() > 1 else v
                for v in self.source_vars
            ]).to(current_vars.device)

            target_stat = torch.cat([current_means, current_vars], dim=0)
            source_stat = torch.cat([source_means, source_vars], dim=0)
        else:
            # No norm layers to align
            device = next(self.parameters()).device
            target_stat = torch.zeros(0, device=device)
            source_stat = torch.zeros(0, device=device)

        return target_stat, source_stat


class CascadedNormEngine(AdaptationEngine):
    """
    Cascaded Norm Adaptation Engine
    Injects CascadedNorm module into the base model.

    Supports:
    - nn.BatchNorm2d / nn.BatchNorm1d / FrozenBatchNorm2d
    - nn.LayerNorm

    The engine extracts normalization layers from the base model,
    overrides foward to capture batch statistics, and optimizes
    input transformation to align batch stats with source stats.

    Args:
        base_model: Pre-trained model
        config: CascadedNormConfig with learning parameters

    Example:
        >>> config = CascadedNormConfig(
        ...     adaptation_method='gamma_clamp',
        ...     use_percentile=True,
        ...     cascade_mode='single'
        ... )
        >>> adaptive_model = CascadedNormEngine(base_model, config)
        >>> output = adaptive_model(batch)
    """
    model_name: str = "CascadedNormEngine"
    loss_class = nn.HuberLoss

    def __init__(self, base_model: BaseModel, config: CascadedNormConfig):
        self.dist_norm: CascadedNorm  # will be initialized in _pre_init()
        self.dist_norm_state: dict
        self.config = config

        super().__init__(base_model, config)

    def _pre_init(self):
        # Transformation modules
        self.dist_norm = CascadedNorm(self.config)

    def _post_init(self):
        self.dist_norm.to(self.device)
        self.dist_norm_state = {key: value.cpu() for key, value in self.dist_norm.state_dict().items()}

        # Extract norm layers from base model
        self._extract_norm_layers()

        if self.config.verbose:
            print(f"\n[CascadedNorm] Summary:")
            print(f"  Adaptation method: {self.config.adaptation_method}")
            print(f"  Use percentile: {self.config.use_percentile}")
            print(f"  Cascade mode: {self.config.cascade_mode}")
            print(f"  Total layers: {len(self.dist_norm.norm_layers)}")
            for cls in self.dist_norm.norm_layers:
                print(f"      {cls.__class__.__name__}")

    def _extract_norm_layers(self):
        """
        Find all normalization layers including FrozenBatchNorm.

        Extracts source statistics (running_mean/var) for alignment.
        """
        # Find Norms
        found = []
        for name, module in self.base_model.named_modules():
            module_type = type(module).__name__

            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) or "BatchNorm" in module_type:
                found.append((
                    name, "BN", module, module.running_mean.clone(), module.running_var.clone()
                ))
            elif isinstance(module, nn.LayerNorm):
                found.append((
                    name, "LN", module, torch.tensor(0.0), torch.tensor(1.0)
                ))

        # Filtering & Wrapping
        filtered = self._filter_by_cascade_mode(found)
        self._cascade_wrap(filtered)

        # Populate to CascadedNorm
        for _, _, module, running_mean, running_var in filtered:
            self.dist_norm.norm_layers.append(module)
            self.dist_norm.source_means.append(running_mean)
            self.dist_norm.source_vars.append(running_var)

    def _filter_by_cascade_mode(self, norm_list: list[nn.Module]) -> list[nn.Module]:
        """Filter normalization layers based on cascade mode."""
        match self.config.cascade_mode:
            case "single":
                return [norm_list[0]]
            case "single_last":
                return [norm_list[-1]]
            case "selected":
                return [norm_list[i] for i in self.config.cascade_indices]
            case _:  # all
                return norm_list

    @staticmethod
    def _cascade_wrap(filtered: list[nn.Module]):
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
        """Get learnable parameters for optimization."""
        return self.dist_norm.online_parameters()

    def _reset_stats(self):
        """Initialize statistics tracking."""
        self._stats = {
            'losses': [],
            'reg_losses': [],
            'alignment_losses': [],
            'transform_params': [],
            'config': {
                'adaptation_method': self.config.adaptation_method,
                'use_percentile': self.config.use_percentile,
                'cascade_mode': self.config.cascade_mode,
                'num_layers': len(self.dist_norm.norm_layers),
            }
        }

    def reset(self, reset_stats=False):
        """Reset model to initial state."""
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
        # Zero gradients
        self.optimizer.zero_grad()

        # Forward
        if self.adapting:
            match self.model_provider:
                case ModelProvider.Detectron2:
                    inputs = args[0]  # Detectron2 inputs are List[Dict] with "image" key
                    for x in inputs:
                        x['image'] = self.dist_norm(x['image'].to(self.device))
                case ModelProvider.HuggingFace:
                    pass  # TODO
                case ModelProvider.Ultralytics:
                    pass  # TODO
                case _:
                    raise ValueError(f"Model provider {self.model_provider} is not supported.")
        result = self.base_model(*args, **kwargs)

        # Compute alignment loss
        loss = self.loss_function(*self.dist_norm.diff())
        self._stats['alignment_losses'].append(loss.item())

        # Add regularization loss if applicable
        reg_loss = self.dist_norm.reg_loss()
        if reg_loss is not None:
            self._stats['reg_losses'].append(reg_loss.item())
            loss += reg_loss
        else:
            self._stats['reg_losses'].append(0.0)

        # Backward and update
        loss.backward()
        self.optimizer.step()
        self._stats['losses'].append(loss.item())

        # Update EMA if using LUT
        self.dist_norm.update_ema()

        # Log transform parameters
        if self.config.adaptation_method == "gamma_clamp":
            clip_low, clip_high, gamma = self.dist_norm.current_params
            self._stats['transform_params'].append({
                'clip_low': clip_low.item(),
                'clip_high': clip_high.item(),
                'gamma': gamma.item() if gamma.dim() == 0 else gamma.mean().item()
            })

        return result

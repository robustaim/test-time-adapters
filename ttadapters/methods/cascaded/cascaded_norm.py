""" CascadedNorm: Input Transformation for Norm Statistics Alignment """

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


class DifferentiableHistogramStretcher(nn.Module):
    """
    Vectorized differentiable histogram stretching for high-precision statistics.
    Optimized for resolutions like 800x1280 (~1M pixels).

    Supports two modes:
    1. Percentile mode (use_percentile=True):
        - clip_low/high are percentile values (0-100)
        - Computes actual pixel values via soft_percentile
    2. Clamp mode (use_percentile=False, default):
        - clip_low/high are direct pixel values (0-255)
        - Uses values directly without percentile computation
    """

    def __init__(self, temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature

    def soft_percentile(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Vectorized differentiable percentile approximation using all pixels.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).
            p (torch.Tensor): Target percentile value (0-100, scalar).

        Returns:
            torch.Tensor: Calculated percentiles of shape (B, C).
        """
        B, C, H, W = x.shape
        # Flatten spatial dimensions to (B, C, H*W)
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

    def forward(
        self, image: torch.Tensor,
        clip_low: torch.Tensor, clip_high: torch.Tensor, gamma: torch.Tensor,
        use_percentile: bool = False
    ) -> torch.Tensor:
        """
        Apply vectorized stretching and gamma correction to the entire batch.

        Args:
            image: Input image (B, C, H, W) or (C, H, W)
            clip_low: Lower clipping value
                - If use_percentile=True: percentile value (0-100), e.g., 2.0 for 2nd percentile
                - If use_percentile=False: pixel value (0-255), e.g., 10.0
            clip_high: Upper clipping value
                - If use_percentile=True: percentile value (0-100), e.g., 98.0 for 98th percentile
                - If use_percentile=False: pixel value (0-255), e.g., 245.0
            gamma: Gamma correction value
            use_percentile: If True, interpret clip_low/high as percentile values.
                            If False, interpret them as direct pixel values.

        Returns:
            Transformed image in range [0, 255]
        """
        # Ensure 4D shape (B, C, H, W) even for single image input
        if image.dim() == 3:
            image = image.unsqueeze(0)

        B, C, H, W = image.shape

        # 1. Compute low/high values based on mode
        if use_percentile:
            # Percentile mode: compute actual pixel values from percentiles
            low_val = self.soft_percentile(image, clip_low)   # (B, C)
            high_val = self.soft_percentile(image, clip_high) # (B, C)
            # Reshape for broadcasting: (B, C) -> (B, C, 1, 1)
            low_val_bc = low_val.view(B, C, 1, 1)
            high_val_bc = high_val.view(B, C, 1, 1)
        else:
            # Clamp mode: use clip_low/high directly as pixel values
            # Broadcast scalar values to (B, C, 1, 1)
            low_val_bc = clip_low.view(1, 1, 1, 1).expand(B, C, 1, 1)
            high_val_bc = clip_high.view(1, 1, 1, 1).expand(B, C, 1, 1)

        # 2. Differentiable Clipping using Softplus to keep gradients alive
        scale = 50.0
        clipped = low_val_bc + F.softplus((image - low_val_bc) * scale) / scale
        clipped = high_val_bc - F.softplus((high_val_bc - clipped) * scale) / scale

        # 3. Normalization to [0, 1] and Gamma Correction
        range_val = high_val_bc - low_val_bc + 1e-6
        normalized = (clipped - low_val_bc) / range_val
        gamma_corrected = torch.pow(normalized + 1e-6, gamma)

        # 4. Scale back to [0, 255]
        return torch.clamp(gamma_corrected * 255.0, 0, 255)


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
        current_means, current_vars = [], []
        source_means, source_vars = self.source_means, self.source_vars

        # Query target statistics
        for layer in self.norm_layers:
            current_means.append(layer.current_mean)
            current_vars.append(layer.current_var)

        # Batched computation (single graph node for efficiency)
        if len(current_means) > 0:
            current_means = torch.stack(current_means)
            current_vars = torch.stack(current_vars)
            source_means = torch.stack(source_means).to(current_means.device)
            source_vars = torch.stack(source_vars).to(current_vars.device)

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
        self.dist_norm.to(self.dtype)
        self.dist_norm_state = {key: value.cpu() for key, value in self.dist_norm.state_dict().items()}

        # Inject dist_norm into base model
        self._inject_dist_norm()

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
                if isinstance(x, torch.Tensor) and x.ndim == 4:
                    x = self.dist_norm(x)
                return original_forward(x, *args, **kwargs)
            self.base_model.forward = preprocessing_forward
            return

        # Wrap the first module's forward
        original_forward = first_module.forward

        def preprocessing_forward(x, *args, **kwargs):
            # Apply dist_norm to image input
            x = self.dist_norm(x)
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

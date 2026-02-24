from typing import Literal, Tuple, Dict
from enum import Enum
import re

import numpy as np
import cv2

import torch
from torch import nn
import torch.nn.functional as F

from ...base import AdaptationEngine
from ....models.base import BaseModel, ModelProvider, DataPreparation

from .configuration_flow_adaptation import FlowAdaptationConfig


class GaussianKLDivLoss(nn.Module):
    """KL Divergence Loss between two Gaussian distributions.

    Computes KL(P||Q) where P and Q are Gaussian distributions.
    KL(P||Q) = log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²)/(2σ_q²) - 1/2
    """

    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean"):
        """
        Args:
            reduction: "none" | "mean" | "sum"
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between input and target distributions.

        Args:
            input: Input tensor (mean, var) of Q distribution
                   Shape: (batch_size, 2) or (2,) where [..., 0] is mean, [..., 1] is var
            target: Target tensor (mean, var) of P distribution
                    Shape: (batch_size, 2) or (2,) where [..., 0] is mean, [..., 1] is var

        Returns:
            KL divergence loss
        """
        # Extract means and variances
        if input.dim() == 1:
            q_mean, q_var = input[0], input[1]
            p_mean, p_var = target[0], target[1]
        else:
            q_mean, q_var = input[:, 0], input[:, 1]
            p_mean, p_var = target[:, 0], target[:, 1]

        # Add epsilon for numerical stability
        q_var = q_var + 1e-6
        p_var = p_var + 1e-6

        sigma_p = torch.sqrt(p_var)
        sigma_q = torch.sqrt(q_var)

        # Compute KL divergence: KL(P||Q)
        term1 = torch.log(sigma_q / sigma_p)
        term2 = (p_var + (p_mean - q_mean)**2) / (2 * q_var)

        kl_div = term1 + term2 - 0.5

        # Apply reduction
        if self.reduction == "none":
            return kl_div
        elif self.reduction == "mean":
            return kl_div.mean()
        elif self.reduction == "sum":
            return kl_div.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class CLAHETransform(nn.Module):
    """CLAHE Transform Module. (Not differentiable)"""

    def __init__(self, config: FlowAdaptationConfig):
        super().__init__()
        self.config = config
        self.clip_limit = config.clahe_clip_limit
        self.tile_size = config.clahe_tile_size

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        _, H, W = image.shape  # image shape: (C=3, H, W)

        # Adaptive Grid Size Calculation
        # maintain standard tile size based on the longest edge to keep tile aspect ratio ~1:1
        if W >= H:
             grid_x = self.tile_size
             grid_y = max(1, int(self.tile_size * (H / W)))
        else:
             grid_y = self.tile_size
             grid_x = max(1, int(self.tile_size * (W / H)))

        # Create dynamic CLAHE object
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(grid_x, grid_y))

        image_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, C), RGB
        image_np = image_np.astype(np.uint8)

        image_ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
        image_ycrcb[:, :, 0] = clahe.apply(image_ycrcb[:, :, 0])

        image_rgb = cv2.cvtColor(image_ycrcb, cv2.COLOR_YCrCb2RGB)
        image_rgb = image_rgb.astype(np.float32)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)  # (C, H, W)

        return image_tensor.to(image.device, dtype=image.dtype), {'clip_limit': self.clip_limit, 'tile_size': self.tile_size}


class GammaTransform(nn.Module):
    """Differentiable Gamma Transformation Module by histogram stretching."""

    def __init__(self, config: FlowAdaptationConfig):
        super().__init__()
        self.config = config
        self.temperature = config.gamma_temperature

        self.saturation_limit = torch.tensor(config.gamma_saturation_limit, requires_grad=False)
        self.noise_floor = torch.tensor(config.gamma_noise_floor, requires_grad=False)

        self.gamma = nn.Parameter(torch.tensor(0.0))  # init for identity transform
        self.gamma_range = tuple(config.gamma_range)

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

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Apply stretching to image with gamma correction."""
        C = image.shape[0]  # batch size will be 1 cause this is online learning
        stretched = torch.zeros_like(image, device=image.device)

        # self.gamma init 0.0 -> gamma = 1.0 (Identity)
        gamma = (self.gamma + 1.0).clamp(*self.gamma_range)

        for c in range(C):
            stretched[c] = self.stretch_channel(
                image[c], self.noise_floor, self.saturation_limit, gamma
            )

        return stretched, {'noise_floor': self.noise_floor.item(), 'saturation_limit': self.saturation_limit.item(), 'gamma': gamma.item()}

    def get_regularization_loss(self):
        """Compute regularization loss relative to initialization anchors."""
        # Pull towards Identity (gamma=1.0 -> self.gamma=0.0)
        return self.gamma.pow(2)


class CLAHEGammaTransform(nn.ModuleList):
    def __init__(self, config: FlowAdaptationConfig):
        super().__init__([CLAHETransform(config), GammaTransform(config)])
        self.gate_raw = nn.Parameter(torch.tensor(0.0))

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        clahe_gate = 0.5 * (torch.tanh(self.gate_raw) + 1.0)

        transformed, clahe_params = self.transform[0](image)
        transformed = clahe_gate * transformed + (1 - clahe_gate) * image
        transformed, gamma_params = self.transform[1](transformed)

        return transformed, clahe_params + gamma_params

    def get_regularization_loss(self):
        return self.gamma.get_regularization_loss()


class CLAHEGammaResidualTransform(CLAHEGammaTransform):
    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        residual_gate = 0.5 * (torch.tanh(self.gate_raw) + 1.0)

        transformed, clahe_params = self.transform[0](image)
        residual_base, residual_params = self.transform[1](transformed)
        transformed = transformed + residual_gate * (residual_base - transformed)

        return transformed, clahe_params + residual_params


class InputTransformation(nn.Module):
    """
    Input Transformation Module which creates different view of input image.
    """

    def __init__(self, config: FlowAdaptationConfig):
        super().__init__()
        self.config = config
        self.applied_params: dict = {}
        self.itm_type = config.itm_type

        if config.itm_type == "clahe":
            self.transform = CLAHETransform(config)
        elif config.itm_type == "gamma":
            self.transform = GammaTransform(config)
        elif config.itm_type == "clahe-gamma":
            self.transform = CLAHEGammaTransform(config)
        elif config.itm_type == "clahe-gamma-residual":
            self.transform = CLAHEGammaResidualTransform(config)
        else:
            raise ValueError(f"Unsupported itm_type: {config.itm_type}")

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        """Weighted blending between original image and transformed image."""
        transformed, transform_params = self.transform(original)
        self.applied_params = transform_params
        return transformed

    def get_regularization_loss(self) -> torch.Tensor | None:
        return self.transform.get_regularization_loss()


class CascadedNorm(nn.Module):
    """
    CascadedNorm: Cascaded Input Distribution Normalization via Flow Adaptation.

    The term "normalization" refers to distribution matching
    between source and target domains, achieved through:
    - Gamma Transform: Histogram stretching with gamma correction
    - CLAHE Transform: Histogram equalization with adaptive contrast enhancement

    Instead of adapting individual layers, it learns a single global input transformation
    that satisfies the statistical constraints of all inserted anchors simultaneously.
    (Learns to normalize input pixel distribution through adaptive transformation with cascaded layer-wise supervision.)

    Args:
        config: FlowAdaptationConfig with learning parameters
    """

    def __init__(self, config: FlowAdaptationConfig):
        super().__init__()
        self.config = config

        self.itm = InputTransformation(config)  # Input Transformation Module
        self.anchors = nn.ModuleList()

    def online_parameters(self):
        """Get learnable parameters for optimization."""
        return self.itm.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.itm(x)

    def diff(self) -> torch.Tensor:
        align_loss = torch.stack([anchor.diff() for anchor in self.anchors]).sum()
        reg_loss = self.itm.get_regularization_loss()

        # Combined Loss: Alignment + Regularization
        return (align_loss + reg_loss) if reg_loss is not None else align_loss


class CascadeAnchor(nn.Module):
    """
    Cascade Anchor Module: A checkpoint for Feature Flow Adaptation.

    Supported normalization layers:
    - nn.BatchNorm2d / FrozenBatchNorm2d
    - nn.LayerNorm
    """

    def __init__(self, config: FlowAdaptationConfig, wrapped_module: nn.Module, loss_fn: nn.Module):
        super().__init__()
        self.config = config
        self.wrapped = wrapped_module
        self.loss_fn = loss_fn
        self.reduce_dim: tuple[int, ...] | None = self.config.reduce_dim
        self.device = next(wrapped_module.parameters()).device

        self.forward_stats = dict(mean=[], var=[])  # Will be updated per one forward pass
        self.register_buffer("source_means", torch.tensor([0.0], device=self.device))
        self.register_buffer("source_vars", torch.tensor([1.0], device=self.device))
        self.source_stats = dict(
            mean=self.source_means, var=self.source_vars, running_means=[], running_vars=[], num_samples=0
        )

    def __repr__(self) -> str:
        module_name = self.__class__.__name__
        return f"{module_name}(wrapped={self.wrapped.__class__.__name__}, reduce_dim={self.reduce_dim}, loss_fn={self.loss_fn.__class__.__name__})"

    def train(self, mode: bool = True):
        super().train(mode)
        if not mode:
            if self.source_stats['num_samples'] > 0:  # finalize source stats
                with torch.no_grad():
                    mean = torch.stack(self.source_stats['running_means']).mean(dim=0)
                    var = torch.stack(self.source_stats['running_vars']).mean(dim=0)
                    self.source_means = mean.detach().to(self.device)
                    self.source_vars = var.detach().to(self.device)

            self.source_stats['running_means'] = []
            self.source_stats['running_vars'] = []
            self.source_stats['num_samples'] = 0

    def forward(self, x, *args, **kwargs):
        out = self.wrapped(x, *args, **kwargs)

        self.forward_stats['mean'] = out.mean(dim=self.reduce_dim)
        self.forward_stats['var'] = out.var(dim=self.reduce_dim, unbiased=False)

        if self.training:
            with torch.no_grad():
                self.source_stats['running_means'].append(self.forward_stats['mean'].detach())
                self.source_stats['running_vars'].append(self.forward_stats['var'].detach())
                self.source_stats['num_samples'] += x.shape[0]

        return out

    def diff(self) -> torch.Tensor:
        """Compute Flow Adaptation loss."""
        try:
            if isinstance(self.loss_fn, GaussianKLDivLoss):
                # KL Divergence requires (mean, var) pairs
                # Stack to form (..., 2) for GaussianKLDivLoss
                # If scalars: (2,)
                # If (C,): (C, 2)
                input_dist = torch.stack([self.forward_stats['mean'], self.forward_stats['var']], dim=-1)
                target_dist = torch.stack([self.source_means, self.source_vars], dim=-1)

                loss = self.loss_fn(input_dist, target_dist)
            else:
                mean_loss = self.loss_fn(self.forward_stats['mean'], self.source_means)
                var_loss = self.loss_fn(self.forward_stats['var'], self.source_vars)
                loss = mean_loss + var_loss

            self.forward_stats['mean'] = []  # reset for next forward pass
            self.forward_stats['var'] = []  # reset for next forward pass
        except ValueError as e:
            raise RuntimeError(f"Forward statistics not yet collected for {self.wrapped}.") from e
        return loss


class FlowAdaptationEngine(AdaptationEngine):
    """
    Flow Adaptation Engine
    Injects CascadedNorm module into the base model.

    Supports:
    - nn.BatchNorm2d / FrozenBatchNorm2d
    - nn.LayerNorm

    The engine extracts normalization layers from the base model,
    overrides foward to adapt source flow.

    Args:
        base_model: Pre-trained model
        config: FlowAdaptationConfig with learning parameters

    Example:
        >>> config = FlowAdaptationConfig(
        ...     itm_type="gamma",
        ...     cascade_target=["layer4"]
        ... )
        >>> adaptive_model = FlowAdaptationEngine(base_model, config)
        >>> output = adaptive_model(batch)
    """
    model_name: str = "FlowAdaptationEngine"
    loss_class = nn.MSELoss

    def __init__(self, base_model: BaseModel, config: FlowAdaptationConfig):
        self.dist_norm: CascadedNorm  # will be initialized in _pre_init()
        self.dist_norm_state: dict
        self.config = config

        if self.config.use_kl_divergence:
            self.loss_class = GaussianKLDivLoss

        super().__init__(base_model, config)

    def _pre_init(self):
        # Transformation modules
        self.dist_norm = CascadedNorm(self.config)

    def _post_init(self):
        self.dist_norm.to(self.device)
        self.dist_norm_state = {key: value.cpu() for key, value in self.dist_norm.state_dict().items()}

        # Extract norm layers from base model
        if self.config.verbose:
            print("=" * 60)
            print(f"[{self.model_name}] Initialization Summary")
            print("-" * 60)
            print(f"  View Transform Method : {self.config.itm_type}")
            print(f"  Loss Function         : {self.loss_class.__name__}")
        self._extract_target_layers()
        if self.config.verbose:
            print("-" * 60)
            print("  Flow Adaptation Module:")
            print(self.dist_norm)
            print("=" * 60)

        # Stats
        self._reset_stats()

        # Device
        self.to(self.device, dtype=self.dtype)

    def _extract_target_layers(self):
        # Recursively extract layers and replace with CascadeAnchors
        # Compile pattern: if None, match all; otherwise match provided patterns
        cascade_target_re = None if self.config.cascade_target is None else re.compile(
            "|".join(f"({p})" for p in self.config.cascade_target), flags=re.IGNORECASE
        )
        # Compile exclusion pattern
        exclude_target_re = None
        if self.config.exclude_target:
            exclude_target_re = re.compile("|".join(f"({p})" for p in self.config.exclude_target), flags=re.IGNORECASE)

        applied_list = self.dist_norm.anchors
        applied_key_list = []
        for name, module in self.base_model.named_modules():
            # 1. Safety Check: Skip if matches exclude_target
            if exclude_target_re and exclude_target_re.search(name):
                continue

            # 2. Target Check:
            # If cascade_target is None -> Target All (Default)
            # If cascade_target is Set -> Target Only Matches
            if cascade_target_re is None or cascade_target_re.search(name):
                # Check if already wrapped (to prevent double wrapping if called multiple times)
                if isinstance(module, CascadeAnchor):
                    continue
                anchor = CascadeAnchor(self.config, module, self.loss_function)
                applied_list.append(anchor)
                applied_key_list.append(name)
                self._replace_module(name, anchor)  # Replace the original norm layer with the anchor

        if self.config.verbose:
            target_str = str(self.config.cascade_target) if self.config.cascade_target else "All (Default)"
            print(f"  Target Strategy       : {target_str}")
            print(f"  Applied To            : {len(self.dist_norm.anchors)} layer(s)")

            if len(applied_key_list) > 0:
                print("-" * 60)
                print("  Applied Layers (Sample):")
                # Print sample if too many
                if len(applied_key_list) > 10:
                    for k in applied_key_list[:5]:
                        print(f"    - {k}")
                    print(f"    ... ({len(applied_key_list) - 10} more) ...")
                    for k in applied_key_list[-5:]:
                        print(f"    - {k}")
                else:
                    for k in applied_key_list:
                        print(f"    - {k}")

    def _replace_module(self, module_name: str, new_module: nn.Module):
        """Replace a module in the base model with a new module."""
        name_parts = module_name.split('.')
        parent = self.base_model
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, name_parts[-1], new_module)

    def online_parameters(self):
        """Get learnable parameters for optimization."""
        return self.dist_norm.online_parameters()

    def _reset_stats(self):
        """Initialize statistics tracking."""
        self._stats = {
            'losses': [],
            'transform_params': [],
            'config': vars(self.config)
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

    def fit(self, source_dataset: DataPreparation, batch_size=1, max_samples=2000, dtype=torch.float32, **kwargs):
        from tqdm.auto import tqdm
        from torch.utils.data import DataLoader
        from ....utils.validator import DetectionEvaluator

        print(f"[{self.model_name}] Starting Calibration ({max_samples} samples)...")

        self.offline()  # turn off online mode so that the anchors don't update their parameters
        self.eval()  # turn on eval mode so that the base model doesn't update its parameters
        for anchor in self.dist_norm.anchors:
            anchor.train()
            anchor.wrapped.eval()

        loader = DataLoader(source_dataset, batch_size=batch_size, collate_fn=source_dataset.collate_fn, **kwargs)

        def limit_samples():
            for batch in loader:
                yield batch
                count = self.dist_norm.anchors[0].source_stats['num_samples']
                if count >= max_samples:
                    break

        DetectionEvaluator.evaluate_with_reset(
            self.base_model, desc="Feature Calibration", classes=source_dataset.classes,
            loader=limit_samples(), loader_length=max_samples//batch_size,
            data_preparation=source_dataset, reset=False,
            dtype=dtype, device=self.device
        )
        self.eval()  # finalize the stats

        # save the state dict of the anchors
        self.dist_norm_state = {key: value.cpu() for key, value in self.dist_norm.state_dict().items()}
        print(f"[{self.model_name}] Calibration Complete. {len(self.dist_norm.anchors)} anchors.")

    def _get_letterbox_range(self, img):
        # Create spatial mask shaped (H, W) or (1, H, W)
        diff = (img - self.config.mask_value).abs()  # Check difference from 114 (letterbox value)
        # Reduce channel dim: (C, H, W) -> (H, W) if all channels match padding
        spatial_mask = (diff < 1e-2).all(dim=0)

        # Find valid rows (H)
        # spatial_mask is True for padding, False for valid
        # So rows where ALL pixels are padding are padding rows.
        is_padding_row = spatial_mask.all(dim=1) # (H,)
        valid_rows = torch.where(~is_padding_row)[0]

        if len(valid_rows) > 0:
            y_min = valid_rows[0].item()
            y_max = valid_rows[-1].item() + 1
            return y_min, y_max
        else:
            return None

    def forward(self, *args, **kwargs):
        # Zero gradients
        self.optimizer.zero_grad()

        # Dist norm forward
        if self.adapting:
            match self.model_provider:
                case ModelProvider.Detectron2:
                    inputs = args[0]  # Detectron2 inputs are List[Dict] with "image" key
                    for x in inputs:
                        img = x['image'].to(self.device)

                        # recover original scale
                        original_scale = img.max() <= 1.0
                        if original_scale:
                            img = img * 255.0

                        # apply dist norm
                        img_transformed = self.dist_norm(img)
                        x['image'] = img_transformed / 255.0 if original_scale else img_transformed
                case ModelProvider.HuggingFace | ModelProvider.Ultralytics:
                    if kwargs:  # HuggingFace (RT-DETR) uses "pixel_values"
                        input_dict = kwargs

                        # Determine which key to use
                        if 'pixel_values' in input_dict:
                            img_key = 'pixel_values'
                        elif 'img' in input_dict:
                            img_key = 'img'
                        else:
                            raise ValueError(f"No image key found in input_dict: {input_dict}")

                        img = input_dict[img_key].to(self.device)
                    else:  # Ultralytics models basically use args[0] as input tensor
                        img = args[0]  
                        if isinstance(img, dict) and "img" in img:
                            img = img['img'].to(self.device)

                    # recover original scale
                    original_scale = img.max() <= 1.0
                    if original_scale:
                        img = img * 255.0

                    # apply dist norm for each image in batch
                    transformed_list = []
                    for i in range(img.shape[0]):
                        img_current = img[i]

                        letterbox_range = None
                        if self.config.masked_processing:
                            letterbox_range = self._get_letterbox_range(img_current)

                        if letterbox_range:  # Apply Masked Processing if enabled (Letterbox removal)
                            img_transformed = img_current.clone()
                            y_min, y_max = letterbox_range
                            img_transformed[:, y_min:y_max, :] = self.dist_norm(img_current[:, y_min:y_max, :])
                        else:  # Standard processing
                            img_transformed = self.dist_norm(img_current)

                        transformed_list.append(img_transformed)

                    img_batch = torch.stack(transformed_list, dim=0)
                    model_input = img_batch / 255.0 if original_scale else img_batch

                    # Update args
                    if kwargs:
                        input_dict[img_key] = model_input
                    else:
                        args = (model_input,) + args[1:]
                case _:
                    raise ValueError(f"Model provider {self.model_provider} is not supported yet.")

        # Base model forward
        result = self.base_model(*args, **kwargs)
        if not self.adapting:
            return result

        # Backward
        loss = self.dist_norm.diff()
        loss.backward()
        self.optimizer.step()
        self._stats['losses'].append(loss.item())

        # Logging
        self._stats['transform_params'].append(self.dist_norm.itm.applied_params)

        return result

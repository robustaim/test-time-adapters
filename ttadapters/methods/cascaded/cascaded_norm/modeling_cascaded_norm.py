from typing import Literal, Tuple
from enum import Enum
import re

import numpy as np
import cv2

import torch
from torch import nn
import torch.nn.functional as F

from ...base import AdaptationEngine
from ....models.base import BaseModel, ModelProvider
from .configuration_cascaded_norm import CascadedNormConfig


class CLAHETransform(nn.Module):
    """CLAHE Transform Module. (Not differentiable)"""

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config
        self.clip_limit = config.clahe_clip_limit
        self.tile_size = config.clahe_tile_size

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[float, int]]:
        image_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, C), RGB
        image_np = image_np.astype(np.uint8)

        image_ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_size, self.tile_size))
        image_ycrcb[:, :, 0] = clahe.apply(image_ycrcb[:, :, 0])

        image_rgb = cv2.cvtColor(image_ycrcb, cv2.COLOR_YCrCb2RGB)
        image_rgb = image_rgb.astype(np.float32)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1)  # (C, H, W)

        return image_tensor, (self.clip_limit, self.tile_size)


class GammaTransform(nn.Module):
    """Differentiable Gamma Transformation Module by histogram stretching."""

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config
        self.temperature = config.gamma_temperature

        self.saturation_limit = torch.tensor(config.gamma_saturation_limit, requires_grad=False)
        self.noise_floor = torch.tensor(config.gamma_noise_floor, requires_grad=False)

        # sigmoid(x) = (1.0 - 0.5) / 1.5 = 1/3
        # x = logit(1/3) = log( (1/3) / (2/3) ) = log(0.5)
        self.gamma = nn.Parameter(torch.tensor(-0.6931))  # init for identity transform

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

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[float, float, float]]:
        """Apply stretching to image with gamma correction."""
        C = image.shape[0]  # batch size will be 1 cause this is online learning
        stretched = torch.zeros_like(image, device=image.device)
        gamma = F.softplus(self.gamma) + 0.1  # range [0.1, inf)
        gamma = torch.clamp(gamma, 0.1, 5.0)  # constrain to reasonable range [0.1, 5.0]

        for c in range(C):
            stretched[c] = self.stretch_channel(
                image[c], self.noise_floor, self.saturation_limit, gamma
            )

        return stretched, (self.noise_floor.item(), self.saturation_limit.item(), gamma.item())


class InputTransformation(nn.Module):
    """
    Input Transformation Module which creates different view of input image.
    Transformed image and original image will be blended by learnable gate.
    """

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config
        self.applied_params = None
        self.itm_type = config.itm_type

        self.gate_raw = nn.Parameter(torch.tensor(0.0))
        self.transform = CLAHETransform(config) if config.itm_type == "clahe" else GammaTransform(config)

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        """Weighted blending between original image and transformed image."""
        transformed, transform_params = self.transform(original)
        gate = 0.5 * (torch.tanh(self.gate_raw) + 1.0)
        self.applied_params = gate.item(), *transform_params
        return gate * transformed.to(original.device, dtype=original.dtype) + (1 - gate) * original


class SupportedNormType(Enum):
    BN = "BN"
    LN = "LN"


class AnchorList(nn.ModuleList):
    """Wrapper for nn.ModuleList to customize __repr__."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bn_count = 0
        self.ln_count = 0

    def append(self, item: "CascadeAnchor"):
        if not isinstance(item, CascadeAnchor):
            raise TypeError("item must be CascadeAnchor")

        super().append(item)
        if item.anchor_type == SupportedNormType.BN:
            self.bn_count += 1
        elif item.anchor_type == SupportedNormType.LN:
            self.ln_count += 1

    def __repr__(self) -> str:
        params = []
        if self.bn_count > 0:
            params.append(f"BN={self.bn_count}")
        if self.ln_count > 0:
            params.append(f"LN={self.ln_count}")
        param_str = ", ".join(params)
        module_name = self.__class__.__name__
        return super().__repr__().replace(module_name, f"{module_name}({param_str})")


class CascadedNorm(nn.Module):
    """
    CascadedNorm: Cascaded Input Distribution Normalization
        with Source Flow Adaptation loss via Norm Linearization

    Learns to normalize input pixel distribution through adaptive
    transformation with cascaded layer-wise supervision.

    The term "normalization" refers to distribution matching
    between source and target domains, achieved through:
    - Gamma Transform: Histogram stretching with gamma correction
    - CLAHE Transform: Histogram equalization with adaptive contrast enhancement

    Args:
        config: CascadedNormConfig with learning parameters
    """

    def __init__(self, config: CascadedNormConfig):
        super().__init__()
        self.config = config

        self.itm = InputTransformation(config)  # Input Transformation Module
        self.anchors = AnchorList()

    def online_parameters(self):
        """Get learnable parameters for optimization."""
        return self.itm.parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.itm(x)

    def diff(self) -> torch.Tensor:
        align_loss = torch.stack([anchor.diff() for anchor in self.anchors]).sum()
        reg_loss = (self.itm.gate_raw - 0.0).pow(2).mean()
        return align_loss + 0.1 * reg_loss


class CascadeAnchor(nn.Module):
    """
    Cascade Anchor Module which tries to linearize the original normalization layer.

    Supported normalization layers:
    - nn.BatchNorm2d / FrozenBatchNorm2d
    - nn.LayerNorm
    """

    def __init__(self, config: CascadedNormConfig, original_norm: nn.Module, loss_fn: nn.Module):
        super().__init__()
        self.config = config
        self.norm = original_norm
        self.loss_fn = loss_fn
        self.anchor_type: SupportedNormType = SupportedNormType.BN
        self.norm_shape: Tuple[int, ...] = tuple()
        self.target_shape: Tuple[int, ...] = tuple()

        self._forward_stats = dict(mean=[], var=[])  # Will be updated per one forward pass
        self._target_stats = dict(mean=[], var=[])  # N(0, 1)

        self.weight = original_norm.weight
        self.bias = original_norm.bias
        self.eps = original_norm.eps

        if isinstance(original_norm, nn.BatchNorm2d) or "BatchNorm2d" in original_norm.__class__.__name__:
            self.anchor_type = SupportedNormType.BN
            self.norm_shape = original_norm.running_mean.shape  # (C,)
            self.target_shape = self.norm_shape

            # BN target: Channel-wise statistics (C,)
            self._target_stats = dict(
                mean=torch.zeros(self.target_shape, device=self.weight.device),
                var=torch.ones(self.target_shape, device=self.weight.device)
            )
        elif isinstance(original_norm, nn.LayerNorm) or "LayerNorm" in original_norm.__class__.__name__:
            self.anchor_type = SupportedNormType.LN
            self.target_shape = ()  # Scalar target for position-wise stats
            self.norm_shape = original_norm.normalized_shape

            # LN target: Scalar statistics (0, 1) broadcasted to (B, H, W)
            self._target_stats = dict(
                mean=torch.tensor(0.0, device=self.weight.device),
                var=torch.tensor(1.0, device=self.weight.device)
            )
        else:
            raise NotImplementedError(f"Unsupported normalization layer: {type(original_norm)}")

    def __repr__(self) -> str:
        module_name = self.__class__.__name__
        return f"{module_name}(norm={self.norm}, loss_fn={self.loss_fn.__class__.__name__})"

    def forward(self, x):
        # Norm Linearization
        match self.anchor_type:
            case SupportedNormType.BN:
                # calculate stats -> this will be optimized to N(0,1)
                dims = (0, 2, 3)
                self._forward_stats['mean'] = forward_mean = x.mean(dim=dims)
                self._forward_stats['var'] = forward_var = x.var(dim=dims, unbiased=False)
            case SupportedNormType.LN:
                # calculate stats -> this will be optimized to N(0,1)
                # Input: (B, ..., *normalized_shape) → mean over normalized_shape → shape: (B, H, W)
                # Target: Scalar (0, 1) <expanded to (B, H, W)>
                dims = tuple(range(-len(self.norm_shape), 0))
                self._forward_stats['mean'] = x.mean(dim=dims)
                self._forward_stats['var'] = x.var(dim=dims, unbiased=False)

                # Update target stats to match forward stats shape (using expand for efficiency)
                # We use the initial scalar targets (0, 1) and expand them
                self._target_stats['mean'] = torch.tensor(0.0, device=x.device).expand_as(self._forward_stats['mean'])
                self._target_stats['var'] = torch.tensor(1.0, device=x.device).expand_as(self._forward_stats['var'])

                # do scale and bias
                return self.norm(x)  # LN is automatically bypassed softly
            case _:
                raise ValueError(f"Unsupported anchor type: {self.anchor_type}")

    def diff(self) -> torch.Tensor:
        """Compute Flow Adaptation loss."""
        try:
            mean_loss = self.loss_fn(self._forward_stats['mean'], self._target_stats['mean'])
            var_loss = self.loss_fn(self._forward_stats['var'], self._target_stats['var'])
            self._forward_stats['mean'] = []  # reset for next forward pass
            self._forward_stats['var'] = []  # reset for next forward pass
        except Exception as e:
            raise RuntimeError(f"Forward statistics not yet collected for {self.norm}.") from e
        return mean_loss + var_loss


class CascadedNormEngine(AdaptationEngine):
    """
    Cascaded Norm Adaptation Engine
    Injects CascadedNorm module into the base model.

    Supports:
    - nn.BatchNorm2d / FrozenBatchNorm2d
    - nn.LayerNorm

    The engine extracts normalization layers from the base model,
    overrides foward to adapt source flow.

    Args:
        base_model: Pre-trained model
        config: CascadedNormConfig with learning parameters

    Example:
        >>> config = CascadedNormConfig(
        ...     itm_type="gamma",
        ...     cascade_target=["layer4"]
        ... )
        >>> adaptive_model = CascadedNormEngine(base_model, config)
        >>> output = adaptive_model(batch)
    """
    model_name: str = "CascadedNormEngine"
    loss_class = nn.MSELoss

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
        if self.config.verbose:
            print(f"[CascadedNorm] Summary:")
            print(f"  View Transform method: {self.config.itm_type}")
        self._extract_norm_layers()

        # Stats
        self._reset_stats()

        # Device
        self.to(self.device, dtype=self.dtype)

        if self.config.verbose:
            print("\n", self.dist_norm)

    @staticmethod
    def _is_norm_layer(module: nn.Module) -> bool:
        return isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)) or "BatchNorm2d" in module.__class__.__name__ or "LayerNorm" in module.__class__.__name__

    def _extract_norm_layers(self):
        # count every norm layers
        if self.config.verbose:
            norm_layer_keys = []
            for name, module in self.base_model.named_modules():
                if self._is_norm_layer(module):
                    norm_layer_keys.append(name)
            print(f"  Total norm layers: {len(norm_layer_keys)}")
            print(f"  Norm layer keys: {norm_layer_keys}")

        # Recursively extract norm layers and replace with CascadeAnchors
        # Compile pattern: if None, match all; otherwise match provided patterns
        cascade_target_re = None if self.config.cascade_target is None else re.compile(
            "|".join(f"({p})" for p in self.config.cascade_target)
        )
        applied_list = self.dist_norm.anchors
        applied_key_list = []
        for name, module in self.base_model.named_modules():
            # Check if module is a norm layer and matches the pattern (if specified)
            if self._is_norm_layer(module) and (cascade_target_re is None or cascade_target_re.search(name)):
                # Check if already wrapped (to prevent double wrapping if called multiple times)
                if isinstance(module, CascadeAnchor):
                    continue
                anchor = CascadeAnchor(self.config, module, self.loss_function)
                applied_list.append(anchor)
                applied_key_list.append(name)
                self._replace_module(name, anchor)  # Replace the original norm layer with the anchor
        print(f"  Applied to {len(self.dist_norm.anchors)} norm layers")
        print(f"  Applied norm layer keys: {applied_key_list}")

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
                    input_dict = kwargs if kwargs else args[0]  # HuggingFace (RT-DETR) uses "pixel_values", Ultralytics (YOLO) uses "img"

                    # Determine which key to use
                    if 'pixel_values' in input_dict:
                        img_key = 'pixel_values'
                    elif 'img' in input_dict:
                        img_key = 'img'
                    else:
                        raise ValueError(f"No image key found in input_dict: {input_dict}")

                    img = input_dict[img_key].to(self.device)

                    # recover original scale
                    original_scale = img.max() <= 1.0
                    if original_scale:
                        img = img * 255.0

                    # apply dist norm for each image in batch
                    transformed_list = []
                    for i in range(img.shape[0]):
                        img_transformed = self.dist_norm(img[i])
                        transformed_list.append(img_transformed)

                    img_batch = torch.stack(transformed_list, dim=0)
                    input_dict[img_key] = img_batch / 255.0 if original_scale else img_batch
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
        transform_params = {}
        if self.config.itm_type == "gamma":
            gate, clip_low, clip_high, gamma = self.dist_norm.itm.applied_params
            transform_params = {
                'gate': gate,
                'gamma': gamma,
                'clip_low': clip_low,
                'clip_high': clip_high
            }
        elif self.config.itm_type == "clahe":
            gate, clip_limit, tile_size = self.dist_norm.itm.applied_params
            transform_params = {
                'gate': gate,
                'clip_limit': clip_limit,
                'tile_size': tile_size
            }
        self._stats['transform_params'].append(transform_params)

        return result

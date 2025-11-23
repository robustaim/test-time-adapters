"""
v10: Speed-optimized version of v6

Optimizations applied (no impact on gradient flow):
1. FFT channel parallelization + rfft2 (~2x faster)
2. Mask caching (avoid recomputation for same resolution)
3. Downsampling for degradation analysis (16x faster FFT)

All optimizations are in compute_degradation which runs under torch.no_grad(),
so gradient flow through controller and stretcher is unchanged.
"""
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Literal, List

# Import base classes for TTA integration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ttadapters.methods.base import AdaptationEngine, AdaptationConfig
from ttadapters.models.base import BaseModel


class FrequencyDegradationAnalyzer:
    """
    Analyzes frequency degradation of images using FFT.

    v10 optimizations:
    - rfft2 instead of fft2 (2x faster for real input)
    - Batch processing all channels at once
    - Mask caching for same resolution
    - Optional downsampling for faster analysis
    """

    def __init__(
        self,
        cutoff_ratio: float = 0.1,
        device: str = 'cuda',
        downsample_size: Optional[int] = 256  # New: downsample for speed
    ):
        self.cutoff_ratio = cutoff_ratio
        self.device = device
        self.downsample_size = downsample_size

        # Mask cache
        self._mask_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _get_mask(self, H: int, W: int) -> torch.Tensor:
        """Get or create cached low-pass mask."""
        # For rfft2, output width is W//2 + 1
        W_rfft = W // 2 + 1
        cache_key = (H, W_rfft)

        if cache_key not in self._mask_cache:
            crow, ccol = H // 2, 0  # rfft2 has DC at left
            radius = int(min(H, W) * self.cutoff_ratio)

            y = torch.arange(H, device=self.device).view(-1, 1)
            x = torch.arange(W_rfft, device=self.device).view(1, -1)

            # For rfft2, we need to handle the shifted frequency layout
            # DC component is at (0, 0), not center
            # Create mask for low frequencies (near DC)
            mask_low = (x**2 + y**2 <= radius**2).float()
            # Also include the mirrored low frequencies
            mask_low = mask_low + ((x**2 + (H - y)**2 <= radius**2) & (y > 0)).float()
            mask_low = torch.clamp(mask_low, 0, 1)

            self._mask_cache[cache_key] = mask_low

        return self._mask_cache[cache_key]

    def compute_degradation(self, image: torch.Tensor) -> Tuple[float, float]:
        """
        Compute frequency-based degradation score.

        Optimized version:
        - Downsamples image for faster FFT
        - Uses rfft2 (real FFT, 2x faster)
        - Processes all channels in parallel
        - Caches frequency masks

        Args:
            image: torch.Tensor of shape (C, H, W) or (H, W, C), values [0-255]

        Returns:
            degradation_score: float in [0, 1], higher = more degraded
            high_low_ratio: float, ratio of high/low frequency energy
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = image.to(self.device).float()

        # Handle channel dimension: convert to (C, H, W)
        if len(image.shape) == 3:
            if image.shape[0] > image.shape[2]:  # Likely (H, W, C)
                image = image.permute(2, 0, 1)
        elif len(image.shape) == 2:  # Grayscale
            image = image.unsqueeze(0)

        C, H, W = image.shape

        # Optimization 3: Downsample for faster FFT
        if self.downsample_size is not None and min(H, W) > self.downsample_size:
            scale = self.downsample_size / min(H, W)
            new_H, new_W = int(H * scale), int(W * scale)
            # Make dimensions even for FFT
            new_H = new_H - (new_H % 2)
            new_W = new_W - (new_W % 2)
            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_H, new_W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            H, W = new_H, new_W

        # Optimization 1 & 2: Batch rfft2 for all channels at once
        # rfft2 is ~2x faster than fft2 for real input
        f = torch.fft.rfft2(image)  # (C, H, W//2+1)

        # Get cached mask
        mask_low = self._get_mask(H, W)  # (H, W//2+1)

        # Compute energies for all channels at once
        f_magnitude_sq = torch.abs(f) ** 2  # (C, H, W//2+1)

        low_energy = (f_magnitude_sq * mask_low).sum(dim=(1, 2))  # (C,)
        high_energy = (f_magnitude_sq * (1 - mask_low)).sum(dim=(1, 2))  # (C,)

        # Average ratio across channels
        high_low_ratios = high_energy / (low_energy + 1e-8)  # (C,)
        avg_high_low_ratio = high_low_ratios.mean().item()

        # Log scale normalization
        log_ratio = np.log10(avg_high_low_ratio + 1e-8)
        normalized_ratio = np.clip((log_ratio + 4) / 4, 0, 1)

        # high_low_ratio 높음 = detail 많음 = degradation 낮음
        degradation_score = 1.0 - normalized_ratio

        return degradation_score, avg_high_low_ratio

    def clear_cache(self):
        """Clear mask cache."""
        self._mask_cache.clear()


class StretchingController(nn.Module):
    """
    Learnable MLP controller that maps degradation score to stretching parameters.

    g_φ(s) → (clip_low, clip_high, strength)

    Output ranges:
        - clip_low: [0, 10] (percentile)
        - clip_high: [90, 100] (percentile)
        - strength: [0.0, 1.0]
    """

    def __init__(self, hidden_dim: int = 16, init_to_prior: bool = True):
        """
        Args:
            hidden_dim: Hidden layer dimension
            init_to_prior: If True, initialize weights to approximate hand-crafted prior
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()  # Output in [0, 1]
        )

        if init_to_prior:
            self._init_to_prior()

    def _init_to_prior(self):
        """Initialize network to approximate hand-crafted mapping."""
        # Small random init - will rely on L_reg to guide toward prior
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.zeros_(m.bias)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            s: degradation score, shape (B,) or (B, 1), values in [0, 1]

        Returns:
            clip_low: shape (B,), values in [0, 10]
            clip_high: shape (B,), values in [90, 100]
            strength: shape (B,), values in [0, 1]
        """
        if s.dim() == 1:
            s = s.unsqueeze(-1)  # (B,) -> (B, 1)

        raw = self.net(s)  # (B, 3), values in [0, 1]

        # Scale to meaningful ranges
        clip_low = raw[:, 0] * 10          # [0, 10]
        clip_high = 90 + raw[:, 1] * 10    # [90, 100]
        strength = raw[:, 2]               # [0, 1]

        return clip_low, clip_high, strength

    @staticmethod
    def get_prior(s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hand-crafted mapping h(s) - serves as regularization anchor.

        Args:
            s: degradation score, shape (B,) or scalar

        Returns:
            clip_low_prior, clip_high_prior, strength_prior
        """
        if not isinstance(s, torch.Tensor):
            s = torch.tensor(s)

        s = s.flatten()

        # Hand-crafted mapping:
        # degradation_score=0 (clear): strength=0.2, low=5, high=95
        # degradation_score=1 (degraded): strength=1.0, low=1, high=99
        strength_prior = 0.2 + 0.8 * s
        clip_low_prior = 5 - 4 * s
        clip_high_prior = 95 + 4 * s

        return clip_low_prior, clip_high_prior, strength_prior


class DifferentiableHistogramStretcher(nn.Module):
    """
    Differentiable histogram stretching module.
    Uses soft percentile approximation for gradient flow.
    """

    def __init__(self, temperature: float = 0.01):
        """
        Args:
            temperature: Temperature for soft percentile (lower = sharper)
        """
        super().__init__()
        self.temperature = temperature

    def soft_percentile(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Differentiable approximation of percentile.

        Args:
            x: Input tensor, shape (C, H, W) - single channel or full image
            p: Percentile value(s) in [0, 100], shape () or (1,)

        Returns:
            Approximate percentile value
        """
        x_flat = x.flatten()
        n = x_flat.shape[0]

        # Ensure p is on same device
        p = p.to(x.device)

        # Target index
        idx = (p / 100.0) * (n - 1)

        # Create soft index weights
        indices = torch.arange(n, device=x.device, dtype=x.dtype)
        weights = F.softmax(-(indices - idx).abs() / (self.temperature * n), dim=0)

        # Sort and compute weighted sum
        sorted_x, _ = torch.sort(x_flat)
        return (weights * sorted_x).sum()

    def stretch_channel(
        self,
        channel: torch.Tensor,
        clip_low: torch.Tensor,
        clip_high: torch.Tensor,
        strength: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply stretching to a single channel.

        Args:
            channel: shape (H, W), values [0, 255]
            clip_low: percentile for low clip [0, 100]
            clip_high: percentile for high clip [0, 100]
            strength: blending strength [0, 1]

        Returns:
            Stretched channel, shape (H, W)
        """
        # Compute soft percentiles
        low_val = self.soft_percentile(channel, clip_low)
        high_val = self.soft_percentile(channel, clip_high)

        # Soft clipping using sigmoid
        # Instead of hard clamp, use smooth approximation
        scale = 50.0  # Sharpness of soft clamp

        # Soft lower bound: max(x, low_val) ≈ low_val + softplus(x - low_val)
        clipped = low_val + F.softplus((channel - low_val) * scale) / scale
        # Soft upper bound: min(x, high_val)
        clipped = high_val - F.softplus((high_val - clipped) * scale) / scale

        # Normalize to [0, 1]
        range_val = high_val - low_val + 1e-6
        normalized = (clipped - low_val) / range_val

        # Blend with original
        original_normalized = channel / 255.0
        stretched_normalized = normalized * strength + original_normalized * (1 - strength)

        # Scale back to [0, 255]
        return torch.clamp(stretched_normalized * 255.0, 0, 255)

    def forward(
        self,
        image: torch.Tensor,
        clip_low: torch.Tensor,
        clip_high: torch.Tensor,
        strength: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply differentiable histogram stretching.

        Args:
            image: shape (C, H, W), values [0, 255]
            clip_low: shape () or (1,), percentile [0, 100]
            clip_high: shape () or (1,), percentile [0, 100]
            strength: shape () or (1,), blending [0, 1]

        Returns:
            Stretched image, shape (C, H, W)
        """
        C, H, W = image.shape
        stretched = torch.zeros_like(image)

        for c in range(C):
            stretched[c] = self.stretch_channel(
                image[c], clip_low, clip_high, strength
            )

        return stretched


class ContrastStretchingLoss(nn.Module):
    """
    Loss functions for training the stretching controller.

    L = L_improve + λ_reg * L_reg + λ_loc * L_loc

    - L_improve: Entropy improvement loss (stretched should have lower entropy)
    - L_reg: Regularization toward hand-crafted prior
    - L_loc: Localization consistency (optional)
    """

    def __init__(
        self,
        lambda_reg: float = 1.0,
        lambda_loc: float = 0.0,
        entropy_margin: float = 0.1,
        confidence_threshold: float = 0.3
    ):
        """
        Args:
            lambda_reg: Weight for regularization loss
            lambda_loc: Weight for localization consistency loss
            entropy_margin: Margin for entropy improvement
            confidence_threshold: Threshold for high-confidence predictions
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.lambda_loc = lambda_loc
        self.entropy_margin = entropy_margin
        self.confidence_threshold = confidence_threshold

    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of probability distribution.

        Args:
            probs: shape (N, C) or (N,) for binary

        Returns:
            Mean entropy, scalar
        """
        if probs.dim() == 1:
            probs = torch.stack([probs, 1 - probs], dim=-1)

        # Clamp for numerical stability
        probs = torch.clamp(probs, 1e-8, 1 - 1e-8)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)

        return entropy.mean()

    def entropy_improvement_loss(
        self,
        entropy_orig: torch.Tensor,
        entropy_stretched: torch.Tensor
    ) -> torch.Tensor:
        """
        L_improve = max(0, H(stretched) - H(orig) + margin)

        Penalize if stretched has higher entropy than original.
        """
        return F.relu(entropy_stretched - entropy_orig + self.entropy_margin)

    def regularization_loss(
        self,
        params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        prior_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        L_reg = ||params - prior_params||^2

        Keep controller output close to hand-crafted mapping.
        """
        clip_low, clip_high, strength = params
        clip_low_prior, clip_high_prior, strength_prior = prior_params

        loss = (
            (clip_low - clip_low_prior) ** 2 +
            (clip_high - clip_high_prior) ** 2 +
            (strength - strength_prior) ** 2 * 100  # Scale strength loss
        )

        return loss.mean()

    def localization_loss(
        self,
        boxes_orig: torch.Tensor,
        boxes_stretched: torch.Tensor,
        iou_threshold: float = 0.5
    ) -> torch.Tensor:
        """
        L_loc = 1 - IoU(boxes_orig, boxes_stretched)

        Encourage spatial consistency between original and stretched predictions.
        """
        if boxes_orig.shape[0] == 0 or boxes_stretched.shape[0] == 0:
            return torch.tensor(0.0, device=boxes_orig.device)

        # Simple L1 loss on matched boxes (assuming same order)
        # For more robust matching, use Hungarian algorithm
        n = min(boxes_orig.shape[0], boxes_stretched.shape[0])

        return F.l1_loss(boxes_orig[:n], boxes_stretched[:n])

    def forward(
        self,
        entropy_orig: torch.Tensor,
        entropy_stretched: torch.Tensor,
        params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        prior_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        boxes_orig: Optional[torch.Tensor] = None,
        boxes_stretched: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Returns:
            Dictionary with 'total', 'improve', 'reg', 'loc' losses
        """
        l_improve = self.entropy_improvement_loss(entropy_orig, entropy_stretched)
        l_reg = self.regularization_loss(params, prior_params)

        l_loc = torch.tensor(0.0, device=entropy_orig.device)
        if self.lambda_loc > 0 and boxes_orig is not None and boxes_stretched is not None:
            l_loc = self.localization_loss(boxes_orig, boxes_stretched)

        total = l_improve + self.lambda_reg * l_reg + self.lambda_loc * l_loc

        return {
            'total': total,
            'improve': l_improve,
            'reg': l_reg,
            'loc': l_loc
        }


class FrequencyAdaptiveNormalizer(nn.Module):
    """
    Full pipeline with learnable controller:

    1. Analyze degradation: A(x) → s
    2. Controller: g_φ(s) → (clip_low, clip_high, strength)
    3. Stretch: stretch(x, params) → x̃

    Training:
    - θ (detector) is frozen
    - φ (controller) is updated
    """

    def __init__(
        self,
        cutoff_ratio: float = 0.1,
        hidden_dim: int = 16,
        temperature: float = 0.01,
        device: str = 'cuda',
        enabled: bool = True,
        downsample_size: Optional[int] = 256  # v10: downsample for speed
    ):
        super().__init__()

        self.analyzer = FrequencyDegradationAnalyzer(
            cutoff_ratio=cutoff_ratio,
            device=device,
            downsample_size=downsample_size
        )
        self.controller = StretchingController(
            hidden_dim=hidden_dim,
            init_to_prior=True
        )
        self.stretcher = DifferentiableHistogramStretcher(
            temperature=temperature
        )

        self.enabled = enabled
        self.device = device

        # Move controller to device
        self.controller = self.controller.to(device)

        # Statistics tracking
        self.stats = {
            'degradation_scores': [],
            'high_low_ratios': [],
            'params': [],  # (clip_low, clip_high, strength)
        }

    def forward(
        self,
        x: torch.Tensor,
        return_params: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: Image tensor, shape (C, H, W) or (B, C, H, W), values [0-255]
            return_params: If True, also return controller parameters

        Returns:
            Stretched image(s)
            Optionally: (stretched, params, prior_params, degradation_score)
        """
        if not self.enabled:
            if return_params:
                return x, None, None, None
            return x

        # Handle batch dimension
        if len(x.shape) == 4:  # (B, C, H, W)
            return self._process_batch(x, return_params)
        else:  # (C, H, W)
            return self._process_single(x, return_params)

    def _process_single(
        self,
        image: torch.Tensor,
        return_params: bool = False
    ):
        """Process a single image."""
        # 1. Analyze degradation (no gradient needed)
        with torch.no_grad():
            degradation_score, high_low_ratio = self.analyzer.compute_degradation(image)

        # Convert to tensor
        s = torch.tensor([degradation_score], device=self.device, dtype=torch.float32)

        # 2. Get controller parameters (gradient flows here)
        clip_low, clip_high, strength = self.controller(s)

        # 3. Get prior for regularization
        prior_params = self.controller.get_prior(s)
        prior_params = tuple(p.to(self.device) for p in prior_params)

        # 4. Apply stretching (gradient flows here)
        stretched = self.stretcher(
            image,
            clip_low[0],
            clip_high[0],
            strength[0]
        )

        # Track statistics
        self.stats['degradation_scores'].append(degradation_score)
        self.stats['high_low_ratios'].append(high_low_ratio)
        self.stats['params'].append((
            clip_low[0].item(),
            clip_high[0].item(),
            strength[0].item()
        ))

        if return_params:
            params = (clip_low, clip_high, strength)
            return stretched, params, prior_params, s

        return stretched

    def _process_batch(
        self,
        images: torch.Tensor,
        return_params: bool = False
    ):
        """Process a batch of images."""
        B = images.shape[0]
        results = []
        all_params = []
        all_priors = []
        all_scores = []

        for i in range(B):
            if return_params:
                stretched, params, prior, s = self._process_single(
                    images[i], return_params=True
                )
                all_params.append(params)
                all_priors.append(prior)
                all_scores.append(s)
            else:
                stretched = self._process_single(images[i], return_params=False)

            results.append(stretched)

        batch_result = torch.stack(results, dim=0)

        if return_params:
            # Stack parameters
            stacked_params = tuple(
                torch.cat([p[j] for p in all_params], dim=0)
                for j in range(3)
            )
            stacked_priors = tuple(
                torch.cat([p[j] for p in all_priors], dim=0)
                for j in range(3)
            )
            stacked_scores = torch.cat(all_scores, dim=0)
            return batch_result, stacked_params, stacked_priors, stacked_scores

        return batch_result

    def get_stats(self) -> Optional[Dict]:
        """Get statistics of processed images."""
        if len(self.stats['degradation_scores']) == 0:
            return None

        params_array = np.array(self.stats['params'])

        return {
            'mean_degradation': np.mean(self.stats['degradation_scores']),
            'std_degradation': np.std(self.stats['degradation_scores']),
            'mean_high_low_ratio': np.mean(self.stats['high_low_ratios']),
            'mean_clip_low': np.mean(params_array[:, 0]),
            'mean_clip_high': np.mean(params_array[:, 1]),
            'mean_strength': np.mean(params_array[:, 2]),
            'num_processed': len(self.stats['degradation_scores'])
        }

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'degradation_scores': [],
            'high_low_ratios': [],
            'params': [],
        }


class FrequencyControlledModel(nn.Module):
    """
    Model wrapper that applies frequency-based normalization with learnable controller.

    Usage for TTA:
        model = FrequencyControlledModel(base_model, normalizer)
        optimizer = torch.optim.Adam(model.normalizer.controller.parameters(), lr=1e-4)

        for batch in dataloader:
            # Forward with params for loss computation
            outputs, params, priors, scores = model(batch, return_params=True)

            # Compute loss and update controller
            loss = loss_fn(...)
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        base_model: nn.Module,
        normalizer: FrequencyAdaptiveNormalizer
    ):
        super().__init__()
        self.base_model = base_model
        self.normalizer = normalizer

        # Copy attributes from base model
        if hasattr(base_model, 'model_name'):
            self.model_name = f"{base_model.model_name}+FreqCtrl"

        for attr in ['num_classes', 'dataset_name', 'pixel_mean', 'pixel_std',
                     'input_format', 'model_provider']:
            if hasattr(base_model, attr):
                setattr(self, attr, getattr(base_model, attr))

    def forward(
        self,
        batched_inputs,
        return_params: bool = False
    ):
        """
        Args:
            batched_inputs: List of input dicts with 'image' key
            return_params: If True, also return controller parameters

        Returns:
            Model outputs
            Optionally: (outputs, params, priors, scores)
        """
        normalized_inputs = []
        all_params = []
        all_priors = []
        all_scores = []

        for input_dict in batched_inputs:
            if 'image' in input_dict:
                img = input_dict['image']

                # Ensure [0-255] range
                if img.max() <= 1.0:
                    img = img * 255.0

                # Apply normalization
                if return_params:
                    normalized_img, params, priors, score = self.normalizer(
                        img, return_params=True
                    )
                    all_params.append(params)
                    all_priors.append(priors)
                    all_scores.append(score)
                else:
                    normalized_img = self.normalizer(img)

                # Restore original scale if needed
                if input_dict['image'].max() <= 1.0:
                    normalized_img = normalized_img / 255.0

                new_input = input_dict.copy()
                new_input['image'] = normalized_img
                normalized_inputs.append(new_input)
            else:
                normalized_inputs.append(input_dict)

        # Forward through base model
        outputs = self.base_model(normalized_inputs)

        if return_params and all_params:
            return outputs, all_params, all_priors, all_scores

        return outputs

    def eval(self):
        self.base_model.eval()
        return self

    def train(self, mode: bool = True):
        # Keep base model in eval, only controller in train
        self.base_model.eval()
        self.normalizer.controller.train(mode)
        return self


# Convenience function for creating the full pipeline
def create_frequency_tta_pipeline(
    base_model: nn.Module,
    cutoff_ratio: float = 0.1,
    hidden_dim: int = 16,
    lr: float = 1e-4,
    lambda_reg: float = 1.0,
    lambda_loc: float = 0.0,
    device: str = 'cuda',
    downsample_size: Optional[int] = 256  # v10: downsample for speed
) -> Tuple[FrequencyControlledModel, torch.optim.Optimizer, ContrastStretchingLoss]:
    """
    Create complete TTA pipeline with model, optimizer, and loss.

    Returns:
        model: FrequencyControlledModel
        optimizer: Adam optimizer for controller
        loss_fn: ContrastStretchingLoss
    """
    normalizer = FrequencyAdaptiveNormalizer(
        cutoff_ratio=cutoff_ratio,
        hidden_dim=hidden_dim,
        device=device,
        downsample_size=downsample_size
    )

    model = FrequencyControlledModel(base_model, normalizer)

    # Only optimize controller parameters
    optimizer = torch.optim.Adam(
        model.normalizer.controller.parameters(),
        lr=lr
    )

    loss_fn = ContrastStretchingLoss(
        lambda_reg=lambda_reg,
        lambda_loc=lambda_loc
    )

    return model, optimizer, loss_fn


# =============================================================================
# AdaptationEngine Integration for TTA Pipeline
# =============================================================================

@dataclass
class FrequencyAdaptationConfig(AdaptationConfig):
    """Configuration for FrequencyAdaptationEngine."""
    adaptation_name: str = "FrequencyAdaptationEngine"

    # Frequency analyzer settings
    cutoff_ratio: float = 0.1
    downsample_size: Optional[int] = 256  # v10: downsample for speed (None to disable)

    # Controller settings
    hidden_dim: int = 16
    temperature: float = 0.01

    # Loss settings
    lambda_reg: float = 1.0
    lambda_loc: float = 0.0
    entropy_margin: float = 0.1
    confidence_threshold: float = 0.3

    # Optimizer settings
    optim: Literal["SGD", "Adam"] = "Adam"
    adapt_lr: float = 1e-4


class FrequencyAdaptationEngine(AdaptationEngine):
    """
    TTA Engine with learnable frequency-based contrast stretching.

    v10 optimizations:
    - rfft2 instead of fft2 (2x faster)
    - Batch channel processing
    - Mask caching
    - Optional downsampling for degradation analysis

    Pipeline:
    1. Compute degradation score s = A(x)
    2. Controller predicts params: g_φ(s) → (clip_low, clip_high, strength)
    3. Apply stretching: x̃ = stretch(x, params)
    4. Forward through frozen detector: y = f_θ(x̃)
    5. Compute loss and update φ

    Loss:
    L = L_improve + λ_reg * L_reg
    - L_improve: max(0, H(x̃) - H(x) + margin)
    - L_reg: ||params - prior||²
    """

    model_name: str = "FrequencyAdaptationEngine"

    def __init__(self, base_model: BaseModel, config: FrequencyAdaptationConfig):
        # Initialize parent
        super().__init__(base_model, config)

        self.freq_config = config

        # Initialize frequency components (with v10 optimizations)
        self.analyzer = FrequencyDegradationAnalyzer(
            cutoff_ratio=config.cutoff_ratio,
            device='cuda',
            downsample_size=config.downsample_size
        )

        self.controller = StretchingController(
            hidden_dim=config.hidden_dim,
            init_to_prior=True
        )

        self.stretcher = DifferentiableHistogramStretcher(
            temperature=config.temperature
        )

        self.loss_fn = ContrastStretchingLoss(
            lambda_reg=config.lambda_reg,
            lambda_loc=config.lambda_loc,
            entropy_margin=config.entropy_margin,
            confidence_threshold=config.confidence_threshold
        )

        # Statistics tracking
        self.stats = {
            'degradation_scores': [],
            'params': [],
            'losses': [],
        }

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.controller = self.controller.to(self._device)
        self.stretcher = self.stretcher.to(self._device)
        self.analyzer.device = self._device
        return result

    def online_parameters(self):
        """Only controller parameters are updated during TTA."""
        return self.controller.parameters()

    @property
    def optimizer(self):
        if self._optimizer is None:
            if self.freq_config.optim == "Adam":
                self._optimizer = optim.Adam(
                    self.online_parameters(),
                    lr=self.freq_config.adapt_lr
                )
            else:
                self._optimizer = optim.SGD(
                    self.online_parameters(),
                    lr=self.freq_config.adapt_lr
                )
        return self._optimizer

    def _normalize_image(self, img: torch.Tensor, return_params: bool = False):
        """Apply frequency-based normalization to a single image."""
        # Ensure device
        img = img.to(self._device)

        # Ensure [0-255] range
        original_scale = img.max() <= 1.0
        if original_scale:
            img = img * 255.0

        # 1. Analyze degradation (no gradient)
        with torch.no_grad():
            degradation_score, _ = self.analyzer.compute_degradation(img)

        s = torch.tensor([degradation_score], device=self._device, dtype=torch.float32)

        # 2. Get controller parameters (gradient flows here)
        clip_low, clip_high, strength = self.controller(s)

        # 3. Get prior for regularization
        prior_params = self.controller.get_prior(s)
        prior_params = tuple(p.to(self._device) for p in prior_params)

        # 4. Apply stretching
        stretched = self.stretcher(img, clip_low[0], clip_high[0], strength[0])

        # Restore original scale
        if original_scale:
            stretched = stretched / 255.0

        if return_params:
            params = (clip_low, clip_high, strength)
            return stretched, params, prior_params, s

        return stretched

    def _compute_entropy_from_outputs(self, outputs: List[Dict]) -> torch.Tensor:
        """
        Compute mean entropy from detection outputs.

        Works with both Detectron2-style and YOLO-style outputs.
        """
        all_scores = []

        for output in outputs:
            if 'instances' in output:
                # Detectron2 style
                instances = output['instances']
                if hasattr(instances, 'scores') and len(instances.scores) > 0:
                    scores = instances.scores
                    # Filter by confidence
                    mask = scores > self.freq_config.confidence_threshold
                    if mask.sum() > 0:
                        all_scores.append(scores[mask])
            elif 'scores' in output:
                # Direct scores
                scores = output['scores']
                mask = scores > self.freq_config.confidence_threshold
                if mask.sum() > 0:
                    all_scores.append(scores[mask])

        if not all_scores:
            # Return zero entropy if no confident predictions
            return torch.tensor(0.0, device=self._device)

        scores = torch.cat(all_scores)

        # Binary entropy: H = -p*log(p) - (1-p)*log(1-p)
        scores = torch.clamp(scores, 1e-8, 1 - 1e-8)
        entropy = -scores * torch.log(scores) - (1 - scores) * torch.log(1 - scores)

        return entropy.mean()

    def forward(self, batched_inputs):
        """
        Forward pass with TTA.

        During adaptation (self.adapting=True):
        1. Get original predictions (no grad)
        2. Apply stretching and get stretched predictions
        3. Compute loss and update controller

        During inference (self.adapting=False):
        Just apply stretching and forward.
        """
        if not self.adapting:
            # Simple inference mode
            normalized_inputs = []
            for input_dict in batched_inputs:
                if 'image' in input_dict:
                    new_input = input_dict.copy()
                    new_input['image'] = self._normalize_image(input_dict['image'])
                    normalized_inputs.append(new_input)
                else:
                    normalized_inputs.append(input_dict)
            return self.base_model(normalized_inputs)

        # === Adaptation mode ===

        # 1. Get original predictions (no gradient)
        with torch.no_grad():
            outputs_orig = self.base_model(batched_inputs)
            entropy_orig = self._compute_entropy_from_outputs(outputs_orig)

        # 2. Apply stretching with gradient
        normalized_inputs = []
        all_params = []
        all_priors = []

        for input_dict in batched_inputs:
            if 'image' in input_dict:
                stretched, params, priors, s = self._normalize_image(
                    input_dict['image'], return_params=True
                )
                new_input = input_dict.copy()
                new_input['image'] = stretched
                normalized_inputs.append(new_input)
                all_params.append(params)
                all_priors.append(priors)

                # Track stats
                self.stats['degradation_scores'].append(s.item())
                self.stats['params'].append((
                    params[0].item(), params[1].item(), params[2].item()
                ))
            else:
                normalized_inputs.append(input_dict)

        # 3. Get stretched predictions
        outputs_stretched = self.base_model(normalized_inputs)
        entropy_stretched = self._compute_entropy_from_outputs(outputs_stretched)

        # 4. Compute loss
        if all_params:
            # Stack params
            stacked_params = tuple(
                torch.stack([p[i] for p in all_params]).mean()
                for i in range(3)
            )
            stacked_priors = tuple(
                torch.stack([p[i] for p in all_priors]).mean()
                for i in range(3)
            )

            losses = self.loss_fn(
                entropy_orig, entropy_stretched,
                stacked_params, stacked_priors
            )

            # 5. Backward and update
            self.optimizer.zero_grad()
            losses['total'].backward()
            self.optimizer.step()

            # Track loss
            self.stats['losses'].append({
                k: v.item() for k, v in losses.items()
            })

        return outputs_stretched

    def reset(self):
        """Reset controller to initial state."""
        super().reset()
        # Re-initialize controller
        self.controller = StretchingController(
            hidden_dim=self.freq_config.hidden_dim,
            init_to_prior=True
        ).to(self._device)
        self._optimizer = None  # Reset optimizer
        # Clear mask cache
        self.analyzer.clear_cache()
        self.stats = {
            'degradation_scores': [],
            'params': [],
            'losses': [],
        }

    def get_stats(self) -> Optional[Dict]:
        """Get adaptation statistics."""
        if not self.stats['degradation_scores']:
            return None

        params_array = np.array(self.stats['params'])

        result = {
            'mean_degradation': np.mean(self.stats['degradation_scores']),
            'std_degradation': np.std(self.stats['degradation_scores']),
            'mean_clip_low': np.mean(params_array[:, 0]),
            'mean_clip_high': np.mean(params_array[:, 1]),
            'mean_strength': np.mean(params_array[:, 2]),
            'num_processed': len(self.stats['degradation_scores']),
        }

        if self.stats['losses']:
            losses_array = {
                k: np.mean([l[k] for l in self.stats['losses']])
                for k in self.stats['losses'][0].keys()
            }
            result['mean_losses'] = losses_array

        return result


# =============================================================================
# Backward compatibility: v5-style wrapper (no learning, just preprocessing)
# =============================================================================

class FrequencyNormalizedModel(nn.Module):
    """
    v5-compatible wrapper for simple preprocessing without learning.
    Uses hand-crafted prior directly.
    """

    def __init__(
        self,
        base_model: nn.Module,
        cutoff_ratio: float = 0.1,
        device: str = 'cuda',
        downsample_size: Optional[int] = 256
    ):
        super().__init__()
        self.base_model = base_model
        self.analyzer = FrequencyDegradationAnalyzer(
            cutoff_ratio=cutoff_ratio,
            device=device,
            downsample_size=downsample_size
        )
        self.device = device

        # Copy attributes from base model
        if hasattr(base_model, 'model_name'):
            self.model_name = f"{base_model.model_name}+FreqNorm"
        for attr in ['num_classes', 'dataset_name', 'pixel_mean', 'pixel_std',
                     'input_format', 'model_provider', 'DataPreparation']:
            if hasattr(base_model, attr):
                setattr(self, attr, getattr(base_model, attr))

    def _stretch_with_prior(self, image: torch.Tensor) -> torch.Tensor:
        """Apply stretching using hand-crafted prior."""
        original_scale = image.max() <= 1.0
        if original_scale:
            image = image * 255.0

        # Get degradation score
        with torch.no_grad():
            degradation_score, _ = self.analyzer.compute_degradation(image)

        # Use hand-crafted prior
        s = degradation_score
        strength = 0.2 + 0.8 * s
        clip_low = 5 - 4 * s
        clip_high = 95 + 4 * s

        # Apply per-channel stretching
        C, H, W = image.shape
        stretched = torch.zeros_like(image)

        for c in range(C):
            channel = image[c]
            low_val = torch.quantile(channel, clip_low / 100.0)
            high_val = torch.quantile(channel, clip_high / 100.0)

            clipped = torch.clamp(channel, low_val, high_val)
            if high_val - low_val > 1e-6:
                normalized = (clipped - low_val) / (high_val - low_val)
            else:
                normalized = clipped / 255.0

            stretched_norm = normalized * strength + channel / 255.0 * (1 - strength)
            stretched[c] = torch.clamp(stretched_norm * 255.0, 0, 255)

        if original_scale:
            stretched = stretched / 255.0

        return stretched

    def forward(self, batched_inputs):
        normalized_inputs = []
        for input_dict in batched_inputs:
            if 'image' in input_dict:
                new_input = input_dict.copy()
                new_input['image'] = self._stretch_with_prior(input_dict['image'])
                normalized_inputs.append(new_input)
            else:
                normalized_inputs.append(input_dict)
        return self.base_model(normalized_inputs)

    def eval(self):
        self.base_model.eval()
        return self

    def train(self, mode=True):
        self.base_model.train(mode)
        return self

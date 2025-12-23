"""
not_heuristic_apt_frequencyv3.py

Relative Sharpness Loss (완전 Not Heuristic!)

v2의 문제점:
- Parameter regularization이 fixed target (strength=0.5) 사용 → 여전히 heuristic
- Sharpness loss가 absolute value → strength=0으로 수렴 가능

v3의 해결책:
- Relative sharpness loss: "Original 대비 얼마나 향상되었는가"
- NO hand-crafted prior
- NO fixed target
- NO parameter regularization
- Self-regulating: 자동으로 optimal strength 찾음

핵심 아이디어:
L_sharp = -log(high_ratio_stretched / high_ratio_original)

논리:
- strength=0 → stretched=original → improvement=1 → loss=0
- strength 증가 → sharpness 향상 → improvement>1 → loss<0 (reward!)
- strength 너무 높음 → artifact → sharpness 감소 → loss>0 (penalty!)

Pipeline:
1. s = compute_degradation(x)  ← 실제 측정 (frequency-based)
2. Controller: g_φ(s) → (clip_low, clip_high, strength)
3. Stretch: x̃ = stretch(x, params)
4. Loss: L_relative_sharp(x̃, x)  ← Relative improvement!
5. Update φ
6. Forward: y = f_θ(x̃)
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


class SimpleDegradationAnalyzer:
    """
    Frequency-based degradation analyzer (NOT fixed 0.5!).

    논리: Clear images have more high-frequency content
          Degraded/blurry images lose high-frequency details
    """

    def __init__(self, device: str = 'cuda', cutoff_ratio: float = 0.1):
        self.device = device
        self.cutoff_ratio = cutoff_ratio
        self._mask_cache = {}

    def _get_high_freq_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Get or create cached high-frequency mask for rfft2 output."""
        W_rfft = W // 2 + 1
        cache_key = (H, W_rfft)

        if cache_key not in self._mask_cache:
            radius = int(min(H, W) * self.cutoff_ratio)

            y = torch.arange(H, device=device).view(-1, 1)
            x = torch.arange(W_rfft, device=device).view(1, -1)

            # Low frequency mask (near DC component)
            mask_low = (x**2 + y**2 <= radius**2).float()
            mask_low = mask_low + ((x**2 + (H - y)**2 <= radius**2) & (y > 0)).float()
            mask_low = torch.clamp(mask_low, 0, 1)

            # High frequency mask = 1 - low
            mask_high = 1.0 - mask_low

            self._mask_cache[cache_key] = mask_high

        return self._mask_cache[cache_key]

    def compute_degradation(self, image: torch.Tensor) -> Tuple[float, float]:
        """
        Compute degradation score based on frequency analysis.

        Args:
            image: torch.Tensor of shape (C, H, W), values [0-255]

        Returns:
            degradation_score: 0.0 (clear) to 1.0 (degraded)
            high_freq_ratio: statistic value for logging
        """
        with torch.no_grad():
            C, H, W = image.shape

            # Downsample for faster computation
            downsample_size = 64
            if min(H, W) > downsample_size:
                scale = downsample_size / min(H, W)
                new_H, new_W = int(H * scale), int(W * scale)
                new_H = new_H - (new_H % 2)
                new_W = new_W - (new_W % 2)
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=(new_H, new_W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                H, W = new_H, new_W

            # Move to device
            image = image.to(self.device)

            # FFT
            f = torch.fft.rfft2(image)  # (C, H, W//2+1)
            f_magnitude = torch.abs(f)

            # Get high frequency mask
            mask_high = self._get_high_freq_mask(H, W, image.device)

            # Compute energies
            high_energy = (f_magnitude * mask_high).sum()
            total_energy = f_magnitude.sum() + 1e-8

            # High frequency ratio (0 to 1)
            high_ratio = (high_energy / total_energy).item()

            # Degradation score = 1 - high_ratio
            # Clear image (high ratio) → low degradation score
            # Blurry image (low ratio) → high degradation score
            degradation_score = 1.0 - high_ratio

            # Clamp to [0, 1]
            degradation_score = max(0.0, min(1.0, degradation_score))

            return degradation_score, high_ratio


class StretchingController(nn.Module):
    """
    Learnable MLP controller that maps degradation score to stretching parameters.

    g_φ(s) → (clip_low, clip_high, strength)

    Output ranges:
        - clip_low: [0, 10] (percentile)
        - clip_high: [90, 100] (percentile)
        - strength: [0.0, 1.0]

    Note: NO hand-crafted prior! Controller learns freely.
    """

    def __init__(self, hidden_dim: int = 16):
        """
        Args:
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Random initialization (no prior initialization)
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


class RelativeSharpnessLoss(nn.Module):
    """
    Relative frequency-based sharpness loss (완전 Not Heuristic!).

    L = -log(high_ratio_stretched / high_ratio_original)

    논리적 정당화:
    1. Stretching의 효과를 직접 측정: "얼마나 sharpness가 향상되었는가?"
    2. Self-regulating:
       - strength=0 → no improvement → loss=0
       - strength 증가 → sharpness 향상 → loss<0 (reward!)
       - strength 너무 높음 → artifact → sharpness 감소 → loss>0 (penalty!)
    3. NO hand-crafted prior, NO heuristic threshold, NO fixed target
    """

    def __init__(
        self,
        cutoff_ratio: float = 0.1,
        downsample_size: int = 64,
    ):
        """
        Args:
            cutoff_ratio: Cutoff ratio for low/high frequency separation
            downsample_size: Downsample size for faster FFT computation
        """
        super().__init__()
        self.cutoff_ratio = cutoff_ratio
        self.downsample_size = downsample_size

        # Mask cache
        self._mask_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _get_high_freq_mask(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Get or create cached high-frequency mask for rfft2 output."""
        W_rfft = W // 2 + 1
        cache_key = (H, W_rfft)

        if cache_key not in self._mask_cache:
            radius = int(min(H, W) * self.cutoff_ratio)

            y = torch.arange(H, device=device).view(-1, 1)
            x = torch.arange(W_rfft, device=device).view(1, -1)

            # Low frequency mask (near DC component)
            mask_low = (x**2 + y**2 <= radius**2).float()
            mask_low = mask_low + ((x**2 + (H - y)**2 <= radius**2) & (y > 0)).float()
            mask_low = torch.clamp(mask_low, 0, 1)

            # High frequency mask = 1 - low
            mask_high = 1.0 - mask_low

            self._mask_cache[cache_key] = mask_high

        return self._mask_cache[cache_key]

    def _compute_high_freq_ratio(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute high-frequency ratio of an image.

        Args:
            image: shape (C, H, W), values [0, 255]

        Returns:
            high_freq_ratio: scalar tensor
        """
        C, H, W = image.shape

        # Downsample for faster FFT
        if min(H, W) > self.downsample_size:
            scale = self.downsample_size / min(H, W)
            new_H, new_W = int(H * scale), int(W * scale)
            new_H = new_H - (new_H % 2)
            new_W = new_W - (new_W % 2)
            image = F.interpolate(
                image.unsqueeze(0),
                size=(new_H, new_W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            H, W = new_H, new_W

        # FFT
        f = torch.fft.rfft2(image)  # (C, H, W//2+1)
        f_magnitude = torch.abs(f)

        # Get high frequency mask
        mask_high = self._get_high_freq_mask(H, W, image.device)

        # Compute energies
        high_energy = (f_magnitude * mask_high).sum()
        total_energy = f_magnitude.sum() + 1e-8

        # High frequency ratio
        high_ratio = high_energy / total_energy

        return high_ratio

    def forward(
        self,
        stretched_image: torch.Tensor,
        original_image: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute relative sharpness loss.

        Goal: Maximize sharpness improvement over original
        Loss = -log(high_ratio_stretched / high_ratio_original)

        Args:
            stretched_image: shape (C, H, W), stretched image
            original_image: shape (C, H, W), original image

        Returns:
            Dictionary with 'total', 'improvement_ratio' losses
        """
        # Compute high-frequency ratios
        high_ratio_original = self._compute_high_freq_ratio(original_image)
        high_ratio_stretched = self._compute_high_freq_ratio(stretched_image)

        # Relative improvement
        # high_ratio_stretched > high_ratio_original → improvement > 1 → log > 0 → loss < 0 (reward!)
        # high_ratio_stretched < high_ratio_original → improvement < 1 → log < 0 → loss > 0 (penalty!)
        improvement = high_ratio_stretched / (high_ratio_original + 1e-8)
        loss = -torch.log(improvement + 1e-8)

        return {
            'total': loss,
            'improvement_ratio': improvement,
            'high_ratio_original': high_ratio_original,
            'high_ratio_stretched': high_ratio_stretched,
        }


class SimpleAdaptiveNormalizer(nn.Module):
    """
    Full pipeline with learnable controller (완전 Not Heuristic!):

    1. Analyze degradation: A(x) → s (frequency-based measurement)
    2. Controller: g_φ(s) → (clip_low, clip_high, strength)
    3. Stretch: stretch(x, params) → x̃

    Training:
    - θ (detector) is frozen
    - φ (controller) is updated using relative sharpness loss
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        temperature: float = 0.01,
        device: str = 'cuda',
        enabled: bool = True
    ):
        super().__init__()

        self.analyzer = SimpleDegradationAnalyzer(device=device)
        self.controller = StretchingController(hidden_dim=hidden_dim)
        self.stretcher = DifferentiableHistogramStretcher(temperature=temperature)

        self.enabled = enabled
        self.device = device

        # Move controller to device
        self.controller = self.controller.to(device)

        # Statistics tracking
        self.stats = {
            'degradation_scores': [],
            'high_freq_ratios': [],
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
            Optionally: (stretched, params, degradation_score)
        """
        if not self.enabled:
            if return_params:
                return x, None, None
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
        # 1. Analyze degradation (frequency-based)
        with torch.no_grad():
            degradation_score, high_freq_ratio = self.analyzer.compute_degradation(image)

        # Convert to tensor
        s = torch.tensor([degradation_score], device=self.device, dtype=torch.float32)

        # 2. Get controller parameters (gradient flows here)
        clip_low, clip_high, strength = self.controller(s)

        # 3. Apply stretching (gradient flows here)
        stretched = self.stretcher(
            image,
            clip_low[0],
            clip_high[0],
            strength[0]
        )

        # Track statistics
        self.stats['degradation_scores'].append(degradation_score)
        self.stats['high_freq_ratios'].append(high_freq_ratio)
        self.stats['params'].append((
            clip_low[0].item(),
            clip_high[0].item(),
            strength[0].item()
        ))

        if return_params:
            params = (clip_low, clip_high, strength)
            return stretched, params, s

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
        all_scores = []

        for i in range(B):
            if return_params:
                stretched, params, s = self._process_single(
                    images[i], return_params=True
                )
                all_params.append(params)
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
            stacked_scores = torch.cat(all_scores, dim=0)
            return batch_result, stacked_params, stacked_scores

        return batch_result

    def get_stats(self) -> Optional[Dict]:
        """Get statistics of processed images."""
        if len(self.stats['degradation_scores']) == 0:
            return None

        params_array = np.array(self.stats['params'])

        return {
            'mean_degradation': np.mean(self.stats['degradation_scores']),
            'std_degradation': np.std(self.stats['degradation_scores']),
            'mean_high_freq_ratio': np.mean(self.stats['high_freq_ratios']),
            'mean_clip_low': np.mean(params_array[:, 0]),
            'mean_clip_high': np.mean(params_array[:, 1]),
            'mean_strength': np.mean(params_array[:, 2]),
            'num_processed': len(self.stats['degradation_scores'])
        }

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'degradation_scores': [],
            'high_freq_ratios': [],
            'params': [],
        }


# =============================================================================
# AdaptationEngine Integration for TTA Pipeline
# =============================================================================

@dataclass
class FrequencyTTAConfig(AdaptationConfig):
    """Configuration for FrequencyTTAEngine (Relative Sharpness - 완전 Not Heuristic!)."""
    adaptation_name: str = "FrequencyTTAEngine_v3"

    # Controller settings
    hidden_dim: int = 16
    temperature: float = 0.01

    # Frequency loss settings
    cutoff_ratio: float = 0.1
    downsample_size: int = 64

    # Optimizer settings
    optim: Literal["SGD", "Adam"] = "SGD"
    adapt_lr: float = 1e-4


class FrequencyTTAEngine(AdaptationEngine):
    """
    TTA Engine with Relative Sharpness Loss (완전 Not Heuristic!).

    핵심 개선:
    - Degradation score를 실제 측정 (frequency-based)
    - Relative sharpness loss: 원본 대비 개선 효과 측정
    - NO hand-crafted prior
    - NO heuristic threshold
    - NO parameter regularization
    - Self-regulating: 자동으로 optimal strength 찾음

    Pipeline:
    1. s = compute_degradation(x)  ← 실제 측정!
    2. Controller: g_φ(s) → (clip_low, clip_high, strength)
    3. Stretch: x̃ = stretch(x, params)
    4. Loss: -log(sharpness(x̃) / sharpness(x))  ← Relative improvement!
    5. Update φ
    6. Forward: y = f_θ(x̃)

    Loss:
    L = -log(high_ratio_stretched / high_ratio_original)
    - Improvement > 1 → loss < 0 (reward!)
    - Improvement < 1 → loss > 0 (penalty!)
    - Improvement = 1 → loss = 0 (neutral)
    """

    model_name: str = "FrequencyTTAEngine_v3"

    def __init__(self, base_model: BaseModel, config: FrequencyTTAConfig):
        # Initialize parent
        super().__init__(base_model, config)

        self.config = config

        # Initialize components
        self.analyzer = SimpleDegradationAnalyzer(device='cuda')

        self.controller = StretchingController(
            hidden_dim=config.hidden_dim
        )

        self.stretcher = DifferentiableHistogramStretcher(
            temperature=config.temperature
        )

        # Relative sharpness loss (완전 Not Heuristic!)
        self.loss_fn = RelativeSharpnessLoss(
            cutoff_ratio=config.cutoff_ratio,
            downsample_size=config.downsample_size,
        )

        # Statistics tracking
        self.stats = {
            'degradation_scores': [],
            'params': [],
            'losses': [],
            'improvements': [],
        }

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.controller = self.controller.to(self._device)
        self.stretcher = self.stretcher.to(self._device)
        self.analyzer.device = self._device
        return result

    @property
    def names(self):
        """Forward names from base model for Ultralytics compatibility."""
        return self.base_model.names

    def online_parameters(self):
        """Only controller parameters are updated during TTA."""
        return self.controller.parameters()

    @property
    def optimizer(self):
        if self._optimizer is None:
            if self.config.optim == "Adam":
                self._optimizer = optim.Adam(
                    self.online_parameters(),
                    lr=self.config.adapt_lr
                )
            else:
                self._optimizer = optim.SGD(
                    self.online_parameters(),
                    lr=self.config.adapt_lr
                )
        return self._optimizer

    def _normalize_image(self, img: torch.Tensor, return_params: bool = False):
        """Apply normalization to a single image."""
        # Ensure device
        img = img.to(self._device)

        # Ensure [0-255] range
        original_scale = img.max() <= 1.0
        if original_scale:
            img = img * 255.0

        # 1. Measure degradation (frequency-based)
        with torch.no_grad():
            degradation_score, _ = self.analyzer.compute_degradation(img)

        s = torch.tensor([degradation_score], device=self._device, dtype=torch.float32)

        # 2. Get controller parameters (gradient flows here)
        clip_low, clip_high, strength = self.controller(s)

        # 3. Apply stretching
        stretched = self.stretcher(img, clip_low[0], clip_high[0], strength[0])

        # Keep in [0-255] for loss computation, will convert later
        if return_params:
            params = (clip_low, clip_high, strength)
            return stretched, params, s, original_scale, img.clone()  # Return original for loss

        # Restore original scale for inference
        if original_scale:
            stretched = stretched / 255.0

        return stretched

    def _normalize_batch(self, imgs: torch.Tensor, return_params: bool = False):
        """Apply normalization to a batch of images (B, C, H, W)."""
        imgs = imgs.to(self._device)

        # Ensure [0-255] range
        original_scale = imgs.max() <= 1.0
        if original_scale:
            imgs = imgs * 255.0

        B = imgs.shape[0]
        all_stretched = []
        all_params = []
        all_s = []
        all_originals = []

        for i in range(B):
            img = imgs[i]  # (C, H, W)

            # 1. Measure degradation
            with torch.no_grad():
                degradation_score, _ = self.analyzer.compute_degradation(img)

            s = torch.tensor([degradation_score], device=self._device, dtype=torch.float32)

            # 2. Get controller parameters
            clip_low, clip_high, strength = self.controller(s)

            # 3. Apply stretching
            stretched = self.stretcher(img, clip_low[0], clip_high[0], strength[0])

            all_stretched.append(stretched)
            all_params.append((clip_low, clip_high, strength))
            all_s.append(s)
            all_originals.append(img.clone())

        batch_stretched = torch.stack(all_stretched, dim=0)

        if return_params:
            batch_originals = torch.stack(all_originals, dim=0)
            return batch_stretched, all_params, all_s, original_scale, batch_originals

        # Restore original scale for inference
        if original_scale:
            batch_stretched = batch_stretched / 255.0

        return batch_stretched

    def forward(self, batched_inputs):
        """
        Forward pass with TTA.

        핵심: Relative sharpness loss로 self-regulating!

        During adaptation (self.adapting=True):
        1. Apply stretching (with gradient)
        2. Compute relative sharpness loss (stretched vs original)
        3. Update controller
        4. Forward through detector

        During inference (self.adapting=False):
        Just apply stretching and forward.
        """
        # Handle tensor input (from validator.py: model(batch['img']))
        if isinstance(batched_inputs, torch.Tensor):
            return self._forward_tensor(batched_inputs)

        # Handle list of dicts input (original format)
        return self._forward_dict_list(batched_inputs)

    def _forward_tensor(self, imgs: torch.Tensor):
        """Handle tensor input (B, C, H, W)."""
        if not self.adapting:
            # Simple inference mode
            normalized = self._normalize_batch(imgs)
            return self.base_model(normalized)

        # Adaptation mode
        batch_stretched, all_params, all_s, original_scale, batch_originals = \
            self._normalize_batch(imgs, return_params=True)

        # Track stats
        for i, (params, s) in enumerate(zip(all_params, all_s)):
            self.stats['degradation_scores'].append(s.item())
            self.stats['params'].append((
                params[0].item(), params[1].item(), params[2].item()
            ))

        # Compute relative sharpness loss
        total_loss = torch.tensor(0.0, device=self._device)
        all_losses = {'total': 0.0, 'improvement_ratio': 0.0}

        for i in range(batch_stretched.shape[0]):
            stretched = batch_stretched[i]  # (C, H, W)
            original = batch_originals[i]   # (C, H, W)

            # Compute loss
            losses = self.loss_fn(stretched, original)
            total_loss = total_loss + losses['total']

            all_losses['total'] += losses['total'].item()
            all_losses['improvement_ratio'] += losses['improvement_ratio'].item()

            # Track improvement
            self.stats['improvements'].append(losses['improvement_ratio'].item())

        # Average
        n = batch_stretched.shape[0]
        total_loss = total_loss / n
        all_losses = {k: v / n for k, v in all_losses.items()}

        # Backward and update controller
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Track loss
        self.stats['losses'].append(all_losses)

        # Prepare input for model (restore scale if needed)
        if original_scale:
            model_input = batch_stretched.detach() / 255.0
        else:
            model_input = batch_stretched.detach()

        return self.base_model(model_input)

    def _forward_dict_list(self, batched_inputs):
        """Handle list of dicts input (original format)."""
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

        normalized_inputs = []
        all_stretched = []
        all_originals = []
        all_params = []

        # 1. Apply stretching with gradient
        for input_dict in batched_inputs:
            if 'image' in input_dict:
                stretched, params, s, original_scale, original = self._normalize_image(
                    input_dict['image'], return_params=True
                )

                all_stretched.append(stretched)
                all_originals.append(original)
                all_params.append(params)

                # Track stats
                self.stats['degradation_scores'].append(s.item())
                self.stats['params'].append((
                    params[0].item(), params[1].item(), params[2].item()
                ))

                # Prepare input for model (restore scale if needed)
                new_input = input_dict.copy()
                if original_scale:
                    new_input['image'] = stretched / 255.0
                else:
                    new_input['image'] = stretched
                normalized_inputs.append(new_input)
            else:
                normalized_inputs.append(input_dict)

        # 2. Compute relative sharpness loss
        if all_stretched:
            total_loss = torch.tensor(0.0, device=self._device)
            all_losses = {'total': 0.0, 'improvement_ratio': 0.0}

            for i, (stretched, original) in enumerate(zip(all_stretched, all_originals)):
                losses = self.loss_fn(stretched, original)
                total_loss = total_loss + losses['total']

                all_losses['total'] += losses['total'].item()
                all_losses['improvement_ratio'] += losses['improvement_ratio'].item()

                # Track improvement
                self.stats['improvements'].append(losses['improvement_ratio'].item())

            # Average
            n = len(all_stretched)
            total_loss = total_loss / n
            all_losses = {k: v / n for k, v in all_losses.items()}

            # 3. Backward and update controller
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Track loss
            self.stats['losses'].append(all_losses)

        # 4. Forward through model
        # detach stretched images for model forward
        for i, input_dict in enumerate(normalized_inputs):
            if 'image' in input_dict:
                normalized_inputs[i]['image'] = input_dict['image'].detach()

        outputs = self.base_model(normalized_inputs)

        return outputs

    def reset(self):
        """Reset controller to initial state."""
        super().reset()
        # Re-initialize controller
        self.controller = StretchingController(
            hidden_dim=self.config.hidden_dim
        ).to(self._device)
        self._optimizer = None  # Reset optimizer
        # Clear mask cache
        self.loss_fn._mask_cache.clear()
        self.analyzer._mask_cache.clear()
        self.stats = {
            'degradation_scores': [],
            'params': [],
            'losses': [],
            'improvements': [],
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

        if self.stats['improvements']:
            result['mean_improvement'] = np.mean(self.stats['improvements'])
            result['std_improvement'] = np.std(self.stats['improvements'])

        return result


# =============================================================================
# Backward compatibility aliases
# =============================================================================

# For drop-in replacement testing
FrequencyAdaptationConfig = FrequencyTTAConfig
FrequencyAdaptationEngine = FrequencyTTAEngine

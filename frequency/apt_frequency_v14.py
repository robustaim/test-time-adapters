"""
v14: Frequency-based self-supervised loss (TTT-Linear style)

v12에서 텐서 입력 처리 추가 (validator.py 호환성).

v11에서 forward 2번 문제를 해결.
Detection model의 entropy 비교 대신 frequency sharpness loss 사용.

핵심 아이디어:
- "선명한 이미지 = high-frequency 성분이 많다"
- Stretched 이미지의 high-freq가 증가하도록 loss 설계
- Detection model forward 없이 loss 계산 가능 → 1번만 forward

Pipeline:
1. s = 0.5 (fixed)
2. Controller: g_φ(s) → (clip_low, clip_high, strength)
3. Stretch: x̃ = stretch(x, params)
4. Loss: L_sharp(x̃) + λ_reg * L_reg(params, prior)  ← model forward 불필요!
5. Forward: y = f_θ(x̃)  ← 1번만!

TTT-Linear와 유사:
- Main task (detection)의 output 없이 auxiliary task (frequency)로 업데이트
- Self-supervised signal로 adaptation
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
    Fixed degradation analyzer - always returns 0.5.
    Baseline for ablation study.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def compute_degradation(self, image: torch.Tensor) -> Tuple[float, float]:
        """
        Always return fixed degradation score of 0.5.

        Args:
            image: torch.Tensor of shape (C, H, W) or (H, W, C), values [0-255]

        Returns:
            degradation_score: 0.5 (fixed)
            statistic_value: 0.5 (fixed)
        """
        return 0.5, 0.5


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


class FrequencySharpnessLoss(nn.Module):
    """
    Frequency-based self-supervised loss for TTA.

    TTT-Linear 스타일: Detection model forward 없이 auxiliary task로 업데이트.

    L = L_sharp + λ_reg * L_reg

    - L_sharp: Frequency sharpness loss (maximize high-freq energy)
    - L_reg: Regularization toward hand-crafted prior
    """

    def __init__(
        self,
        lambda_reg: float = 1.0,
        cutoff_ratio: float = 0.1,
        downsample_size: int = 64,
    ):
        """
        Args:
            lambda_reg: Weight for regularization loss
            cutoff_ratio: Cutoff ratio for low/high frequency separation
            downsample_size: Downsample size for faster FFT computation
        """
        super().__init__()
        self.lambda_reg = lambda_reg
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

    def compute_sharpness_loss(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency sharpness loss.

        Goal: Maximize high-frequency energy ratio
        Loss = -log(high_energy / total_energy)

        Args:
            image: shape (C, H, W), values [0, 255]

        Returns:
            Sharpness loss (lower = sharper image)
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

        # Loss: negative log ratio (minimize this = maximize high freq ratio)
        high_ratio = high_energy / total_energy
        loss = -torch.log(high_ratio + 1e-8)

        return loss

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

    def forward(
        self,
        stretched_image: torch.Tensor,
        params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        prior_params: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        NO detection model forward needed!

        Args:
            stretched_image: shape (C, H, W), stretched image
            params: (clip_low, clip_high, strength) from controller
            prior_params: (clip_low, clip_high, strength) from prior

        Returns:
            Dictionary with 'total', 'sharp', 'reg' losses
        """
        l_sharp = self.compute_sharpness_loss(stretched_image)
        l_reg = self.regularization_loss(params, prior_params)

        total = l_sharp + self.lambda_reg * l_reg

        return {
            'total': total,
            'sharp': l_sharp,
            'reg': l_reg,
        }


class SimpleAdaptiveNormalizer(nn.Module):
    """
    Full pipeline with learnable controller (NO FFT):

    1. Analyze degradation: A(x) → s (fixed 0.5)
    2. Controller: g_φ(s) → (clip_low, clip_high, strength)
    3. Stretch: stretch(x, params) → x̃

    Training:
    - θ (detector) is frozen
    - φ (controller) is updated
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
            'stat_values': [],
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
            degradation_score, stat_value = self.analyzer.compute_degradation(image)

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
        self.stats['stat_values'].append(stat_value)
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
            'mean_stat_value': np.mean(self.stats['stat_values']),
            'mean_clip_low': np.mean(params_array[:, 0]),
            'mean_clip_high': np.mean(params_array[:, 1]),
            'mean_strength': np.mean(params_array[:, 2]),
            'num_processed': len(self.stats['degradation_scores'])
        }

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'degradation_scores': [],
            'stat_values': [],
            'params': [],
        }


# =============================================================================
# AdaptationEngine Integration for TTA Pipeline
# =============================================================================

@dataclass
class FrequencyTTAConfig(AdaptationConfig):
    """Configuration for FrequencyTTAEngine (TTT-Linear style, 1x forward)."""
    adaptation_name: str = "FrequencyTTAEngine"

    # Controller settings
    hidden_dim: int = 16
    temperature: float = 0.01

    # Frequency loss settings
    lambda_reg: float = 1.0
    cutoff_ratio: float = 0.1
    downsample_size: int = 64

    # Optimizer settings
    optim: Literal["SGD", "Adam"] = "SGD"
    adapt_lr: float = 1e-4


class FrequencyTTAEngine(AdaptationEngine):
    """
    TTA Engine with frequency-based self-supervised loss (TTT-Linear style).

    핵심: Detection model forward 1번만!
    - 기존 (v11): forward 2번 (orig entropy + stretched entropy)
    - v12: forward 1번 (frequency loss는 model 없이 계산)

    Pipeline:
    1. s = 0.5 (fixed)
    2. Controller: g_φ(s) → (clip_low, clip_high, strength)
    3. Stretch: x̃ = stretch(x, params)
    4. Loss: L_sharp(x̃) + λ_reg * L_reg  ← NO model forward!
    5. Update φ
    6. Forward: y = f_θ(x̃)  ← 1번만!

    Loss:
    L = L_sharp + λ_reg * L_reg
    - L_sharp: -log(high_freq_energy / total_energy)
    - L_reg: ||params - prior||²
    """

    model_name: str = "FrequencyTTAEngine"

    def __init__(self, base_model: BaseModel, config: FrequencyTTAConfig):
        # Initialize parent
        super().__init__(base_model, config)

        self.config = config

        # Initialize components
        self.analyzer = SimpleDegradationAnalyzer(device='cuda')

        self.controller = StretchingController(
            hidden_dim=config.hidden_dim,
            init_to_prior=True
        )

        self.stretcher = DifferentiableHistogramStretcher(
            temperature=config.temperature
        )

        # Frequency-based loss (NO model forward needed!)
        self.loss_fn = FrequencySharpnessLoss(
            lambda_reg=config.lambda_reg,
            cutoff_ratio=config.cutoff_ratio,
            downsample_size=config.downsample_size,
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

        # 1. Fixed degradation score
        degradation_score = 0.5
        s = torch.tensor([degradation_score], device=self._device, dtype=torch.float32)

        # 2. Get controller parameters (gradient flows here)
        clip_low, clip_high, strength = self.controller(s)

        # 3. Get prior for regularization
        prior_params = self.controller.get_prior(s)
        prior_params = tuple(p.to(self._device) for p in prior_params)

        # 4. Apply stretching
        stretched = self.stretcher(img, clip_low[0], clip_high[0], strength[0])

        # Keep in [0-255] for loss computation, will convert later
        if return_params:
            params = (clip_low, clip_high, strength)
            return stretched, params, prior_params, s, original_scale

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
        all_priors = []
        all_s = []

        for i in range(B):
            img = imgs[i]  # (C, H, W)

            # 1. Fixed degradation score
            degradation_score = 0.5
            s = torch.tensor([degradation_score], device=self._device, dtype=torch.float32)

            # 2. Get controller parameters
            clip_low, clip_high, strength = self.controller(s)

            # 3. Get prior
            prior_params = self.controller.get_prior(s)
            prior_params = tuple(p.to(self._device) for p in prior_params)

            # 4. Apply stretching
            stretched = self.stretcher(img, clip_low[0], clip_high[0], strength[0])

            all_stretched.append(stretched)
            all_params.append((clip_low, clip_high, strength))
            all_priors.append(prior_params)
            all_s.append(s)

        batch_stretched = torch.stack(all_stretched, dim=0)

        if return_params:
            return batch_stretched, all_params, all_priors, all_s, original_scale

        # Restore original scale for inference
        if original_scale:
            batch_stretched = batch_stretched / 255.0

        return batch_stretched

    def forward(self, batched_inputs):
        """
        Forward pass with TTA.

        v14 핵심: 텐서 입력과 딕셔너리 리스트 입력 모두 지원!

        During adaptation (self.adapting=True):
        1. Apply stretching (with gradient)
        2. Compute frequency loss (NO model forward!)
        3. Update controller
        4. Forward through detector (1번만!)

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
        batch_stretched, all_params, all_priors, all_s, original_scale = \
            self._normalize_batch(imgs, return_params=True)

        # Track stats
        for i, (params, s) in enumerate(zip(all_params, all_s)):
            self.stats['degradation_scores'].append(s.item())
            self.stats['params'].append((
                params[0].item(), params[1].item(), params[2].item()
            ))

        # Compute frequency loss
        total_loss = torch.tensor(0.0, device=self._device)
        all_losses = {'total': 0.0, 'sharp': 0.0, 'reg': 0.0}

        for i in range(batch_stretched.shape[0]):
            stretched = batch_stretched[i]  # (C, H, W)
            params = all_params[i]
            priors = all_priors[i]

            losses = self.loss_fn(stretched, params, priors)
            total_loss = total_loss + losses['total']

            for k in all_losses:
                all_losses[k] += losses[k].item()

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

        # === Adaptation mode (1x forward!) ===

        normalized_inputs = []
        all_stretched = []
        all_params = []
        all_priors = []

        # 1. Apply stretching with gradient
        for input_dict in batched_inputs:
            if 'image' in input_dict:
                stretched, params, priors, s, original_scale = self._normalize_image(
                    input_dict['image'], return_params=True
                )

                all_stretched.append(stretched)
                all_params.append(params)
                all_priors.append(priors)

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

        # 2. Compute frequency loss (NO model forward needed!)
        if all_stretched:
            total_loss = torch.tensor(0.0, device=self._device)
            all_losses = {'total': 0.0, 'sharp': 0.0, 'reg': 0.0}

            for stretched, params, priors in zip(all_stretched, all_params, all_priors):
                losses = self.loss_fn(stretched, params, priors)
                total_loss = total_loss + losses['total']

                for k in all_losses:
                    all_losses[k] += losses[k].item()

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

        # 4. Forward through model (1번만!)
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
            hidden_dim=self.config.hidden_dim,
            init_to_prior=True
        ).to(self._device)
        self._optimizer = None  # Reset optimizer
        # Clear mask cache
        self.loss_fn._mask_cache.clear()
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
# Backward compatibility aliases
# =============================================================================

# For drop-in replacement testing
FrequencyAdaptationConfig = FrequencyTTAConfig
FrequencyAdaptationEngine = FrequencyTTAEngine

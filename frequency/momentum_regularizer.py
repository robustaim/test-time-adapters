"""
Momentum-Based Regularization Module

Inspired by Nested Learning's interpretation of momentum as associative memory.
Replaces hand-crafted priors with learned EMA buffers for parameter regularization.

Author: FrequencyAPT Team
Date: 2026-01-06
"""

import torch
import torch.nn as nn
from typing import Tuple, Literal, Optional


class MomentumRegularizer(nn.Module):
    """
    Momentum-based regularization using EMA buffers.
    
    Theoretical Foundation (from Nested Learning):
    - EMA buffers act as associative memory storing learned parameter distributions
    - Regularization encourages temporal consistency with memory
    - β controls memory decay: high β = long-term memory, low β = short-term adaptation
    
    Mathematical Formulation:
        L_reg = ||params - EMA(params)||²
        
        where EMA_t = β * EMA_{t-1} + (1-β) * params_t
        
    This replaces the old heuristic-based formulation:
        L_reg = ||params - h(s)||²  where h(s) is hand-crafted
    
    Args:
        beta: EMA decay coefficient ∈ [0, 1]
              - 0.9: Standard momentum (similar to BatchNorm, Adam)
              - 0.99: Longer memory
              - 0.5: Faster adaptation
        init_mode: Initialization strategy for EMA buffers
                   - 'zero': Start from zero (default)
                   - 'uniform': Start from mid-range values
                   - 'from_prior': Warm start from old hand-crafted prior (for comparison)
        warmup_steps: Number of steps to warm up EMA buffers before computing loss
                      During warmup, L_reg is scaled down to avoid penalizing random init
    """
    
    def __init__(
        self,
        beta: float = 0.9,
        init_mode: Literal['zero', 'uniform', 'from_prior'] = 'zero',
        warmup_steps: int = 10
    ):
        super().__init__()
        
        self.beta = beta
        self.warmup_steps = warmup_steps
        
        # Initialize EMA buffers
        if init_mode == 'zero':
            # Start from zero - let network learn from scratch
            init_low, init_high, init_strength = 0.0, 0.0, 0.0
        elif init_mode == 'uniform':
            # Start from mid-range values
            init_low, init_high, init_strength = 5.0, 95.0, 0.5
        elif init_mode == 'from_prior':
            # Warm start from old hand-crafted prior (s=0.5)
            init_low, init_high, init_strength = 3.0, 97.0, 0.6
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")
        
        # Register as buffers (will be saved/loaded with model)
        self.register_buffer('ema_clip_low', torch.tensor(init_low))
        self.register_buffer('ema_clip_high', torch.tensor(init_high))
        self.register_buffer('ema_strength', torch.tensor(init_strength))
        
        # Step counter for warmup
        self.register_buffer('step_count', torch.tensor(0))
    
    def update_memory(
        self, 
        clip_low: torch.Tensor, 
        clip_high: torch.Tensor, 
        strength: torch.Tensor
    ) -> None:
        """
        Update EMA buffers with new parameters.
        
        EMA update rule:
            EMA_t = β * EMA_{t-1} + (1-β) * param_t
        
        Args:
            clip_low: Current clip_low parameter, shape (B,) or ()
            clip_high: Current clip_high parameter, shape (B,) or ()
            strength: Current strength parameter, shape (B,) or ()
        """
        with torch.no_grad():
            # Take mean if batch dimension exists
            clip_low_mean = clip_low.mean() if clip_low.numel() > 1 else clip_low
            clip_high_mean = clip_high.mean() if clip_high.numel() > 1 else clip_high
            strength_mean = strength.mean() if strength.numel() > 1 else strength
            
            # EMA update
            self.ema_clip_low = self.beta * self.ema_clip_low + (1 - self.beta) * clip_low_mean
            self.ema_clip_high = self.beta * self.ema_clip_high + (1 - self.beta) * clip_high_mean
            self.ema_strength = self.beta * self.ema_strength + (1 - self.beta) * strength_mean
            
            # Increment step counter
            self.step_count += 1
    
    def compute_regularization(
        self,
        clip_low: torch.Tensor,
        clip_high: torch.Tensor,
        strength: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute momentum-based regularization loss.
        
        L_reg = ||params - EMA(params)||²
        
        During warmup, loss is scaled by (step / warmup_steps) to avoid
        penalizing random initialization.
        
        Args:
            clip_low: Current clip_low parameter, shape (B,) or ()
            clip_high: Current clip_high parameter, shape (B,) or ()
            strength: Current strength parameter, shape (B,) or ()
            
        Returns:
            Regularization loss (scalar)
        """
        # Take mean if batch dimension exists
        clip_low_mean = clip_low.mean() if clip_low.numel() > 1 else clip_low
        clip_high_mean = clip_high.mean() if clip_high.numel() > 1 else clip_high
        strength_mean = strength.mean() if strength.numel() > 1 else strength
        
        # Compute squared differences
        loss = (
            (clip_low_mean - self.ema_clip_low) ** 2 +
            (clip_high_mean - self.ema_clip_high) ** 2 +
            (strength_mean - self.ema_strength) ** 2 * 100  # Scale strength loss (same as old)
        )
        
        # Apply warmup scaling
        if self.warmup_steps > 0 and self.step_count < self.warmup_steps:
            warmup_scale = self.step_count.float() / self.warmup_steps
            loss = loss * warmup_scale
        
        return loss
    
    def get_memory_state(self) -> Tuple[float, float, float]:
        """
        Get current EMA buffer values (for logging/debugging).
        
        Returns:
            (ema_clip_low, ema_clip_high, ema_strength) as Python floats
        """
        return (
            self.ema_clip_low.item(),
            self.ema_clip_high.item(),
            self.ema_strength.item()
        )
    
    def reset_memory(self) -> None:
        """
        Reset EMA buffers to initial state.
        
        Useful for testing or when switching domains.
        """
        self.ema_clip_low.zero_()
        self.ema_clip_high.zero_()
        self.ema_strength.zero_()
        self.step_count.zero_()
    
    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f'beta={self.beta}, warmup_steps={self.warmup_steps}, step_count={self.step_count.item()}'

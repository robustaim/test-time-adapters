from dataclasses import dataclass
from typing import Literal

from ....base import AdaptationConfig


@dataclass
class APTConfig(AdaptationConfig):
    """
    Configuration for APT: Adaptive Plugin using Temporal Consistency
    
    This method adapts object detection models at test-time by leveraging
    temporal consistency between consecutive frames using Kalman filter tracking.

    Key concept: Uses motion predictions from Kalman filter as self-supervised
    signals (NOT pseudo-labels from model predictions).
    """
    adaptation_name: str = "APT"

    # Optimization settings
    optim: Literal["SGD", "Adam", "AdamW"] = "SGD"
    adapt_lr: float = 1e-5
    backbone_lr: float = 1e-6
    head_lr: float = 1e-6
    momentum: float = 0.9
    weight_decay: float = 0.0

    # Temporal consistency settings
    max_age: int = 3
    min_hits: int = 1
    iou_threshold: float = 0.8

    # Loss settings
    loss_type: Literal["l1", "l2", "smooth_l1", "giou"] = "smooth_l1"
    loss_weight: float = 1.0
    use_confidence_weighting: bool = True

    # Confidence thresholding
    conf_threshold: float = 0.7
    min_confidence_for_update: float = 0.3

    # Update strategy (Idea B: Expand adaptation scope)
    update_backbone: bool = False
    update_head: bool = False
    update_bn: bool = True
    update_bn_stats: bool = False  # Idea D
    update_fpn_last_layer: bool = False
    update_box_regressor_last_layer: bool = False
    update_conv_before_bn: bool = False  # Idea B
    update_mlp_after_norm: bool = False  # Idea B

    # Memory settings
    buffer_size: int = 500

    # Loss stabilization
    loss_ema_decay: float = 0.9
    
    # Loss history tracking (Idea A)
    enable_loss_history: bool = False
    loss_history_size: int = 100  # Increased default
    min_history_for_spike_detection: int = 30  # Increased default

    # Domain change detection (Idea A: Loss spike based optimizer reset)
    enable_domain_change_reset: bool = False
    domain_change_loss_spike_threshold: float = 5.0  # More conservative default
    domain_change_loss_spike_std_multiplier: float = 5.0  # More conservative default
    
    # Loss threshold for update skip (Idea A: Skip updates with extremely high loss)
    enable_loss_threshold_skip: bool = False
    loss_threshold_skip_value: float = 20.0  # Higher default
    loss_threshold_skip_relative: float = 10.0  # Higher default

    # Gradient scaling (Idea C: Scale gradients inversely with loss magnitude)
    enable_gradient_scaling: bool = False
    gradient_scaling_mode: Literal["inverse_loss", "inverse_sqrt", "adaptive"] = "adaptive"
    gradient_scaling_min: float = 0.1
    gradient_scaling_max: float = 5.0
    gradient_scaling_base_loss: float = 1.0

    @classmethod
    def baseline(cls):
        """Baseline configuration (original APT with SGD)"""
        return cls(
            adaptation_name="APT_Baseline",
            optim="SGD",
            update_bn=True,
            # All improvements disabled (default)
        )

    @classmethod
    def with_idea_A(cls):
        """Baseline + Idea A: Loss-based adaptation control (CONSERVATIVE)"""
        return cls(
            adaptation_name="APT_IdeaA",
            optim="AdamW",  # A requires better optimizer
            adapt_lr=1e-5,
            weight_decay=1e-4,
            update_bn=True,
            
            # Idea A: Enable all loss control features (CONSERVATIVE settings)
            enable_loss_history=True,
            loss_history_size=100,  # Increased for more stable statistics
            min_history_for_spike_detection=30,  # Wait longer before detecting spikes
            enable_domain_change_reset=True,
            domain_change_loss_spike_threshold=5.0,  # Much higher: 5x mean instead of 2x
            domain_change_loss_spike_std_multiplier=5.0,  # Higher: 5 std instead of 3
            enable_loss_threshold_skip=True,
            loss_threshold_skip_value=20.0,  # Higher absolute threshold
            loss_threshold_skip_relative=10.0,  # Higher relative threshold
            
            # Other ideas disabled
            update_conv_before_bn=False,
            update_mlp_after_norm=False,
            enable_gradient_scaling=False,
            update_bn_stats=False,
        )

    @classmethod
    def with_idea_A_aggressive(cls):
        """Baseline + Idea A: Loss-based adaptation control (AGGRESSIVE)"""
        return cls(
            adaptation_name="APT_IdeaA_Aggressive",
            optim="AdamW",
            adapt_lr=1e-5,
            weight_decay=1e-4,
            update_bn=True,
            
            # Idea A: Aggressive settings for severe domain shifts
            enable_loss_history=True,
            loss_history_size=50,
            min_history_for_spike_detection=10,
            enable_domain_change_reset=True,
            domain_change_loss_spike_threshold=2.5,  # Lower: more sensitive
            domain_change_loss_spike_std_multiplier=3.5,
            enable_loss_threshold_skip=True,
            loss_threshold_skip_value=15.0,
            loss_threshold_skip_relative=8.0,
            
            # Other ideas disabled
            update_conv_before_bn=False,
            update_mlp_after_norm=False,
            enable_gradient_scaling=False,
            update_bn_stats=False,
        )

    @classmethod
    def with_idea_A_moderate(cls):
        """Baseline + Idea A: Loss-based adaptation control (MODERATE)"""
        return cls(
            adaptation_name="APT_IdeaA_Moderate",
            optim="AdamW",
            adapt_lr=1e-5,
            weight_decay=1e-4,
            update_bn=True,
            
            # Idea A: Moderate settings - balanced
            enable_loss_history=True,
            loss_history_size=75,
            min_history_for_spike_detection=20,
            enable_domain_change_reset=True,
            domain_change_loss_spike_threshold=3.5,  # Moderate
            domain_change_loss_spike_std_multiplier=4.0,
            enable_loss_threshold_skip=True,
            loss_threshold_skip_value=15.0,
            loss_threshold_skip_relative=8.0,
            
            # Other ideas disabled
            update_conv_before_bn=False,
            update_mlp_after_norm=False,
            enable_gradient_scaling=False,
            update_bn_stats=False,
        )

    @classmethod
    def with_idea_A_skiponly(cls):
        """Baseline + Idea A: Only skip mechanism (no reset)"""
        return cls(
            adaptation_name="APT_IdeaA_SkipOnly",
            optim="AdamW",
            adapt_lr=1e-5,
            weight_decay=1e-4,
            update_bn=True,
            
            # Idea A: Only use skip mechanism, disable reset
            enable_loss_history=True,
            loss_history_size=100,
            min_history_for_spike_detection=30,
            enable_domain_change_reset=False,  # Disabled
            enable_loss_threshold_skip=True,
            loss_threshold_skip_value=20.0,
            loss_threshold_skip_relative=10.0,
            
            # Other ideas disabled
            update_conv_before_bn=False,
            update_mlp_after_norm=False,
            enable_gradient_scaling=False,
            update_bn_stats=False,
        )

    @classmethod
    def with_idea_B(cls):
        """Baseline + Idea B: Extended adaptation scope"""
        return cls(
            adaptation_name="APT_IdeaB",
            optim="SGD",
            update_bn=True,
            
            # Idea B: Enable extended parameter adaptation
            update_conv_before_bn=True,
            update_mlp_after_norm=True,
            
            # Other ideas disabled
            enable_loss_history=False,
            enable_domain_change_reset=False,
            enable_loss_threshold_skip=False,
            enable_gradient_scaling=False,
            update_bn_stats=False,
        )

    @classmethod
    def with_idea_C(cls):
        """Baseline + Idea C: Gradient scaling"""
        return cls(
            adaptation_name="APT_IdeaC",
            optim="SGD",
            update_bn=True,
            
            # Idea C: Enable gradient scaling
            enable_gradient_scaling=True,
            gradient_scaling_mode="adaptive",
            gradient_scaling_min=0.1,
            gradient_scaling_max=5.0,
            gradient_scaling_base_loss=1.0,
            
            # Other ideas disabled
            enable_loss_history=False,
            enable_domain_change_reset=False,
            enable_loss_threshold_skip=False,
            update_conv_before_bn=False,
            update_mlp_after_norm=False,
            update_bn_stats=False,
        )

    @classmethod
    def with_idea_D(cls):
        """Baseline + Idea D: BN statistics update"""
        return cls(
            adaptation_name="APT_IdeaD",
            optim="SGD",
            update_bn=True,
            
            # Idea D: Enable BN statistics update
            update_bn_stats=True,
            
            # Other ideas disabled
            enable_loss_history=False,
            enable_domain_change_reset=False,
            enable_loss_threshold_skip=False,
            update_conv_before_bn=False,
            update_mlp_after_norm=False,
            enable_gradient_scaling=False,
        )

    @classmethod
    def with_idea_AB(cls):
        """Baseline + Idea A + B"""
        return cls(
            adaptation_name="APT_IdeaAB",
            optim="AdamW",
            adapt_lr=1e-5,
            weight_decay=1e-4,
            update_bn=True,
            
            # Idea A
            enable_loss_history=True,
            enable_domain_change_reset=True,
            enable_loss_threshold_skip=True,
            
            # Idea B
            update_conv_before_bn=True,
            update_mlp_after_norm=True,
            
            # Others disabled
            enable_gradient_scaling=False,
            update_bn_stats=False,
        )

    @classmethod
    def with_idea_ABC(cls):
        """Baseline + Idea A + B + C"""
        return cls(
            adaptation_name="APT_IdeaABC",
            optim="AdamW",
            adapt_lr=1e-5,
            weight_decay=1e-4,
            update_bn=True,
            
            # Idea A
            enable_loss_history=True,
            enable_domain_change_reset=True,
            enable_loss_threshold_skip=True,
            
            # Idea B
            update_conv_before_bn=True,
            update_mlp_after_norm=True,
            
            # Idea C
            enable_gradient_scaling=True,
            gradient_scaling_mode="adaptive",
            
            # D disabled
            update_bn_stats=False,
        )

    @classmethod
    def with_all_ideas(cls):
        """Baseline + All Ideas (A + B + C + D)"""
        return cls(
            adaptation_name="APT_Full",
            optim="AdamW",
            adapt_lr=1e-5,
            weight_decay=1e-4,
            update_bn=True,
            
            # Idea A: Loss control
            enable_loss_history=True,
            loss_history_size=50,
            min_history_for_spike_detection=10,
            enable_domain_change_reset=True,
            domain_change_loss_spike_threshold=2.0,
            domain_change_loss_spike_std_multiplier=3.0,
            enable_loss_threshold_skip=True,
            loss_threshold_skip_value=10.0,
            loss_threshold_skip_relative=5.0,
            
            # Idea B: Extended scope
            update_conv_before_bn=True,
            update_mlp_after_norm=True,
            
            # Idea C: Gradient scaling
            enable_gradient_scaling=True,
            gradient_scaling_mode="adaptive",
            gradient_scaling_min=0.1,
            gradient_scaling_max=5.0,
            gradient_scaling_base_loss=1.0,
            
            # Idea D: BN stats
            update_bn_stats=True,
        )

    @classmethod
    def get_ablation_configs(cls):
        """Get all configurations for ablation study"""
        return {
            "Baseline": cls.baseline(),
            "IdeaA": cls.with_idea_A(),
            "IdeaB": cls.with_idea_B(),
            "IdeaC": cls.with_idea_C(),
            "IdeaD": cls.with_idea_D(),
            "IdeaAB": cls.with_idea_AB(),
            "IdeaABC": cls.with_idea_ABC(),
            "Full": cls.with_all_ideas(),
        }

    def get_active_ideas(self):
        """Return which ideas are active in this config"""
        active = []
        
        # Check Idea A
        if self.enable_domain_change_reset or self.enable_loss_threshold_skip:
            active.append("A")
        
        # Check Idea B
        if self.update_conv_before_bn or self.update_mlp_after_norm:
            active.append("B")
        
        # Check Idea C
        if self.enable_gradient_scaling:
            active.append("C")
        
        # Check Idea D
        if self.update_bn_stats:
            active.append("D")
        
        return active if active else ["Baseline"]

    def __str__(self):
        active_ideas = "+".join(self.get_active_ideas())
        return f"{self.adaptation_name} (Ideas: {active_ideas})"

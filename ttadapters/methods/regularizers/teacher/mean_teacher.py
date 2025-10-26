"""
Mean Teacher Engine

Framework-agnostic implementation using adapter pattern.
Uses teacher-student framework with EMA updates and pseudo-labeling.
"""

import copy
import random
import warnings
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.optim as optim

from ...base import AdaptationEngine, AdaptationConfig
from ...framework_adapters import FrameworkAdapter, create_adapter


@dataclass
class MeanTeacherConfig(AdaptationConfig):
    """Configuration for Mean Teacher engine"""
    adaptation_name: str = "MeanTeacher"

    # Adaptation settings
    adaptation_layers: str = "backbone+encoder"  # "backbone", "encoder", "backbone+encoder"

    # Optimization settings
    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    optimizer_type: str = "SGD"  # "SGD" or "AdamW"

    # Mean Teacher settings
    ema_alpha: float = 0.99  # EMA decay rate for teacher update
    conf_threshold: float = 0.3  # Confidence threshold for pseudo-labels

    # Regularization
    weight_reg: float = 0.0  # Weight regularization strength

    # Device
    device: str = "cuda"


class MeanTeacherEngine(AdaptationEngine):
    """
    Mean Teacher: Self-training with EMA teacher model

    Uses exponential moving average (EMA) teacher model to generate
    pseudo-labels for student training.

    Framework-agnostic implementation that works with:
    - Detectron2 (Faster R-CNN, Mask R-CNN, etc.)
    - Transformers (RT-DETR, DETR, etc.)
    - Ultralytics (YOLO v8, v11, etc.)

    Note: Augmentation and pseudo-labeling are framework-specific
    and may require custom implementation.
    """

    model_name = "MeanTeacher"

    def __init__(
        self,
        basemodel: nn.Module,
        config: MeanTeacherConfig,
        adapter: Optional[FrameworkAdapter] = None
    ):
        super().__init__(basemodel, config)

        # Auto-detect framework adapter if not provided
        if adapter is None:
            adapter = create_adapter(basemodel)

        self.adapter = adapter
        self.cfg = config
        self.device = torch.device(config.device)

        # Teacher model (EMA of student)
        self.teacher_model = None

        # Optimizer
        self.optimizer = None

        # Initial weights for regularization
        self.init_weights = []

        # Setup
        self._setup()

    def _setup(self):
        """Initialize the Mean Teacher engine"""
        # Move model to device
        self.basemodel.to(self.device)

        # Create teacher model
        self._setup_teacher_model()

        # Setup trainable parameters
        params = self._setup_trainable_params()

        # Setup optimizer
        if self.cfg.optimizer_type == "SGD":
            self.optimizer = optim.SGD(
                params,
                lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer_type == "AdamW":
            self.optimizer = optim.AdamW(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.cfg.optimizer_type}")

        # Store initial weights for regularization
        self.init_weights = [p.clone().detach() for p in params]

    def _setup_teacher_model(self):
        """Create and initialize teacher model"""
        self.teacher_model = copy.deepcopy(self.basemodel)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def _setup_trainable_params(self) -> List[torch.nn.Parameter]:
        """Setup trainable parameters based on adaptation layers"""
        params = []

        # Create layer filter based on adaptation_layers setting
        layer_filter = self._create_layer_filter()

        # Identify normalization layers using adapter
        norm_layers = self.adapter.identify_normalization_layers(
            self.basemodel,
            layer_filter=layer_filter
        )

        # Enable gradient for normalization layer parameters
        for name, module in norm_layers:
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.requires_grad = True
                    params.append(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.requires_grad = True
                    params.append(module.bias)

        # Also adapt Conv2d and Linear biases in selected layers
        for name, module in self.basemodel.named_modules():
            if layer_filter is None or layer_filter(name, module):
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.requires_grad = True
                        params.append(module.bias)

        print(f"MeanTeacherEngine: Training {len(params)} parameters")
        return params

    def _create_layer_filter(self):
        """Create layer filter based on adaptation_layers config"""
        if self.cfg.adaptation_layers == "backbone":
            if hasattr(self.adapter, 'create_layer_filter_backbone'):
                return self.adapter.create_layer_filter_backbone()
            else:
                return lambda name, module: 'backbone' in name.lower()

        elif self.cfg.adaptation_layers == "encoder":
            if hasattr(self.adapter, 'create_layer_filter_encoder'):
                return self.adapter.create_layer_filter_encoder()
            else:
                return lambda name, module: 'encoder' in name.lower()

        elif self.cfg.adaptation_layers == "backbone+encoder":
            if hasattr(self.adapter, 'create_layer_filter_backbone_and_encoder'):
                return self.adapter.create_layer_filter_backbone_and_encoder()
            else:
                return lambda name, module: 'decoder' not in name.lower()

        else:
            return None

    def _update_teacher_ema(self):
        """Update teacher model with EMA"""
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_model.parameters(),
                self.basemodel.parameters()
            ):
                if student_param.requires_grad:
                    teacher_param.data = (
                        self.cfg.ema_alpha * teacher_param.data +
                        (1 - self.cfg.ema_alpha) * student_param.data
                    )

    def forward(self, *args, **kwargs):
        """
        Forward pass with Mean Teacher adaptation

        Args:
            batch: Input batch (framework-specific format)

        Returns:
            Model outputs (from teacher model for stability)
        """
        # Get batch from args or kwargs
        if len(args) > 0:
            batch = args[0]
        else:
            batch = kwargs

        # Get teacher predictions (no gradient)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.adapter.execute_forward(
                self.teacher_model, batch, self.device
            )

        # Train student model
        # Note: This is a simplified version. Full implementation would require:
        # 1. Strong augmentation
        # 2. Pseudo-label generation from teacher outputs
        # 3. Student training with pseudo-labels
        # These are framework-specific and should be implemented in subclasses

        try:
            self.basemodel.train()
            self.optimizer.zero_grad()

            # Student forward pass
            student_outputs = self.adapter.execute_forward(
                self.basemodel, batch, self.device
            )

            # Extract loss if available (framework-dependent)
            loss = self._extract_loss(student_outputs)

            if loss is not None and loss > 0:
                # Add weight regularization if enabled
                if self.cfg.weight_reg > 0:
                    reg_loss = torch.tensor(0.0, device=self.device)
                    for param, init_param in zip(
                        self.optimizer.param_groups[0]['params'],
                        self.init_weights
                    ):
                        reg_loss += torch.mean((param - init_param) ** 2)
                    loss = loss + self.cfg.weight_reg * reg_loss

                # Backward and update
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.basemodel.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Update teacher with EMA
                self._update_teacher_ema()

        except Exception as e:
            warnings.warn(f"Mean Teacher adaptation step failed: {e}")

        # Return teacher outputs for stability
        return teacher_outputs

    def _extract_loss(self, outputs):
        """
        Extract loss from model outputs (framework-dependent)

        Args:
            outputs: Model outputs

        Returns:
            Loss tensor or None
        """
        # Try common loss extraction patterns
        if hasattr(outputs, 'loss'):
            return outputs.loss

        if isinstance(outputs, dict):
            if 'loss' in outputs:
                return outputs['loss']
            # Sum all losses in dict (Detectron2 pattern)
            elif any('loss' in k.lower() for k in outputs.keys()):
                return sum(v for k, v in outputs.items() if 'loss' in k.lower())

        return None

    def online(self, mode=True):
        """Enable/disable online adaptation mode"""
        self.adapting = mode
        return self

    def offline(self):
        """Disable online adaptation mode"""
        return self.online(False)

from typing import Literal
from dataclasses import dataclass

from ....base import AdaptationConfig


@dataclass
class TeSTConfig(AdaptationConfig):
    adaptation_name = "TeSTEngine"

    base_type: Literal["rcnn", "swinrcnn", "rtdetr", "yolo11"] = "rcnn"

    # Optimizer
    optim: Literal["SGD", "Adam", "AdamW"] = "SGD"
    adapt_lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # Pseudo-label confidence threshold & weight regularization
    conf_threshold: float = 0.3
    weight_reg: float = 0.0

    # Augmentation (RandAugment num_ops)
    augment_strength_n: int = 2

    # Stage 1: teacher consistency loss weight
    lambda_cons: float = 1.0

    # Stage 2: entropy minimization weight
    lambda_ent: float = 0.25

    # Stage control
    stage: Literal["teacher", "student", "online"] = "online" # In this experiment, only the online state is used.

    # Offline teacher pre-adaptation epochs (used by fit())
    n_teacher_epochs: int = 10

    # Number of teacher gradient steps per batch in "online" mode
    n_teacher_steps: int = 10

    # Number of student gradient steps per batch in Stage 2
    n_student_steps: int = 1

    @classmethod
    def from_preset(cls, base_model, **kwargs):
        """Create configuration from preset."""
        from .....models import (
            FasterRCNNForObjectDetection, SwinRCNNForObjectDetection,
            RTDetrForObjectDetection, YOLO11ForObjectDetection
        )
        if isinstance(base_model, FasterRCNNForObjectDetection):
            return cls(base_type="rcnn", lambda_ent=0.5, n_teacher_epochs=1, n_student_steps=1, **kwargs)
        elif isinstance(base_model, SwinRCNNForObjectDetection):
            return cls(base_type="swinrcnn", lambda_ent=0.5, n_teacher_epochs=1, n_student_steps=1, **kwargs)
        elif isinstance(base_model, RTDetrForObjectDetection):
            return cls(base_type="rtdetr", lambda_ent=0.5, n_teacher_epochs=1, n_student_steps=1, **kwargs)
        elif isinstance(base_model, YOLO11ForObjectDetection):
            return cls(base_type="yolo11", lambda_ent=0.5, n_teacher_epochs=1, n_student_steps=1, **kwargs)
        else:
            raise ValueError(f"Unsupported base model type: {type(base_model)}")

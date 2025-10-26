"""
Framework Adapters for Test-Time Adaptation

This module provides adapters to handle framework-specific differences
across Detectron2, Transformers (RT-DETR), and Ultralytics.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Callable, Any
import torch
import torch.nn as nn


class FrameworkAdapter(ABC):
    """Base adapter interface for different deep learning frameworks"""

    framework_name: str = "base"

    @abstractmethod
    def identify_normalization_layers(
        self,
        model: nn.Module,
        layer_filter: Optional[Callable[[str, nn.Module], bool]] = None
    ) -> List[Tuple[str, nn.Module]]:
        """
        Identify normalization layers in the model

        Args:
            model: The model to inspect
            layer_filter: Optional filter function(name, module) -> bool

        Returns:
            List of (layer_name, layer_module) tuples
        """
        pass

    @abstractmethod
    def execute_forward(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device
    ) -> Any:
        """
        Execute forward pass with framework-specific input handling

        Args:
            model: The model to run
            batch: Input batch (framework-specific format)
            device: Target device

        Returns:
            Model outputs (framework-specific format)
        """
        pass

    @abstractmethod
    def prepare_batch(self, batch: Any, device: torch.device) -> Any:
        """
        Prepare batch data for the specific framework

        Args:
            batch: Input batch
            device: Target device

        Returns:
            Prepared batch
        """
        pass

    @abstractmethod
    def register_feature_hook(
        self,
        layer: nn.Module,
        hook_handler: Callable
    ) -> Any:
        """
        Register a forward hook on a layer

        Args:
            layer: Target layer
            hook_handler: Hook function

        Returns:
            Hook handle (for removal)
        """
        pass

    @classmethod
    def auto_detect(cls, model: nn.Module) -> "FrameworkAdapter":
        """
        Automatically detect the appropriate framework adapter

        Args:
            model: The model to inspect

        Returns:
            Appropriate FrameworkAdapter instance
        """
        # Check for Detectron2 models
        if hasattr(model, 'backbone') and hasattr(model, 'roi_heads'):
            return Detectron2Adapter()

        # Check for Transformers models (RT-DETR)
        if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
            if 'detr' in model.config.model_type.lower():
                return TransformersAdapter()

        # Check for Ultralytics models
        if hasattr(model, 'model') and hasattr(model, 'names'):
            return UltralyticsAdapter()

        # Default fallback
        raise ValueError(
            f"Could not auto-detect framework for model type {type(model)}. "
            f"Please specify adapter explicitly."
        )


class Detectron2Adapter(FrameworkAdapter):
    """Adapter for Detectron2 models (Faster R-CNN, Mask R-CNN, etc.)"""

    framework_name = "detectron2"

    def identify_normalization_layers(
        self,
        model: nn.Module,
        layer_filter: Optional[Callable[[str, nn.Module], bool]] = None
    ) -> List[Tuple[str, nn.Module]]:
        """Identify normalization layers in Detectron2 models"""
        from detectron2.layers import FrozenBatchNorm2d

        layers = []
        for name, module in model.named_modules():
            # Detectron2 commonly uses FrozenBatchNorm2d and BatchNorm2d
            if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d, nn.LayerNorm)):
                # Apply optional filter
                if layer_filter is None or layer_filter(name, module):
                    layers.append((name, module))

        return layers

    def execute_forward(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device
    ) -> Any:
        """Execute forward pass for Detectron2 models"""
        # Detectron2 models expect a list of dict inputs
        return model(batch)

    def prepare_batch(self, batch: Any, device: torch.device) -> Any:
        """Prepare batch for Detectron2 (already in correct format)"""
        # Detectron2 batches are typically already prepared by the dataset
        return batch

    def register_feature_hook(
        self,
        layer: nn.Module,
        hook_handler: Callable
    ) -> Any:
        """Register forward hook (standard PyTorch)"""
        return layer.register_forward_hook(hook_handler)

    def create_layer_filter_backbone(self) -> Callable[[str, nn.Module], bool]:
        """Create filter for backbone layers only"""
        def filter_fn(name: str, module: nn.Module) -> bool:
            return 'backbone' in name.lower()
        return filter_fn

    def create_layer_filter_roi_head(self) -> Callable[[str, nn.Module], bool]:
        """Create filter for ROI head layers"""
        def filter_fn(name: str, module: nn.Module) -> bool:
            return 'roi_heads' in name.lower()
        return filter_fn


class TransformersAdapter(FrameworkAdapter):
    """Adapter for Transformers models (RT-DETR, DETR, etc.)"""

    framework_name = "transformers"

    def identify_normalization_layers(
        self,
        model: nn.Module,
        layer_filter: Optional[Callable[[str, nn.Module], bool]] = None
    ) -> List[Tuple[str, nn.Module]]:
        """Identify normalization layers in Transformers models"""
        try:
            from transformers.models.rt_detr.modeling_rt_detr import RTDetrFrozenBatchNorm2d
        except ImportError:
            RTDetrFrozenBatchNorm2d = None

        layers = []
        for name, module in model.named_modules():
            # RT-DETR uses RTDetrFrozenBatchNorm2d, BatchNorm2d, and LayerNorm
            is_norm_layer = isinstance(module, (nn.BatchNorm2d, nn.LayerNorm))
            if RTDetrFrozenBatchNorm2d is not None:
                is_norm_layer = is_norm_layer or isinstance(module, RTDetrFrozenBatchNorm2d)

            if is_norm_layer:
                # Apply optional filter
                if layer_filter is None or layer_filter(name, module):
                    layers.append((name, module))

        return layers

    def execute_forward(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device
    ) -> Any:
        """Execute forward pass for Transformers models"""
        # Transformers models expect specific input format
        if isinstance(batch, dict):
            pixel_values = batch['pixel_values'].to(device)

            # Check if labels are provided
            if 'labels' in batch:
                labels = [
                    {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in label.items()}
                    for label in batch['labels']
                ]
                return model(pixel_values=pixel_values, labels=labels)
            else:
                return model(pixel_values=pixel_values)
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")

    def prepare_batch(self, batch: Any, device: torch.device) -> Any:
        """Prepare batch for Transformers models"""
        if isinstance(batch, dict):
            prepared = {'pixel_values': batch['pixel_values'].to(device)}
            if 'labels' in batch:
                prepared['labels'] = batch['labels']
            return prepared
        return batch

    def register_feature_hook(
        self,
        layer: nn.Module,
        hook_handler: Callable
    ) -> Any:
        """Register forward hook (standard PyTorch)"""
        return layer.register_forward_hook(hook_handler)

    def create_layer_filter_backbone(self) -> Callable[[str, nn.Module], bool]:
        """Create filter for backbone layers only"""
        try:
            from transformers.models.rt_detr.modeling_rt_detr import RTDetrFrozenBatchNorm2d
        except ImportError:
            RTDetrFrozenBatchNorm2d = None

        def filter_fn(name: str, module: nn.Module) -> bool:
            # RT-DETR backbone uses RTDetrFrozenBatchNorm2d
            is_backbone = 'backbone' in name.lower() and 'model.backbone' in name
            if RTDetrFrozenBatchNorm2d is not None:
                return is_backbone and isinstance(module, RTDetrFrozenBatchNorm2d)
            return is_backbone
        return filter_fn

    def create_layer_filter_encoder(self) -> Callable[[str, nn.Module], bool]:
        """Create filter for encoder layers only"""
        def filter_fn(name: str, module: nn.Module) -> bool:
            # Encoder includes: encoder_input_proj, lateral_convs, fpn_blocks, encoder.layers
            is_encoder = 'encoder' in name.lower() and 'decoder' not in name.lower()
            return is_encoder and isinstance(module, (nn.BatchNorm2d, nn.LayerNorm))
        return filter_fn

    def create_layer_filter_backbone_and_encoder(self) -> Callable[[str, nn.Module], bool]:
        """Create filter for backbone + encoder layers (exclude decoder)"""
        def filter_fn(name: str, module: nn.Module) -> bool:
            # Exclude decoder
            return 'decoder' not in name.lower()
        return filter_fn


class UltralyticsAdapter(FrameworkAdapter):
    """Adapter for Ultralytics models (YOLO v8, v11, etc.)"""

    framework_name = "ultralytics"

    def identify_normalization_layers(
        self,
        model: nn.Module,
        layer_filter: Optional[Callable[[str, nn.Module], bool]] = None
    ) -> List[Tuple[str, nn.Module]]:
        """Identify normalization layers in Ultralytics models"""
        layers = []
        for name, module in model.named_modules():
            # YOLO uses BatchNorm2d
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                # Apply optional filter
                if layer_filter is None or layer_filter(name, module):
                    layers.append((name, module))

        return layers

    def execute_forward(
        self,
        model: nn.Module,
        batch: Any,
        device: torch.device
    ) -> Any:
        """Execute forward pass for Ultralytics models"""
        # Ultralytics models typically expect tensor input
        if isinstance(batch, dict):
            images = batch.get('images', batch.get('pixel_values'))
            return model(images.to(device))
        elif isinstance(batch, torch.Tensor):
            return model(batch.to(device))
        else:
            return model(batch)

    def prepare_batch(self, batch: Any, device: torch.device) -> Any:
        """Prepare batch for Ultralytics models"""
        if isinstance(batch, dict):
            images = batch.get('images', batch.get('pixel_values'))
            return images.to(device)
        elif isinstance(batch, torch.Tensor):
            return batch.to(device)
        return batch

    def register_feature_hook(
        self,
        layer: nn.Module,
        hook_handler: Callable
    ) -> Any:
        """Register forward hook (standard PyTorch)"""
        return layer.register_forward_hook(hook_handler)

    def create_layer_filter_backbone(self) -> Callable[[str, nn.Module], bool]:
        """Create filter for backbone layers only"""
        def filter_fn(name: str, module: nn.Module) -> bool:
            return 'backbone' in name.lower() or 'model.model' in name
        return filter_fn

    def create_layer_filter_neck(self) -> Callable[[str, nn.Module], bool]:
        """Create filter for neck/FPN layers"""
        def filter_fn(name: str, module: nn.Module) -> bool:
            return 'neck' in name.lower() or 'fpn' in name.lower()
        return filter_fn


def create_adapter(model: nn.Module, adapter_type: Optional[str] = None) -> FrameworkAdapter:
    """
    Factory function to create appropriate framework adapter

    Args:
        model: The model to adapt
        adapter_type: Optional explicit adapter type ("detectron2", "transformers", "ultralytics")
                     If None, will auto-detect

    Returns:
        Appropriate FrameworkAdapter instance
    """
    if adapter_type is None:
        return FrameworkAdapter.auto_detect(model)

    adapter_type = adapter_type.lower()
    if adapter_type == "detectron2":
        return Detectron2Adapter()
    elif adapter_type == "transformers":
        return TransformersAdapter()
    elif adapter_type == "ultralytics":
        return UltralyticsAdapter()
    else:
        raise ValueError(
            f"Unknown adapter type: {adapter_type}. "
            f"Choose from: detectron2, transformers, ultralytics"
        )

"""
FLOPs Counter for Object Detection Models.

Usage:
    from ttadapters.utils.flops_counter import FLOPsCounter

    counter = FLOPsCounter(model, data_preparation=data_prep, device=device)
    result = counter.count(dataset)
    print(result)
"""

import gc
from contextlib import nullcontext

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..models.base import BaseModel, ModelProvider
from ..datasets import DataPreparation


# ── fvcore Wrappers per ModelProvider ──────────────────────────────────────────

class _Detectron2Wrapper(nn.Module):
    """Wraps Detectron2 model to accept (List[Dict],) positional arg for fvcore."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(batch)


class _UltralyticsWrapper(nn.Module):
    """Wraps Ultralytics model to accept img tensor for fvcore."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, img):
        return self.model(img)


class _HuggingFaceWrapper(nn.Module):
    """Wraps HuggingFace model to accept a single dict arg for fvcore."""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch_dict: dict):
        return self.model(**batch_dict)


# ── Main FLOPs Counter ─────────────────────────────────────────────────────────

class FLOPsCounter:
    """
    FLOPs Counter for detection models. Mirrors DetectionEvaluator interface.

    Uses fvcore to measure forward FLOPs from one real batch.

    Args:
        model: BaseModel instance
        data_preparation: DataPreparation instance (handles pre/post-processing)
        device: torch.device
        dtype: data type for inference

    Example:
        >>> counter = FLOPsCounter(model, data_preparation=data_prep, device=device)
        >>> result = counter.count(dataset)
        >>> print(result)
    """

    def __init__(
        self,
        model: BaseModel,
        data_preparation: DataPreparation,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model.to(device).to(dtype)
        self.data_preparation = data_preparation
        self.device = device
        self.dtype = dtype

    @staticmethod
    def count(
        model: BaseModel,
        data_preparation: DataPreparation,
        dataset,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
        warn_unsupported: bool = True,
    ) -> dict:
        """
        Count forward FLOPs using one real batch from the dataset.

        Args:
            model: BaseModel instance
            data_preparation: DataPreparation for the dataset
            dataset: Dataset to sample one batch from
            device: torch.device
            dtype: torch.dtype
            warn_unsupported: print unsupported ops warning (default True)

        Returns:
            dict with keys:
                - total_gflops: total GFLOPs
                - unsupported_ops: dict of ops not counted by fvcore
                - input_shape: shape info of the input used
                - model_provider: model provider name
        """
        try:
            from fvcore.nn import FlopCountAnalysis
        except ImportError:
            raise ImportError("fvcore is required: pip install fvcore")

        torch.cuda.empty_cache()
        gc.collect()

        model = model.to(device).to(dtype)
        model.eval()

        # ── Get one real batch ──────────────────────────────────────────────────
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=data_preparation.collate_fn,
        )
        batch = next(iter(loader))

        # ── Prepare inputs & wrap model per provider ────────────────────────────
        provider = model.model_provider
        base_model = model  # use the full model including pre/post-processing hooks

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype):
                match provider:
                    case ModelProvider.Detectron2:
                        # fvcore uses torch.jit.trace which cannot handle Detectron2's
                        # List[Dict] input format. Instead, preprocess manually and
                        # trace backbone only. FPN/RPN/ROI are excluded from count.
                        det2_model = base_model.model if hasattr(base_model, "model") else base_model
                        images = det2_model.preprocess_image(batch)  # -> ImageList
                        img_tensor = images.tensor.to(device)         # (N, C, H, W) padded
                        wrapped = det2_model.backbone                 # backbone + FPN
                        fvcore_inputs = (img_tensor,)
                        input_shape = list(img_tensor.shape)
                        print("UserWarning: Detectron2: measuring backbone+FPN only (RPN/ROIHead excluded)")

                    case ModelProvider.Ultralytics:
                        img = batch["img"].to(device)
                        wrapped = _UltralyticsWrapper(base_model)
                        fvcore_inputs = (img,)
                        input_shape = list(img.shape)

                    case ModelProvider.HuggingFace:
                        batch = {
                            k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()
                        }
                        batch["labels"] = [
                            {k: v.to(device) if isinstance(v, torch.Tensor) else v
                             for k, v in label.items()}
                            for label in batch["labels"]
                        ]
                        # Remove labels from FLOPs input (inference only)
                        inference_batch = {k: v for k, v in batch.items() if k != "labels"}
                        wrapped = _HuggingFaceWrapper(base_model)
                        fvcore_inputs = (inference_batch,)
                        input_shape = list(batch["pixel_values"].shape) if "pixel_values" in batch else "unknown"

                    case _:
                        raise ValueError(f"Unsupported model provider: {provider}")

        # ── Run fvcore ──────────────────────────────────────────────────────────
        wrapped.eval()
        flops = FlopCountAnalysis(wrapped, fvcore_inputs)
        flops.unsupported_ops_warnings(warn_unsupported)
        flops.uncalled_modules_warnings(False)

        total = flops.total()
        unsupported = flops.unsupported_ops()

        result = {
            "total_gflops": total / 1e9,
            "total_flops": total,
            "unsupported_ops": dict(unsupported),
            "input_shape": input_shape,
            "model_provider": provider.name,
        }

        # ── Pretty print ────────────────────────────────────────────────────────
        print("=" * 55)
        print(f"  FLOPs Analysis: {model.__class__.__name__}")
        print(f"  Provider      : {provider.name}")
        print(f"  Input shape   : {input_shape}")
        print("-" * 55)
        print(f"  Forward FLOPs : {total / 1e9:.2f} GFLOPs")
        if unsupported:
            print(f"  ⚠ Unsupported ops (excluded from count):")
            for op, count in unsupported.items():
                print(f"      {op}: {count} calls")
        else:
            print("  ✓ All ops supported")
        print("=" * 55)

        return result

    def __call__(self, dataset, **kwargs) -> dict:
        return FLOPsCounter.count(
            self.model,
            self.data_preparation,
            dataset,
            device=self.device,
            dtype=self.dtype,
            **kwargs,
        )

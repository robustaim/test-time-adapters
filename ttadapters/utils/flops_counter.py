"""
FLOPs Counter for Object Detection Models.

Uses torch.utils.flop_counter.FlopCounterMode (PyTorch >= 2.0, hook-based),
which does NOT require JIT tracing. Model calls follow the same pattern as
validator.py (DetectionEvaluator).

Usage:
    from ttadapters.utils.flops_counter import FLOPsCounter

    result = FLOPsCounter.count(
        model=model,
        data_preparation=model.DataPreparation(dataset.test, evaluation_mode=True),
        dataset=dataset.test,
        device=device,
        dtype=DATA_TYPE,
    )
"""

import gc
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from ..models.base import BaseModel, ModelProvider
from ..datasets import DataPreparation


class FLOPsCounter:
    """
    FLOPs Counter for object detection models.

    Mirrors DetectionEvaluator interface from validator.py.
    Uses torch.utils.flop_counter.FlopCounterMode (hook-based, no JIT trace).

    Supports:
        - ModelProvider.Detectron2   (FasterRCNN, SwinRCNN)
        - ModelProvider.Ultralytics  (YOLO11)
        - ModelProvider.HuggingFace  (RT-DETR)
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
    ) -> dict:
        """
        Count forward FLOPs using one real batch from the dataset.
        Model call pattern mirrors validator.py (DetectionEvaluator).

        Args:
            model:            BaseModel instance
            data_preparation: DataPreparation for the dataset
            dataset:          Dataset to load one batch from
            device:           torch.device
            dtype:            torch.dtype

        Returns:
            dict:
                total_gflops   – measured forward GFLOPs
                total_flops    – raw FLOPs integer
                input_shape    – shape of the model input used
                model_provider – provider name string
        """
        from torch.utils.flop_counter import FlopCounterMode

        torch.cuda.empty_cache()
        gc.collect()

        model = model.to(device).to(dtype)
        model.eval()

        # ── Load one real batch (same DataLoader pattern as validator.py) ───────
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=data_preparation.collate_fn,
        )
        batch = next(iter(loader))

        provider = model.model_provider

        # ── Prepare batch per provider (mirrors validator.py) ──────────────────
        input_shape = "see provider"

        match provider:
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
                input_shape = list(batch["pixel_values"].shape) if "pixel_values" in batch else "unknown"

            case ModelProvider.Ultralytics:
                batch["img"] = batch["img"].to(device)
                input_shape = list(batch["img"].shape)

            case ModelProvider.Detectron2:
                # No explicit device move — same as validator.py
                input_shape = [
                    next((v.shape for v in item.values()
                          if isinstance(v, torch.Tensor) and v.dim() == 3), "unknown")
                    for item in batch
                ]

            case _:
                raise ValueError(f"Unsupported model provider: {provider}")

        # ── Measure FLOPs via hook-based counter (no JIT trace) ────────────────
        flop_counter = FlopCounterMode(model, display=False)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype):
                with flop_counter:
                    # Exact same call pattern as validator.py
                    match provider:
                        case ModelProvider.Detectron2:
                            _ = model(batch)
                        case ModelProvider.Ultralytics:
                            _ = model(batch["img"])
                        case ModelProvider.HuggingFace:
                            _ = model(**batch)

        total = flop_counter.get_total_flops()

        result = {
            "total_gflops": total / 1e9,
            "total_flops": total,
            "input_shape": input_shape,
            "model_provider": provider.name,
        }

        # ── Find modules with params but 0 FLOPs (likely excluded ops) ─────────
        flop_counts = flop_counter.get_flop_counts()
        zero_flop_modules = []
        for name, module in model.named_modules():
            # Only check leaf-like modules that own params (e.g. Linear, Conv2d)
            own_params = list(module.parameters(recurse=False))
            if own_params:
                module_flops = sum(flop_counts.get(name, {}).values())
                if module_flops == 0:
                    zero_flop_modules.append((name, module.__class__.__name__))

        # ── Pretty print ────────────────────────────────────────────────────────
        print("=" * 60)
        print(f"  FLOPs Analysis : {model.__class__.__name__}")
        print(f"  Provider       : {provider.name}")
        print(f"  Input shape    : {input_shape}")
        print("-" * 60)
        print(f"  Forward FLOPs  : {total / 1e9:.2f} GFLOPs")
        if zero_flop_modules:
            print(f"  ⚠ Modules with params but 0 FLOPs (likely excluded):")
            for name, cls in zero_flop_modules[:10]:  # cap at 10
                print(f"      [{cls}] {name}")
            if len(zero_flop_modules) > 10:
                print(f"      ... ({len(zero_flop_modules) - 10} more)")
        else:
            print("  ✓ All parameterized modules have measured FLOPs")
        print("=" * 60)

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

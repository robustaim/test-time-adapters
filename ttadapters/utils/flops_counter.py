"""
FLOPs Counter for Object Detection Models.

Built on top of DetectionEvaluator's batch-handling logic.
Processes exactly one batch and measures forward FLOPs.

Usage (mirrors DetectionEvaluator):
    from ttadapters.utils.flops_counter import FLOPsCounter

    data_preparation = model.DataPreparation(dataset.test, evaluation_mode=True)
    evaluator_loader_params = dict(
        batch_size=1, shuffle=False, collate_fn=data_preparation.collate_fn
    )
    loader = DataLoader(dataset.test, **evaluator_loader_params)

    result = FLOPsCounter.count(model, loader=loader, device=device, dtype=DATA_TYPE)
"""

from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from ..models.base import BaseModel, ModelProvider
from ..datasets import DataPreparation
from .validator import DetectionEvaluator


class FLOPsCounter(DetectionEvaluator):
    """
    FLOPs Counter. Inherits DetectionEvaluator and reuses its batch-handling.

    Processes exactly ONE batch from the loader and measures forward FLOPs
    via torch.utils.flop_counter.FlopCounterMode (hook-based, no JIT trace).

    Usage:
        loader = DataLoader(dataset.test, **evaluator_loader_params)
        result = FLOPsCounter.count(model, loader=loader, device=device, dtype=dtype)
    """

    @staticmethod
    def count(
        model: BaseModel,
        loader: DataLoader,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float32,
        stream: torch.cuda.Stream | None = None,
    ) -> dict:
        """
        Count forward FLOPs for one batch using the same logic as
        DetectionEvaluator.evaluate_with_reset.

        Args:
            model:   BaseModel instance (eval mode will be set internally)
            loader:  DataLoader — same object passed to evaluate_with_reset
            device:  torch.device
            dtype:   torch.dtype
            stream:  Optional CUDA stream (same as DetectionEvaluator)

        Returns:
            dict:
                total_gflops     – forward GFLOPs
                total_flops      – raw FLOPs integer
                input_shape      – shape info of the input
                model_provider   – provider name string
                zero_flop_modules – list of (name, class) with params but 0 FLOPs
        """
        from torch.utils.flop_counter import FlopCounterMode

        model = model.to(device).to(dtype)
        model.eval()

        # ── Get exactly one batch (mirrors validator.py loop start) ────────────
        batch = next(iter(loader))

        # ── Device handling — exact copy from DetectionEvaluator ───────────────
        if model.model_provider == ModelProvider.HuggingFace:
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

        elif model.model_provider == ModelProvider.Ultralytics:
            batch["img"] = batch["img"].to(device)
            input_shape = list(batch["img"].shape)

        else:  # Detectron2
            input_shape = [
                next((v.shape for v in item.values()
                      if isinstance(v, torch.Tensor) and v.dim() == 3), "unknown")
                for item in batch
            ]

        # ── Measure FLOPs — same model call as DetectionEvaluator ─────────────
        flop_counter = FlopCounterMode(model, display=False)
        stream_context = torch.cuda.stream(stream) if stream is not None else nullcontext()

        with torch.no_grad():
            with stream_context:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    with flop_counter:
                        # Mirrors validator.py model call exactly
                        match model.model_provider:
                            case ModelProvider.Detectron2:
                                _ = model(batch)
                            case ModelProvider.Ultralytics:
                                _ = model(batch["img"])
                            case ModelProvider.HuggingFace:
                                _ = model(**batch)
                            case _:
                                raise ValueError(f"Unsupported provider: {model.model_provider}")

        total = flop_counter.get_total_flops()

        # ── Find modules with params but 0 FLOPs (likely excluded) ────────────
        flop_counts = flop_counter.get_flop_counts()
        zero_flop_modules = [
            (name, module.__class__.__name__)
            for name, module in model.named_modules()
            if list(module.parameters(recurse=False))
            and sum(flop_counts.get(name, {}).values()) == 0
        ]

        result = {
            "total_gflops": total / 1e9,
            "total_flops": total,
            "input_shape": input_shape,
            "model_provider": model.model_provider.name,
            "zero_flop_modules": zero_flop_modules,
        }

        # ── Pretty print ───────────────────────────────────────────────────────
        print("=" * 60)
        print(f"  FLOPs Analysis : {model.__class__.__name__}")
        print(f"  Provider       : {model.model_provider.name}")
        print(f"  Input shape    : {input_shape}")
        print("-" * 60)
        print(f"  Forward FLOPs  : {total / 1e9:.2f} GFLOPs")
        if zero_flop_modules:
            print(f"  ⚠ Modules with params but 0 FLOPs (likely excluded):")
            for name, cls in zero_flop_modules[:10]:
                print(f"      [{cls}] {name}")
            if len(zero_flop_modules) > 10:
                print(f"      ... and {len(zero_flop_modules) - 10} more")
        else:
            print("  ✓ All parameterized modules have measured FLOPs")
        print("=" * 60)

        return result

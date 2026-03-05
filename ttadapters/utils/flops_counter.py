"""
FLOPs Counter for Object Detection Models.

Inherits DetectionEvaluator and reuses its __init__ and batch-handling logic.

Usage (identical interface to DetectionEvaluator):
    data_preparation = model.DataPreparation(dataset.test, evaluation_mode=True)
    evaluator_loader_params = dict(
        batch_size=1, shuffle=False, collate_fn=data_preparation.collate_fn
    )
    loader = DataLoader(dataset.test, **evaluator_loader_params)

    counter = FLOPsCounter(model, classes=CLASSES, data_preparation=data_preparation,
                           dtype=DATA_TYPE, device=device, no_grad=True)
    result = counter.count(loader)
"""

from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from ..models.base import ModelProvider
from .validator import DetectionEvaluator


class FLOPsCounter(DetectionEvaluator):
    """
    FLOPs Counter. Fully inherits DetectionEvaluator (same __init__).

    Adds a single `count(loader)` method that:
    - Takes the same DataLoader used with DetectionEvaluator
    - Processes exactly ONE batch
    - Measures forward FLOPs via FlopCounterMode (hook-based, no JIT trace)

    Usage:
        counter = FLOPsCounter(model, classes=CLASSES,
                               data_preparation=data_prep,
                               dtype=dtype, device=device)
        result = counter.count(loader)
    """

    def count(
        self,
        *args, **kwargs
    ) -> dict:
        """
        Measure forward FLOPs for one batch.
        Batch handling mirrors DetectionEvaluator.evaluate_with_reset exactly.

        Supports being passed as a callback to Scenario.play(), natively
        extracting the DataLoader from args or kwargs.

        Returns:
            dict with total_gflops, total_flops, input_shape, zero_flop_modules
        """
        # Scenario.play passes: (desc, loader, loader_length, **kwargs)
        # Handle flexible args just like evaluate()
        loader = kwargs.get("loader")
        if loader is None:
            # Look for the first argument that is a DataLoader
            for arg in args:
                if isinstance(arg, DataLoader):
                    loader = arg
                    break
        if loader is None and len(args) > 1:
            loader = args[1] # fallback to 2nd pos arg

        if loader is None:
            raise ValueError("Could not find DataLoader in arguments")
        from torch.utils.flop_counter import FlopCounterMode

        model = self.model
        device = self.device
        dtype = self.dtype
        stream = self.stream

        model.to(device).to(dtype)
        model.eval()

        # ── Get exactly one batch ───────────────────────────────────────────────
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

        # ── Measure FLOPs — same model call pattern as DetectionEvaluator ──────
        flop_counter = FlopCounterMode(model, display=False)
        stream_context = torch.cuda.stream(stream) if stream is not None else nullcontext()

        with torch.no_grad():
            with stream_context:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    with flop_counter:
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

        # ── Find modules with params but 0 FLOPs (likely excluded ops) ─────────
        flop_counts = flop_counter.get_flop_counts()
        
        # FlopCounterMode.get_flop_counts() keys might have 'Global.' prefix or 
        # differ slightly from named_modules(). We flatten the counted keys for safer matching.
        counted_keys = set()
        for k in flop_counts.keys():
            # Usually format is "Global.model_name.layer_name" or "layer_name"
            # We strip "Global." if it exists to match named_modules()
            clean_k = k.replace("Global.", "", 1) if k.startswith("Global.") else k
            counted_keys.add(clean_k)

        zero_flop_modules = []
        for name, module in model.named_modules():
            # Only check leaf-like modules that own params (e.g. Linear, Conv2d)
            if list(module.parameters(recurse=False)):
                # If the exact name is not in the flattened counted_keys, and 
                # none of its children contributed either, it's highly likely excluded.
                # However, exact matching is tricky with FlopCounterMode.
                # A safer check: does its name exist anywhere in the flop count keys?
                
                module_flops = sum(
                    sum(v.values()) for k, v in flop_counts.items() 
                    if k == name or k.endswith(f".{name}") or k.startswith(f"Global.{name}")
                )
                
                if module_flops == 0:
                    zero_flop_modules.append((name, module.__class__.__name__))

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
            print("  ⚠ Modules with params but 0 FLOPs (likely excluded):")
            for name, cls in zero_flop_modules[:10]:
                print(f"      [{cls}] {name}")
            if len(zero_flop_modules) > 10:
                print(f"      ... and {len(zero_flop_modules) - 10} more")
        else:
            print("  ✓ All parameterized modules have measured FLOPs")
        print("=" * 60)

        return result

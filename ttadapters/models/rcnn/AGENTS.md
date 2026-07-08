<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-07-08 -->

# rcnn

## Purpose
Detectron2-backed two-stage detectors. Provides `FasterRCNNForObjectDetection` (ResNet-50-FPN) and `SwinRCNNForObjectDetection` (Swin Transformer backbone) wrapped to satisfy `BaseModel` so the TTA harness treats them identically to YOLO11 / RT-DETR. Includes the augmentation/transforms layer and Detectron2 forward hooks needed for intermediate-feature TTA methods.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Public API: `FasterRCNNForObjectDetection`, `SwinRCNNForObjectDetection`. |
| `modeling_rcnn.py` | The detector classes themselves (~20 KB). Both inherit `BaseModel` with `model_provider = ModelProvider.Detectron2`. |
| `transforms.py` | Detectron2-style augmentation pipeline — produces tensors with the channel order Detectron2 expects (BGR uint8). |
| `hooks.py` | Forward hooks (e.g. for capturing FPN feature maps) used by deep-supervision methods. |
| `README.md` | Per-model notes (~27 KB) on Detectron2 setup, weights, and quirks. |

## For AI Agents

### Working In This Directory
- Detectron2 is built from source via `uv sync` because Meta does not publish prebuilt wheels — see the root README CUDA-mismatch instructions if `import detectron2` fails.
- `load_from("detectron2://...")` triggers the `DetectionCheckpointer` branch in `BaseModel.load_from` — `strict=False` is forced and `exclude_keys` is re-applied after the load.
- Faster R-CNN expects BGR channel order; `transforms.py` is responsible for that. Don't bypass it when wiring a new dataset.
- The `hooks.py` file registers forward hooks on FPN levels — TTA methods (ActMAD, GITA, CascadedNorm, FlowAdaptation, etc.) rely on these to extract intermediate activations. If a method needs new hooks, prefer adding them here and importing from method code.
- **DDP-aware.** The trainer supports DistributedDataParallel: `hooks.py` only writes metrics to `EventStorage` on `comm.is_main_process()`, and the eval path unwraps DDP (`getattr(self.model, "module", self.model)`) before reading `pixel_mean` / running the model, so single-process eval still sees the real module attributes. `seed_worker` is a `@staticmethod` referenced as `self.seed_worker` for DataLoader `worker_init_fn` reproducibility. Preserve the unwrap when touching the eval path.

### Testing Requirements
- Smoke-test: `model = FasterRCNNForObjectDetection.from_dataset(coco_dataset)` and run a forward pass on a single image.

### Common Patterns
- TTA configs branch on `isinstance(base_model, FasterRCNNForObjectDetection)` vs `isinstance(base_model, SwinRCNNForObjectDetection)` for backbone-specific defaults — keep these two classes as the canonical Detectron2 entrypoints.

## Dependencies

### Internal
- `..base` — `BaseModel`, `ModelProvider`.

### External
- `detectron2` (forked at `robustaim/detectron2`), `torch`, `torchvision`.

<!-- MANUAL: -->

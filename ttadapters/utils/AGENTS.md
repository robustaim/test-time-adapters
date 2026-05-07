<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# utils

## Purpose
Cross-cutting helpers used by the harness and notebooks: a detection `Evaluator` that knows about all three model providers and can run multiple models in parallel via CUDA Streams + `asyncio`, a `FLOPsCounter` built on top of it, and visualization helpers for displaying bounding-box frames.

## Key Files
| File | Description |
|------|-------------|
| `validator.py` | `Evaluator` and `DetectionEvaluator`. Per-model `torch.cuda.Stream`s for parallel evaluation, COCO-style mAP via `supervision.metrics.MeanAveragePrecision`, OOM-aware `tqdm` cleanup. Used by scenario `play()` callbacks. |
| `flops_counter.py` | `FLOPsCounter` — subclass of `DetectionEvaluator`. Single-batch FLOPs measurement using PyTorch's `FlopCounterMode` (hook-based, no JIT trace). Same `__init__` signature as `DetectionEvaluator`. |
| `visualizer.py` | `visualize_bbox_frame(dataset, idx=...)` — Matplotlib helper for inspecting a dataset entry's image + bboxes (uses `torchvision.tv_tensors.BoundingBoxes`). |

## For AI Agents

### Working In This Directory
- `Evaluator.__init__` accepts a single `BaseModel` *or* a list/tuple. The `do_parallel` flag is set by that check; downstream code branches on it.
- When parallel-evaluating, each model gets its own `torch.cuda.Stream` so that forward passes overlap. Evaluation is dispatched via `asyncio.gather` with `nest_asyncio` enabled — keep that in mind when invoking from inside an existing event loop.
- `FLOPsCounter.count(loader)` processes exactly **one** batch — don't change that semantics; FLOP measurement is meant to be a single-shot characterization.
- `visualize_bbox_frame` defaults to a random index when `idx is None` and uses `IPython.display`/`ipywidgets` — it's notebook-only and will not work in headless CLI runs.

### Testing Requirements
- Sanity-check `Evaluator` after changes by running it on a tiny SHIFT subset (`SHIFTContinuousSubsetForObjectDetection`) with `batch_size=1, dtype=torch.float32, device=torch.device("cuda")`.
- Confirm `FLOPsCounter` returns the expected MACs for a known backbone (e.g. ResNet-50 ~4 GFLOPs at 224 input).

### Common Patterns
- The evaluator is invoked from `BaseScenario.play(script=...)` — `script` is `Evaluator.evaluate` (or a wrapper). Maintain that callable signature: `(desc: str, loader: DataLoader, loader_length: int, **kwargs) -> dict | list[dict]`.
- The `synchronize` and `no_grad` flags are propagated through `evaluate_*` static/instance methods; preserve them when extending.

## Dependencies

### Internal
- `ttadapters.models.base` — `BaseModel`, `ModelProvider`.
- `ttadapters.datasets` — `DataPreparation`.

### External
- `torch`, `torchvision.ops.box_convert`, `torchvision.tv_tensors`, `supervision` (`Detections`, `MeanAveragePrecision`), `tqdm`, `pandas`, `matplotlib`, `IPython`, `ipywidgets`, `nest_asyncio`.

<!-- MANUAL: -->

<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# models

## Purpose
Backbone-agnostic model layer. Wraps three different detection ecosystems (Detectron2, HuggingFace Transformers, Ultralytics) behind a common `BaseModel` API so that TTA `AdaptationEngine`s can be plugged on top without caring about the underlying framework. Provides `BaseModel.load_from()` with branching for `detectron2://`, `http(s)://`, HuggingFace Hub, and local `.pt` weights, plus a `from_dataset()` classmethod that pulls the right preset weights for a given dataset via the per-class `ModelRegistry`.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Public API: `from .rcnn import *`, `from .yolo11 import *`, `from .rt_detr import *`. |
| `base.py` | `BaseModel(nn.Module, PushToHubMixin)`, `ModelProvider` enum (`Detectron2` / `HuggingFace` / `Ultralytics`), `WeightsInfo` dataclass, plus task mixins (`ImageClassificationMixin`, `ObjectDetectionMixin`, `SemanticSegmentationMixin`, `InstanceSegmentationMixin`, `PanopticSegmentationMixin`). |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `rcnn/` | Detectron2-backed Faster R-CNN and Swin-RCNN detectors (see [rcnn/AGENTS.md](rcnn/AGENTS.md)) |
| `rt_detr/` | HuggingFace RT-DETR detector (see [rt_detr/AGENTS.md](rt_detr/AGENTS.md)) |
| `yolo11/` | Ultralytics YOLO11 detector (see [yolo11/AGENTS.md](yolo11/AGENTS.md)) |
| `resnet/` | Image-classification ResNet wrapper (HuggingFace) (see [resnet/AGENTS.md](resnet/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- New models must set both `model_name` (free-form) and `model_provider` (one of `ModelProvider.{Detectron2, HuggingFace, Ultralytics}`). The latter routes `load_from()` correctly.
- For Detectron2 weights, pass `weight_path="detectron2://..."` to `load_from()`. The base class uses `DetectionCheckpointer` and re-applies `exclude_keys` after the load.
- For HuggingFace, `load_from()` instantiates a *reference* model via `from_pretrained` to materialize the architecture, then copies its `state_dict()` and clears the temp via `cuda.empty_cache()` + `gc.collect()`. Replicate this pattern when adding HF backbones.
- `BaseModel.from_dataset(dataset)` looks up `cls.ModelRegistry.<dataset_name>` (a `WeightsInfo`) — register per-dataset weight presets there.
- `resnet/` is **not** wildcard-exported from `models/__init__.py`; import it explicitly if needed.

### Testing Requirements
- After adding a new model, smoke-test all four pathways of `load_from()` if applicable (URL / detectron2 / HF / local).

### Common Patterns
- Per-model folder layout mirrors HuggingFace: `__init__.py`, `modeling_<name>.py`, optional `configuration_<name>.py`, optional `transforms.py` / `wrappers.py` / `hooks.py`, and a per-model `README.md`.
- The `Trainer` and `ModelRegistry` inner classes on `BaseModel` are placeholders that subclasses populate.

## Dependencies

### Internal
- `ttadapters.datasets` — `BaseDataset` / `DataPreparation`. Models accept either a `BaseDataset` instance or a string `dataset_name`.

### External
- `torch.nn`, `transformers.utils.PushToHubMixin`, `detectron2.checkpoint.DetectionCheckpointer` (lazy import), `torch.hub`, `tqdm`.

<!-- MANUAL: -->

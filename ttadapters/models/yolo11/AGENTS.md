<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# yolo11

## Purpose
Ultralytics YOLO11 detector wrapped as `BaseModel`. YOLO11's backbone is built from `model.<i>` blocks (stem, C3k2, etc.) addressed via the `(^|\.)model\.<i>\.` regex pattern that recurs throughout the TTA preset enums.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Public API: `YOLO11ForObjectDetection`. Also installs an alias: `sys.modules['ttadapters.models.yolo11.modelings'] = modeling_yolo11` so older imports keep working. |
| `modeling_yolo11.py` | The detector class (~19 KB). Inherits `BaseModel` with `model_provider = ModelProvider.Ultralytics`. |
| `wrappers.py` | Thin wrappers/adapters between Ultralytics' API and the `BaseModel` contract. |
| `README.md` | Per-model notes (~29 KB) on YOLO11 weights, layer indices, and pre/post-processing. |

## For AI Agents

### Working In This Directory
- The `sys.modules` alias in `__init__.py` exists for backwards compatibility — do not remove without first searching for `ttadapters.models.yolo11.modelings` references.
- YOLO11's preprocessing pads to a square with a default fill of **114** (gray). Many TTA methods read this constant from `mask_value: int = 114` in their configs (`GITAConfig`, `CascadedNormConfig`, `FlowAdaptationConfig`). Don't change `mask_value` defaults without coordinating across configs.
- TTA target patterns for YOLO11 are indexed by layer position: `model.0` (stem), `model.1` (first stride), `model.[246]` (early C3k2 blocks), `model.[24]` (early stages used in CascadedNorm S1–S4 strategies). New presets must respect those layer-index conventions.
- Setting `masked_processing=True` is the YOLO11 default in cascade-style methods because of the gray-padding regions.

### Testing Requirements
- Smoke-test: `from ttadapters.models import YOLO11ForObjectDetection` and a forward pass on a `(B, 3, 640, 640)` tensor.

### Common Patterns
- Wildcard import in `models/__init__.py` re-exports `YOLO11ForObjectDetection` to package level.

## Dependencies

### Internal
- `..base` — `BaseModel`, `ModelProvider`.

### External
- `ultralytics`, `torch`, `torchvision`.

<!-- MANUAL: -->

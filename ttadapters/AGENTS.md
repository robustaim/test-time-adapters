<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# ttadapters

## Purpose
The installable Python package. Provides the four building blocks of the test-time adaptation harness: **datasets** (corruption / domain-shift wrappers), **models** (Detectron2 / Ultralytics / HF detector adapters that share a common `BaseModel`), **methods** (TTA `AdaptationEngine` implementations registered with HuggingFace Auto* classes), and **utils** (evaluator, FLOPs counter, visualizer). All public symbols flow through the leaf packages — adding a new method or model ultimately appears in `ttadapters.methods` / `ttadapters.models` via wildcard re-exports.

## Key Files
| File | Description |
|------|-------------|
| (no top-level Python files) | This package's `__init__.py` is implicit; entrypoints are inside the four sub-packages. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `datasets/` | Dataset wrappers and TTA scenarios (see [datasets/AGENTS.md](datasets/AGENTS.md)) |
| `models/` | Backbone/detector wrappers exposing `BaseModel` (see [models/AGENTS.md](models/AGENTS.md)) |
| `methods/` | TTA methods (`AdaptationConfig` + `AdaptationEngine` per algorithm) (see [methods/AGENTS.md](methods/AGENTS.md)) |
| `utils/` | Evaluator, FLOPs counter, visualizer (see [utils/AGENTS.md](utils/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- The package distinguishes three model providers (`Detectron2`, `HuggingFace`, `Ultralytics`) via `models.base.ModelProvider`. Code that branches on provider must check `model.model_provider`, not class identity.
- Each `AdaptationEngine` wraps a `BaseModel` and stores `base_state` (CPU snapshot) plus `base_grad_state` so `reset()` can restore the source model between scenario steps. Don't bypass `reset()` when introducing per-trial state.
- `AdaptationEngine.online(mode=True)` is the contract for entering test-time mode: `eval()` + only `online_parameters()` get `requires_grad=True`. Subclasses override `online_parameters()` to expose the trainable subset (e.g. only BN affines).
- Wildcard imports (`from .X import *`) propagate symbols up the tree; new public classes need to be exported through every parent's `__init__.py`.

### Testing Requirements
- `pytest` for unit/integration tests; `ruff check` for lint.
- Whenever a new method is added, also register it in `methods/__init__.py` chain and run `python -c "from ttadapters.methods import *"` to confirm AutoConfig/AutoAdaptationEngine registration succeeds without conflicts.

### Common Patterns
- **Auto registration triplet** (per method): `configuration_<name>.py` → `<Name>Config` (subclass of `AdaptationConfig`); `modeling_<name>.py` → `<Name>Engine` (subclass of `AdaptationEngine`); leaf `__init__.py` does `AutoConfig.register(...)` + `AutoAdaptationEngine.register(...)` + `AutoAdaptationEngineForObjectDetection.register(...)`.
- **Backbone dispatch**: configs always implement `from_preset(base_model)` with `isinstance` checks against `FasterRCNNForObjectDetection / SwinRCNNForObjectDetection / RTDetrForObjectDetection / YOLO11ForObjectDetection`.
- **Naming-by-class**: `AdaptationConfig.__init_subclass__` derives `model_type` from `adaptation_name` via snake_case regex. Set `adaptation_name` (e.g. `"DUAEngine"`) and `model_type` populates automatically.

## Dependencies

### Internal
- Cross-cuts the whole repo. `methods/` imports from `models/` (via `from_preset`) and from `datasets/` (via `BaseDataset` / `DataPreparation`).

### External
- `torch`, `torchvision`, `transformers`, `detectron2`, `timm`, `muon-optimizer`, `supervision`, `tqdm`.

<!-- MANUAL: -->

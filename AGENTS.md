<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# tta_harnes (test-time-adapters)

## Purpose
A Python package and ready-to-go playground for **Test-time Adaptation (TTA)** methods, focused on object detection and image classification under domain shift. Wraps detectors from Detectron2, Ultralytics, and HuggingFace Transformers behind a common `BaseModel` / `AdaptationEngine` interface so different TTA methods (BatchNorm replacement, entropy minimization, pseudo-label teachers, parameter-efficient adaptation, deep supervision, sampling) can be plugged into the same harness and evaluated across continual / gradual / standard scenarios on driving datasets (SHIFT, CityScapes, ACDC).

## Key Files
| File | Description |
|------|-------------|
| `pyproject.toml` | Project manifest. Python `>=3.11`, distributed as `ttadapters` 1.0.0 via `uv`. Pulls torch from a private CUDA 12.8 index, plus `detectron2` + `shift-dev` from the `robustaim` GitHub forks. |
| `uv.lock` | Locked dependency graph for `uv sync`. |
| `README.md` | Academic timeline of TTA, methods checklist, SHIFT split sizes, and install/run instructions. |
| `LICENSE` | Project license. |
| `example.ipynb` | End-to-end batch experiment notebook. |
| `pretraining.ipynb` | Source-domain pretraining notebook. |
| `example_batch.sh` / `example_batch_scene.sh` | Shell drivers for batched experiments. |
| `test.py` / `test.ipynb` | Untracked dev/scratch scripts (top-level git status). |
| `.gitignore` / `.gitattributes` | VCS rules. |
| `data` (symlink) | Symlink to `/home/work/t2a_voicestudio/tta/data` — dataset root. Do **not** commit data. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `ttadapters/` | Main Python package: datasets, models, methods, utils (see [ttadapters/AGENTS.md](ttadapters/AGENTS.md)) |
| `docs/` | Documentation, figures, and external reference notebooks (see [docs/AGENTS.md](docs/AGENTS.md)) |
| `.omc/` | oh-my-claudecode session/state (tooling only — not project source) |
| `.venv/` | Local `uv`-managed virtual environment (do not edit) |

## For AI Agents

### Working In This Directory
- Use `uv sync --extra torch` (Linux) / `uv sync --extra torch-cu128` (Windows) to set up; never invoke pip directly. The `torch` index in `pyproject.toml` is `explicit`, so torch must come from that index.
- `detectron2` is built from source via `uv` because Meta does not publish prebuilt wheels for current PyTorch. CUDA mismatches need `CUDA_HOME` / `PATH` / `LD_LIBRARY_PATH` overrides — see README "Usage" block.
- This repository is also installable as a library: `uv add git+https://github.com/robustaim/test-time-adapters.git`. Public surface is `from ttadapters.methods import ...` and `from ttadapters.models import ...`.
- Don't add new top-level scripts; prefer notebooks under repo root or new modules inside `ttadapters/`.

### Testing Requirements
- `pytest` for tests, `ruff check` for linting (per project memory).
- For end-to-end smoke checks, run cells of `example.ipynb` against a small SHIFT subset via `SHIFTContinuousSubsetForObjectDetection`.

### Common Patterns
- **HuggingFace-style configs**: every TTA method ships an `AdaptationConfig` subclass auto-registered with `transformers.AutoConfig`, plus an `AdaptationEngine` registered with `AutoAdaptationEngine` / `AutoAdaptationEngineForObjectDetection`. Adding a new method follows that triplet (`configuration_*.py`, `modeling_*.py`, `__init__.py` registers both).
- **Per-backbone presets**: `AdaptationConfig.from_preset(base_model)` dispatches on `isinstance(base_model, FasterRCNNForObjectDetection | SwinRCNNForObjectDetection | RTDetrForObjectDetection | YOLO11ForObjectDetection)` and returns sensible defaults. Match this contract when adding methods.
- **Round/seed harness**: `MethodContainer.go_rounds()` flips `cudnn.deterministic=True` and seeds `random/numpy/torch/cuda` from `seed_base+round`. Keep deterministic guarantees when changing it.

## Dependencies

### External
- **PyTorch / TorchVision** (CUDA 12.8 wheels, explicit index)
- **transformers** (>=4.57.6) — `PreTrainedModel`, `AutoConfig`, `AutoModel` are the integration points
- **detectron2** (forked at `robustaim/detectron2`) — Faster R-CNN / Swin-RCNN backbones
- **timm** — backbone provider for some configs
- **muon-optimizer** — `Muon`/`MuonWithAuxAdam` optimizers exposed by `AdaptationEngine`
- **supervision**, **pycocotools**, **cityscapesscripts**, **shift-dev** (forked) — dataset/eval tooling
- **imagecorruptions** — corruption synthesis
- **accelerate**, **ipykernel**, **tqdm**, **gdown**, **hf-xet** — runtime/dev plumbing

<!-- MANUAL: Custom project notes can be added below -->

<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# methods

## Purpose
The catalog of **test-time adaptation (TTA) methods**. Each method is a self-contained sub-tree following a HuggingFace-style triplet: `<Name>Config` (subclass of `AdaptationConfig`), `<Name>Engine` (subclass of `AdaptationEngine`), and a leaf `__init__.py` that registers both with `transformers.AutoConfig`, `AutoAdaptationEngine`, and `AutoAdaptationEngineForObjectDetection`. Methods are organized by **adaptation strategy** (BatchNorm replacement, entropy minimization, deep supervision, regularizers, parameter-efficient adaptation, sample selection).

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Re-exports everything in `base` plus `from .auxtasks/batchnorms/deepsupervisions/entropies/pefts/regularizers import *`. Note: `samplers/` is **not** wildcard-exported (kept for active-sampling helpers). |
| `base.py` | The contract everyone implements. Defines `MethodContainer` (round/seed harness), `OnlineMixin`, `AdaptationConfig` (HF `PretrainedConfig` subclass with `optim`, `adapt_lr`, `momentum`, `verbose`), `AdaptationEngine` (full TTA lifecycle: `online()`/`offline()`, `optimizer` lazy property, `reset()` that restores `base_state`, `online_parameters()` hook, `fit()` for offline pretraining), and `AutoAdaptationEngine{,ForObjectDetection}` registries. |
| `auto.py` | One-liner re-export of `transformers.AutoConfig` (placeholder; `AutoAdaptationEngine` registers Configs into HF's `CONFIG_MAPPING`). |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `auxtasks/` | Auxiliary-task TTA (TTT family) — currently empty stubs (see [auxtasks/AGENTS.md](auxtasks/AGENTS.md)) |
| `batchnorms/` | Batch-norm statistics replacement (NORM, DUA) (see [batchnorms/AGENTS.md](batchnorms/AGENTS.md)) |
| `deepsupervisions/` | Layer-wise / cascaded supervision (ActMAD, CascadedNorm, FlowAdaptation, GITA, ObjectNorm) (see [deepsupervisions/AGENTS.md](deepsupervisions/AGENTS.md)) |
| `entropies/` | Entropy-minimization TTA (TENT, TENT++, EATA, MEMO, DeYO) — empty stubs (see [entropies/AGENTS.md](entropies/AGENTS.md)) |
| `pefts/` | Parameter-Efficient Fine-Tuning TTA (WHW low-rank, SGP pruning) (see [pefts/AGENTS.md](pefts/AGENTS.md)) |
| `regularizers/` | Consistency-regularization TTA (Mean Teacher, TeST) (see [regularizers/AGENTS.md](regularizers/AGENTS.md)) |
| `samplers/` | Sample selection / active learning helpers (ActMAD active sampler) (see [samplers/AGENTS.md](samplers/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- Adding a new method = create a new leaf `<group>/<family>/<name>/` folder containing `configuration_<name>.py`, `modeling_<name>.py`, and `__init__.py` that registers all three Auto classes (config, engine, engine-for-OD).
- Always implement `from_preset(base_model, **kwargs)` on the config — methods key per-backbone defaults off `isinstance` against the four detector classes (`FasterRCNN`, `SwinRCNN`, `RTDetr`, `YOLO11`).
- The `MethodContainer` round mechanism (`names()`, `methods()`, `go_rounds()`) is what `example.ipynb` uses to iterate seeded trials. It flips `cudnn.deterministic=True` and seeds `random/np/torch/cuda` from `seed_base + round`. Don't change seeding semantics without updating the notebook contract.
- `AdaptationEngine.online_parameters()` defaults to ALL `base_model.parameters()`; subclasses MUST narrow this to the trainable subset (e.g. only BN affines, only adapters), otherwise the optimizer touches frozen weights and `reset()` cannot restore source.
- `__getattr__` on `AdaptationEngine` delegates to `base_model` — adding methods that *shadow* base model attributes silently breaks delegation; pick non-conflicting names.

### Testing Requirements
- Each new method should at minimum register without exception: `python -c "from ttadapters.methods import *"`.
- For functional tests, hook into a small `SHIFTContinuousSubsetForObjectDetection` and run a few batches through `Evaluator.evaluate(...)` from `ttadapters.utils.validator`.

### Common Patterns
- **Optimizer dispatch** lives entirely in `AdaptationEngine.optimizer` — five choices: `SGD`, `Adam`, `AdamW`, `Muon`, `MuonWithAuxAdam`. The Muon variants split params by `ndim` (matrix vs vector). Custom optimizers belong in the engine subclass, not in `base.py`.
- **Loss class** is set via `loss_class = nn.MSELoss` at class level; lazily instantiated as `_loss_function` on first access.
- **`reset()` semantics**: load `base_state` back into `base_model`, re-apply `online()` and dtype/device, zero grads, optionally return + clear stats. Methods that maintain teacher / EMA / pruning masks must override `reset()` to also restore those.
- **`from_preset` is mandatory** for the smoke-tests in `example.ipynb` to work. Match existing branches when adding a new backbone.

## Dependencies

### Internal
- `ttadapters.models` — engines hold a `base_model: BaseModel`; configs import `models` lazily inside `from_preset` to avoid circular imports.
- `ttadapters.datasets` — `DataPreparation` is consumed by `fit()` and by the evaluator.

### External
- `torch.nn`, `torch.optim`, `transformers.PretrainedConfig` / `PreTrainedModel` / `AutoModel`, `muon-optimizer` (`Muon`, `MuonWithAuxAdam`), `numpy`.

<!-- MANUAL: -->

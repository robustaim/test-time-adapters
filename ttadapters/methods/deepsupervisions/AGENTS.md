<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# deepsupervisions

## Purpose
**Layer-wise / cascaded supervision** TTA — adapts at test time by applying losses to *intermediate* layers (not just the model output). Methods here split into three flavors based on where supervision is injected: **full** (every layer aligned to source feature stats, ActMAD), **input** (input-transformation modules + early-layer alignment, CascadedNorm / FlowAdaptation / GITA), **local** (object-region normalization, ObjectNorm — placeholder).

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | `from .full.actmad import *` and `from .input.gita import *`. Note: `cascaded_norm`, `flow_adaptation`, and `local.objectnorm` are intentionally NOT re-exported here — they're imported directly when needed. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `full/` | Full-network feature alignment to source stats (ActMAD) (see [full/AGENTS.md](full/AGENTS.md)) |
| `input/` | Input-transformation + early-layer cascade alignment (CascadedNorm, FlowAdaptation, GITA) (see [input/AGENTS.md](input/AGENTS.md)) |
| `local/` | Object-local normalization (ObjectNorm) (see [local/AGENTS.md](local/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- Methods here typically **need source statistics** (mean/variance per channel per layer). Configs expose `source_stats_path` / `statistic_save_path` / `clean_bn_extract_batch` for offline collection. Train-time `fit()` is responsible for that collection step — keep it idempotent.
- Layer-targeting is regex-based via `cascade_target` (a `list[str]` of regex patterns). Each method ships a `TargetKeyPreset` enum keyed by backbone (`RESNET`, `SWIN(T)`, `C3K2`/`YOLO11_*`); `from_preset(base_model)` selects the right preset + masking defaults.
- `mask_value=114` (YOLO11 gray padding) plus `masked_processing=True` is the YOLO11 default for input-side cascade methods; do not change without coordinating with `models/yolo11/`.
- The `exclude_target` default (`["stem", "patch_embed", "embedder"]`) intentionally skips token-embedder layers across all three backbone families.

### Testing Requirements
- These methods compile a per-method statistic tensor (sometimes large). Verify `source_stats_path` round-trips via `torch.save` / `torch.load` after changes.
- Smoke-test on `SHIFTContinuousSubsetForObjectDetection` for at least one backbone.

### Common Patterns
- `use_kl_divergence: bool = True` (with MSE fallback when False) is the canonical alignment-loss switch in this group.
- ActMAD and friends use `loss_type: str = "L1"` for the alignment objective; CascadedNorm/FlowAdaptation/GITA use KL by default.

## Dependencies

### Internal
- `..base` — `AdaptationConfig`, `AdaptationEngine`.
- `....models` (for `from_preset` dispatch on detector classes — imported lazily inside the classmethod to avoid circular imports).

### External
- `torch.nn`, `torch.nn.functional`, `transformers.AutoConfig`.

<!-- MANUAL: -->

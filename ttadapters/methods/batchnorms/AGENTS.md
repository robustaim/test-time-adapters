<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# batchnorms

## Purpose
**Batch-statistics replacement** TTA — adapt at test time by recomputing or interpolating BatchNorm running statistics (mean, variance) on the target stream rather than updating any learnable parameters. Two paradigms live here: **covariate-shift** (`DUA` — exponential decay of momentum) and **dynamic** (`NORM` — recompute from accumulated source samples).

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | `from .covariate.dua import *` and `from .dynamic.norm import *`. Wildcard re-exports propagate to `ttadapters.methods`. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `covariate/` | DUA-family methods that adjust BN momentum based on covariate shift (see [covariate/AGENTS.md](covariate/AGENTS.md)) |
| `dynamic/` | NORM-family methods that dynamically replace BN statistics (see [dynamic/AGENTS.md](dynamic/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- All batchnorm methods only mutate BN running stats — `online_parameters()` should return an **empty** iterator for these, so the optimizer is essentially unused. Configs still inherit `optim`/`adapt_lr` from `AdaptationConfig` for uniformity but they're inert.
- New BN method families should add a new sibling folder under `batchnorms/` (not under `covariate/` or `dynamic/`).
- Both DUA and NORM use the per-backbone presets `adaptation_layers="backbone"` for Faster R-CNN / YOLO11 and `"backbone+encoder"` for RT-DETR.

### Testing Requirements
- BN-only methods are cheap; smoke-test by running a SHIFT continuous subset and verifying mAP differs from the source-only baseline.

### Common Patterns
- The `from_preset` branches always switch on `isinstance(base_model, ...)` for the four detector classes (`FasterRCNN`, `RTDetr`, `YOLO11`); SwinRCNN is intentionally NOT supported because Swin uses LayerNorm rather than BatchNorm.

## Dependencies

### Internal
- `..base` — `AdaptationConfig`, `AdaptationEngine`.

### External
- `torch.nn.BatchNorm2d`, `transformers.AutoConfig`.

<!-- MANUAL: -->

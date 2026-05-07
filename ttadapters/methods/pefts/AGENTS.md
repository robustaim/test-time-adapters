<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# pefts

## Purpose
**Parameter-Efficient Fine-Tuning** TTA — adapts at test time by training only a small fraction of parameters (added adapters or surviving channels), keeping the rest of the source model frozen. Two paradigms live here: **low-rank** (adapter-style, e.g. WHW from "When, Where, and How to Adapt?") and **pruning** (sensitivity-guided, e.g. SGP).

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | `from .low_rank.whw import *` and `from .pruning.sgp import *`. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `low_rank/` | Low-rank adapter PEFT TTA (WHW) (see [low_rank/AGENTS.md](low_rank/AGENTS.md)) |
| `pruning/` | Sensitivity-guided pruning PEFT TTA (SGP, CVPR 2025) (see [pruning/AGENTS.md](pruning/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- PEFT engines override `online_parameters()` to expose only the trainable subset (adapter weights for WHW, surviving BN affines for SGP). The optimizer step otherwise touches frozen weights.
- Both methods support `from_preset(FasterRCNNForObjectDetection | SwinRCNNForObjectDetection)`. **SGP does NOT support Swin** because Swin uses LayerNorm, not BatchNorm — `from_preset(SwinRCNN)` raises `NotImplementedError`. Match this restriction when adding pruning-based methods.
- WHW exposes a `WHWSkipConfig` / `WHWSkipEngine` pair that adds redundancy-skip logic (`skip_redundant`, `skip_tau`, `skip_period`, `skip_beta`). When adding sibling skip-variants, follow the same naming pattern.

### Testing Requirements
- Verify trainable-param ratio after instantiation: `sum(p.numel() for p in engine.online_parameters() if p.requires_grad) / sum(p.numel() for p in engine.parameters())` should match the documented PEFT ratio (e.g. WHW `adapter_ratio=32` → ~3% trainable).
- Smoke-test on Faster R-CNN + SHIFT continuous.

### Common Patterns
- Both WHW and SGP need source statistics (`source_stats_path: str`) — the offline `fit()` step computes and pickles them. Make sure new PEFT methods follow the same statistic-collection convention or document why they don't.
- Both methods default `optim` to a non-SGD choice (`SGD` for WHW, `Adam` for SGP) — pick what's been validated in the upstream paper.

## Dependencies

### Internal
- `..base` — `AdaptationConfig`, `AdaptationEngine`.
- `....models` — `FasterRCNNForObjectDetection`, `SwinRCNNForObjectDetection` (lazily imported inside `from_preset`).

### External
- `torch.nn`, `transformers.AutoConfig`.

<!-- MANUAL: -->

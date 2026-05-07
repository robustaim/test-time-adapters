<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# whw

## Purpose
**WHW** ("When, Where, and How to Adapt?") — TTA via LoRA-style adapter modules inserted into the backbone, optimized at test time with KL/BN-stat alignment plus optional gradient clipping. Exposes both a vanilla `WHWEngine` and a `WHWSkipEngine` variant that adds redundancy-skipping (`stat`/`period`/`ema`) to avoid spending gradient steps on uninformative batches.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Defines `WHWSkipEngine` (subclass of `WHWEngine` with `config_class = WHWSkipConfig`); registers both `WHWConfig`/`WHWEngine` and `WHWSkipConfig`/`WHWSkipEngine` with `transformers.AutoConfig` and `AutoAdaptationEngine{,ForObjectDetection}`. |
| `configuration_whw.py` | `WHWConfig` and `WHWSkipConfig`. Hyperparameters: `backbone ∈ {rcnn, swinrcnn}` (no RT-DETR/YOLO11 support), `optim ∈ {SGD, AdamW}` default `SGD`, `adapt_lr=4e-5` (RCNN) / `9e-6` (SwinRCNN), `weight_decay=1e-4`, `clip_grad_enabled=True`, `clip_grad_type ∈ {value, norm}`, `adapter_ratio=32`, `alpha_gl=1.0`, `alpha_fg=1.0`, `gl_align ∈ {KL, bn_stats, None}`, `fg_align ∈ {KL, None}`, `ema_gamma=128`, `freq_weight=True`, `source_stats_path: str`, `skip_redundant ∈ {stat, period, ema, stat-period-ema, None}`, `skip_tau=1.1`, `skip_period=10`, `skip_beta=1.05`, `collect_iou_thr=0.5`, plus Swin-specific adapter knobs (`adapter_layernorm_option`, `adapter_scalar`, `adapter_init_option`, `adapter_dropout`, `out_batch_norm`, `out_batch_resolution=32`). |
| `modeling_whw.py` | `WHWEngine` — main implementation (~33 KB). Inserts adapters in target modules, runs the alignment loss, and (in the Skip variant) gates gradient steps. |

## For AI Agents

### Working In This Directory
- Only Faster R-CNN and SwinRCNN are supported. `from_preset(RTDetr|YOLO11)` raises — RT-DETR and YOLO11 don't have the matching adapter targets implemented.
- The Skip variant has different defaults: `skip_redundant="stat-period-ema"` and `adapt_lr=5e-4` for RCNN. Don't change those without coordinating with the upstream paper's results.
- `adapter_ratio=32` controls the rank: trainable params ≈ original / 32. Lower it cautiously (more capacity, more overfit risk).
- `source_stats_path` is consumed by `fit()` to load a precomputed statistics tensor — collect it once and cache.

### Testing Requirements
- Validate the trainable-param ratio: ~3% of total `base_model.parameters()` for `adapter_ratio=32`.
- Run a SHIFT continuous subset on Faster R-CNN with both `WHWEngine` and `WHWSkipEngine`; the skip variant should achieve similar mAP with materially fewer optimizer steps.

### Common Patterns
- Reference: "When, Where, and How to Adapt?" (test-time adaptation paper). README marks WHW as `[x]`.
- The Swin adapter uses LayerNorm-aware scaling (`adapter_layernorm_option="in"`); don't change the option string.

## Dependencies

### Internal
- `.....base` — `AdaptationConfig`, `AdaptationEngine`, `AutoAdaptationEngine{,ForObjectDetection}`.
- `......models` — `FasterRCNNForObjectDetection`, `SwinRCNNForObjectDetection` (lazy import in `from_preset`).

### External
- `transformers.AutoConfig`, `torch.nn`, `torch.optim`.

<!-- MANUAL: -->

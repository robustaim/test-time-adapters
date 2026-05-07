<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# actmad

## Purpose
**ActMAD** (Activation Means Adaptation by Discrepancy) — TTA via L1 alignment of every targeted BN's activation mean/std to a pre-computed source-domain statistic. The full backbone (and optionally the encoder) is supervised: the engine collects clean BN statistics over `clean_bn_extract_batch=32` source samples once via `fit()`, then at test time minimizes a per-layer L1 discrepancy between target activations and the saved clean stats.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Registers `ActMADConfig` / `ActMADEngine` with `transformers.AutoConfig`, `AutoAdaptationEngine`, `AutoAdaptationEngineForObjectDetection`. |
| `configuration_actmad.py` | `ActMADConfig`. Hyperparameters: `base_type ∈ {rcnn, swinrcnn, rtdetr, yolo11}`, `adaptation_layers`, `optim ∈ {SGD, AdamW}` (default `SGD`), `adapt_lr=1e-7`, `momentum=0.9`, `weight_decay=1e-4`, `loss_type="L1"`, `statistic_save_path: str | None`, `clean_bn_extract_batch=32`. |
| `modeling_actmad.py` | `ActMADEngine` (~18 KB). Hooks every targeted BN, accumulates mean/std during `fit()`, restores from `statistic_save_path` if given, and at test time computes the discrepancy loss. |

## For AI Agents

### Working In This Directory
- Per-backbone learning rates are highly tuned: `1e-8` (Faster R-CNN), `1e-6` (Swin-RCNN), `1e-7` (RT-DETR), `1e-9` (YOLO11). Don't widen these defaults without re-tuning on the SHIFT continuous benchmark.
- This is the only ActMAD `Engine` exposed under `methods/`. The same name appears under `methods/samplers/active/actmad/` — that's the **sampler** half (active sample selection), used by ActMADEngine internally. They share the algorithm name but live in separate trees by design.
- `fit(source_dataset, batch_size, max_samples=2000, dtype)` is required: it iterates `max_samples` from the source dataset to compute clean BN statistics. Set `statistic_save_path` to cache the result on disk and skip recomputation across runs.

### Testing Requirements
- After `fit()`, verify the saved statistic tensor is non-empty and ungapped (no NaNs from layers that received zero samples).
- Smoke-test on each of the four backbones — ActMAD is the most-tested method in the repo and a regression here is high signal.

### Common Patterns
- README marks ActMAD as `[x]` (Active Learning bullet) and as `[x]` again under "Layer-wise Supervision (Cascaded)". Reference: "ActMAD: Activation Matching for Test-Time Adaptation" (Mirza et al., CVPR 2023).
- The active-sampler counterpart in `methods/samplers/active/actmad/` is invoked from inside `ActMADEngine` to gate which batches contribute to the loss.

## Dependencies

### Internal
- `.....base` — `AdaptationConfig`, `AdaptationEngine`.
- `......models` — all four detector classes (lazy import).
- `......samplers.active.actmad` — active sampler counterpart.

### External
- `transformers.AutoConfig`, `torch.nn`, `torch.optim`.

<!-- MANUAL: -->

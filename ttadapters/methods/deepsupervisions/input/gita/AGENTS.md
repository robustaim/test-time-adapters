<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# gita

## Purpose
**GITA** — TTA via an Input Transformation Module (ITM) plus KL-divergence alignment to early-block normalization statistics. Compared to CascadedNorm, GITA targets a *single early region* per backbone (e.g. ResNet `res2`, Swin `layer0–layer1` blocks 0/1, YOLO11 `model.[246]`'s C3k2 shortcut/bottleneck-final BNs, or just the YOLO11 stem `model.0`). The differentiable gamma/CLAHE ITM is fully exposed via `gamma_temperature`, `gamma_range`, `gamma_noise_floor`, `gamma_saturation_limit`, and `clahe_*` parameters.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Registers `GITAConfig` / `GITAEngine` with `transformers.AutoConfig`, `AutoAdaptationEngine`, `AutoAdaptationEngineForObjectDetection`. (Unlike CascadedNorm and FlowAdaptation, GITA *is* HF-registered.) |
| `configuration_gita.py` | `GITAConfig` + `TargetKeyPreset` enum: `RESNET`, `SWIN`, `YOLO11_C3K2`, `YOLO11_STEM`, `YOLO11_FIRST_STRIDE`, `YOLO11`. Hyperparameters: `optim="Adam"`, `adapt_lr=1e-3`, `itm_type="gamma"`, `cascade_target: list[str]`, `disable_blending=False`, `blend_ratio=0.6`, `mask_value=114`, `masked_processing=False`, full CLAHE/gamma knobs, `use_kl_divergence=True`, `force_use_feature_stat=False`. |
| `modeling_gita.py` | `GITAEngine` — main implementation (~30 KB). |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `ablations/` | Ablation studies (AB1–AB2 + AAB1–AAB2 follow-ups) (see [ablations/AGENTS.md](ablations/AGENTS.md)) |
| `component_anal/` | Component-level analyses (CA1, CA2) (see [component_anal/AGENTS.md](component_anal/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- `from_preset(YOLO11ForObjectDetection)` defaults to **`YOLO11_STEM`** (just `model.0.bn`) — the most aggressive variant. Override with `cascade_target=TargetKeyPreset.YOLO11_C3K2.value` (or `YOLO11`, `YOLO11_FIRST_STRIDE`) when running ablations.
- `gamma_temperature=0.01` controls the soft gamma's sharpness; `gamma_range=(0.5, 2.0)` clamps the multiplicative range to "/2 to ×2".
- `force_use_feature_stat` bypasses BN running stats and computes statistics from the live feature batch — useful when source stats are unreliable.
- The `ablations/` and `component_anal/` subdirectories contain *experimental copies* of `modeling_gita.py` plus notebooks; only the top-level `modeling_gita.py` is the canonical implementation.

### Testing Requirements
- For YOLO11, smoke-test each of the four presets (`YOLO11`, `YOLO11_C3K2`, `YOLO11_STEM`, `YOLO11_FIRST_STRIDE`) — they exercise different regex matchers.
- Verify that gamma updates are bounded by `gamma_range` after several adaptation steps.

### Common Patterns
- KL-divergence alignment is the standard loss; switching `use_kl_divergence=False` falls back to MSE.
- The ITM blends original and transformed inputs by `blend_ratio` (`0.6` default) when `disable_blending=False`.

## Dependencies

### Internal
- `....base`, `.....models` (lazy import in `from_preset`).

### External
- `transformers.AutoConfig`, `torch.nn`, `torch.nn.functional`.

<!-- MANUAL: -->

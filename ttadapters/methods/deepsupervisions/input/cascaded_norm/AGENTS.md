<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# cascaded_norm

## Purpose
**CascadedNorm** — TTA via an Input Transformation Module (ITM) plus cascaded normalization alignment in the **early backbone blocks**. Five intra-block strategies are exposed (S0–S4 implied by the regex presets) corresponding to which BN/LN within each block participates: `S1` first-only (Source Anchor), `S2` last-only (Target Anchor), `S3` Proximal (first + immediate next), `S4` Distal (last + immediate prev). Note: `cascaded_norm` is **not** auto-registered — it's imported directly when used (no `transformers.AutoConfig.register` call in its `__init__.py`).

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Plain `from .configuration_cascaded_norm import CascadedNormConfig, TargetKeyPreset, TARGET_KEY_PRESET; from .modeling_cascaded_norm import CascadedNormEngine`. No HF Auto registration. |
| `configuration_cascaded_norm.py` | `CascadedNormConfig` (~5 KB) + `TargetKeyPreset` enum with `RESNET`, `RESNET_S1..S4`, `SWINT`, `SWINT_S1`, `SWINT_S2`, `SWINT_S3=SWINT_S4=SWINT` (Swin only has 2 LNs per block), `C3K2`, `C3K2_S1..S4`. Hyperparameters: `itm_type ∈ {clahe, gamma, clahe-gamma}`, `itm_combination_method ∈ {residual, hierarchical, frequency, None}`, `cascade_target: list[str]`, `exclude_target=["stem", "patch_embed", "embedder"]`, `use_kl_divergence=True`, `use_feature_alignment=False`, `use_bn_running_stat=False`. |
| `modeling_cascaded_norm.py` | `CascadedNormEngine` — the heart of the method (~38 KB). Implements ITM construction, strategy-aware hooks, and the alignment loss. |

## For AI Agents

### Working In This Directory
- `from_preset` defaults to **strategy S3 (Proximal)** for every backbone (`RESNET_S3`, `SWINT_S3`, `C3K2_S3`). S3 was the published choice; non-default strategies are exposed for ablation studies.
- For YOLO11, `from_preset` also flips `masked_processing=True` and `mask_value=114` to handle the gray padding from letterboxing.
- `itm_combination_method` is only meaningful when `itm_type == "clahe-gamma"` — the literal `"residual"`, `"hierarchical"`, `"frequency"` decide how CLAHE and gamma outputs are fused. Frequency mode uses `frequency_combination_kernel_size=3`, `frequency_combination_sigma=1.0`.
- Because `CascadedNormConfig` is not registered with `AutoAdaptationEngine`, callers must construct it directly: `CascadedNormConfig(...) → CascadedNormEngine(config, base_model)`.

### Testing Requirements
- Verify all five strategies for at least one backbone family run a full step without shape mismatches.
- Confirm gamma parameters are clamped to `gamma_range=(0.5, 2.0)` after backward steps.

### Common Patterns
- README marks the entire CascadedNorm section as `[x]` and emphasizes "self-supervised, layer-wise, cascaded, not end-to-end" — preserve that framing in any narrative additions.
- The S3=S4=ALL collapse on Swin is a load-bearing detail: Swin blocks only have `norm1` and `norm2`, so Proximal/Distal can't be distinguished.

## Dependencies

### Internal
- `....base` — `AdaptationConfig`.
- `.....models` — all four detector classes (lazy import in `from_preset`).

### External
- `torch.nn`, `torch.nn.functional`, OpenCV/`kornia` for differentiable CLAHE (likely).

<!-- MANUAL: -->

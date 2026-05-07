<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# flow_adaptation

## Purpose
**FlowAdaptation** — TTA via an Input Transformation Module (ITM) coupled with **per-block** alignment in the early backbone (early 5 blocks per the docstring). Differs from CascadedNorm in that targets are *whole BottleneckBlocks/SwinBlocks/C3k2 bottlenecks* rather than individual norms inside them, and exposes a `reduce_dim` knob to control which dimensions are reduced when computing per-channel statistics.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Plain `from .configuration_flow_adaptation import FlowAdaptationConfig; from .modeling_flow_adaptation import FlowAdaptationEngine`. No HF Auto registration. |
| `configuration_flow_adaptation.py` | `FlowAdaptationConfig` (~3 KB) + `TargetKeyPreset` enum with `RESNET` (`res[23].[012]` / `stages.[01].layers.[012]`), `SWIN` (`layers.[01].blocks.[012]`), `C3K2` (`model.[24].m.0.m.[01]`). Hyperparameters: `itm_type ∈ {clahe, gamma, clahe-gamma, clahe-gamma-residual}`, `disable_blending=False`, `mask_value=114`, `masked_processing=False`, `use_kl_divergence=True`, `reduce_dim: tuple[int, ...] | None`. |
| `modeling_flow_adaptation.py` | `FlowAdaptationEngine` (~27 KB). |

## For AI Agents

### Working In This Directory
- `reduce_dim` is critical and backbone-specific: `(0, 2, 3)` for ResNet/RT-DETR/YOLO11 (BCHW → reduce over batch + spatial), `(0, 1, 2)` for Swin (BHWC tokens → reduce over batch + sequence). Don't transpose this convention.
- For YOLO11, `from_preset` flips `masked_processing=True` and `mask_value=114` like the other input methods.
- Like CascadedNorm and unlike GITA, this method is NOT registered with `AutoAdaptationEngine` — instantiate directly.
- The `gamma_noise_floor=2.0` and `gamma_saturation_limit=98.0` defaults differ from GITA's `0.0`/`100.0`. The non-zero noise floor was tuned to avoid degenerate gamma updates on near-black pixels.

### Testing Requirements
- Smoke-test on each of the four backbones, verifying `reduce_dim` matches the input layout actually produced by the targeted block.
- Confirm the ITM remains differentiable across CLAHE blocks (CLAHE in CV is not naturally differentiable; check the implementation).

### Common Patterns
- README does NOT explicitly mark FlowAdaptation as `[x]`; it appears between CascadedNorm and GITA in the cascaded family. Treat it as research-grade.
- Differs from CascadedNorm by targeting **whole blocks**, not individual norms. Useful when block-level statistics are more stable than intra-block ones.

## Dependencies

### Internal
- `....base` — `AdaptationConfig`.
- `.....models` — all four detector classes (lazy import in `from_preset`).

### External
- `torch.nn`, `torch.nn.functional`.

<!-- MANUAL: -->

<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# input

## Purpose
**Input-level** deep supervision — TTA via an Input Transformation Module (ITM) and early-layer cascade alignment. These methods learn to *normalize the input* (gamma correction, CLAHE, residual blending) at test time so that the backbone's early layers see distribution-aligned activations. Three concrete methods live here: CascadedNorm, FlowAdaptation, and GITA — each represents a different framing of "learn to normalize the target image so the source backbone behaves correctly".

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `cascaded_norm/` | CascadedNorm: ITM + cascaded BN/LN alignment with strategy presets S1–S4 (see [cascaded_norm/AGENTS.md](cascaded_norm/AGENTS.md)) |
| `flow_adaptation/` | FlowAdaptation: ITM + per-block alignment (RES/SWIN/C3K2) (see [flow_adaptation/AGENTS.md](flow_adaptation/AGENTS.md)) |
| `gita/` | GITA: ITM + KL-divergence alignment to early-block stats, with extensive ablations (see [gita/AGENTS.md](gita/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- All three methods share the ITM concept: `itm_type ∈ {clahe, gamma, clahe-gamma, clahe-gamma-residual}`. The transformation is differentiable so the ITM parameters are updated at test time.
- All three use a `TargetKeyPreset` enum with `RESNET` / `SWIN(T)` / `C3K2` regex variants for the four detector backbones. Strategy variants (`_S1` through `_S4`) refine which BNs/LNs within early blocks are aligned.
- `disable_blending`, `blend_ratio`, and `mask_value` are the canonical knobs for handling padded regions (especially YOLO11's 114-gray padding).
- `use_kl_divergence=True` is the default alignment loss; set False for MSE on the per-channel mean/std vectors.

### Testing Requirements
- Verify the ITM is differentiable: `engine.itm.parameters()` should produce trainable tensors that move after a few steps.
- Visualize an ITM-transformed image vs the input to confirm sensible gamma/CLAHE behavior.

### Common Patterns
- The early-block emphasis: every preset's `S1`/`S2`/`S3`/`S4` keys target the *first two stages* of the backbone (e.g. ResNet `res2 + res3`, Swin `layer0 + layer1`, YOLO11 `model.[24]`). Don't add presets that hit the deeper layers — that's `full/`'s job.
- `exclude_target` defaults to `["stem", "patch_embed", "embedder"]` to skip token embedders that aren't meaningfully "normalized" by an ITM.

## Dependencies

### Internal
- `....base`, `.....models` (lazy import in `from_preset`).

### External
- `torch.nn`, `torch.nn.functional`, `transformers.AutoConfig`.

<!-- MANUAL: -->

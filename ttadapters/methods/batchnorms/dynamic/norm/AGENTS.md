<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# norm

## Purpose
**NORM** — TTA via direct batch-statistics replacement. Aggregates BN statistics over a fixed number of source samples (`source_sum=128`) and uses those (or a blend) to override the running mean/variance of every selected BN layer. No gradient updates; useful as a strong baseline for covariate-shift problems.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Registers `NORMConfig` / `NORMEngine` with `transformers.AutoConfig`, `AutoAdaptationEngine`, `AutoAdaptationEngineForObjectDetection`. |
| `configuration_norm.py` | `NORMConfig(AdaptationConfig)`. Hyperparameters: `base_type ∈ {rcnn, rtdetr, yolo11}`, `source_sum=128`, `adaptation_layers ∈ {backbone, encoder, backbone+encoder}`. SwinRCNN is **not** supported. |
| `modeling_norm.py` | `NORMEngine` (~11 KB). Iterates submodules to replace BN running stats. |

## For AI Agents

### Working In This Directory
- `source_sum=128` is the count of *source-domain samples* whose statistics get accumulated as the new BN running stats. Reducing it speeds up adaptation but increases variance — only change with empirical justification.
- Like DUA, NORM has no learnable parameters; `online_parameters()` is effectively empty.
- The `from_preset(base_model)` defaults match DUA's: `"backbone"` for Faster R-CNN / YOLO11, `"backbone+encoder"` for RT-DETR.

### Testing Requirements
- Smoke-test on each of the three supported backbones using `SHIFTContinuousSubsetForObjectDetection`.

### Common Patterns
- README marks NORM as `[x]` (implemented). Reference: "Improving Robustness Against Common Corruptions by Covariate Shift Adaptation" (Schneider et al., NeurIPS 2020).

## Dependencies

### Internal
- `.....base`, `......models` (lazy import in `from_preset`).

### External
- `transformers.AutoConfig`, `torch.nn.BatchNorm2d`.

<!-- MANUAL: -->

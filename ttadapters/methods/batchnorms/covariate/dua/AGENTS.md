<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# dua

## Purpose
**DUA** (Dynamic Unsupervised Adaptation) — TTA via decaying BN momentum. Starting from a high momentum (`mom_pre`), the running statistics are progressively updated with each test batch using a momentum that decays by `decay_factor` per step but is floored at `min_momentum_constant`. No learnable parameters change.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Registers `DUAConfig` / `DUAEngine` with `transformers.AutoConfig`, `AutoAdaptationEngine`, and `AutoAdaptationEngineForObjectDetection`. |
| `configuration_dua.py` | `DUAConfig(AdaptationConfig)`. Hyperparameters: `min_momentum_constant=0.0001`, `decay_factor=0.94`, `mom_pre=0.01`, `base_type ∈ {rcnn, rtdetr, yolo11}`, `adaptation_layers ∈ {backbone, encoder, backbone+encoder}`. SwinRCNN is **not** supported (no BN). |
| `modeling_dua.py` | `DUAEngine` (~7 KB). Walks the target submodules selected by `adaptation_layers`, registers BN-momentum decay logic, and overrides `online_parameters()` to be effectively empty. |

## For AI Agents

### Working In This Directory
- DUA only mutates BN momentum and running stats — no gradient updates. The optimizer is constructed but its `step()` is a no-op for these layers.
- `from_preset(base_model)` returns `adaptation_layers="backbone"` for Faster R-CNN / YOLO11 and `"backbone+encoder"` for RT-DETR. Don't change those defaults; they were tuned for the SHIFT continuous benchmark.
- Keep momentum decay deterministic: avoid randomness in the per-step momentum update so that `MethodContainer.go_rounds` can produce reproducible trials.

### Testing Requirements
- Smoke-test on each of the three supported backbones using `SHIFTContinuousSubsetForObjectDetection`.
- Verify that `engine.online_parameters()` yields zero trainable params: `assert not list(engine.online_parameters())`.

### Common Patterns
- The README marks DUA as `[x]` (implemented). Reference paper: "DUA — The Norm Must Go On" (Mirza et al., CVPR 2022).

## Dependencies

### Internal
- `.....base` — `AdaptationConfig`, `AdaptationEngine`, `AutoAdaptationEngine{,ForObjectDetection}`.
- `......models` — detector classes (lazily imported in `from_preset`).

### External
- `transformers.AutoConfig`, `torch.nn.BatchNorm2d`.

<!-- MANUAL: -->

<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# sgp

## Purpose
**SGP** (Sensitivity-Guided Pruning) — TTA via channel pruning of domain-sensitive BN channels (using BatchNorm2d γ scaling factors) plus feature alignment for the remaining domain-invariant ones. From "Efficient Test-time Adaptive Object Detection via Sensitivity-Guided Pruning" (CVPR 2025). Includes instance-level (RoI-based) sensitivity to prioritize foreground regions and stochastic channel reactivation to avoid permanent dead channels.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Registers `SGPConfig` / `SGPEngine` with HF Auto classes. |
| `configuration_sgp.py` | `SGPConfig` (~3 KB) + `ExcludeLayerPreset` enum (`RESNET_FPN = [r"\.res5\."]` to keep the last stage out of pruning). Hyperparameters: `optim="Adam"`, `adapt_lr=5e-3`, `momentum=0.9`, `pruning_rate=0.10`, `pruning_threshold=0.05` (\|γ\| < threshold → prune), `lambda_align=1.0`, `lambda_mu_std=1.0`, `lambda_sparse=0.05`, `reactivation_prob=0.01`, `use_instance_sensitivity=True`, `roi_output_size=7`, `fg_confidence_threshold=0.5`, `exclude_layers: list[str] | None`, `source_stats_path: str | None`. |
| `modeling_sgp.py` | `SGPEngine` — main implementation (~26 KB). Pruning + alignment + RoI-Align based instance sensitivity. |

## For AI Agents

### Working In This Directory
- **SGP only supports Faster R-CNN (ResNet-FPN with BN).** `from_preset(SwinRCNN | RTDetr | YOLO11)` raises `NotImplementedError` because Swin uses LayerNorm and the others use different backbone layouts.
- `RESNET_FPN` excludes `res5` (last stage) — preserving the most task-specific features. Don't add new layers to `exclude_layers` without justification; over-excluding kills the pruning advantage.
- `pruning_rate=0.10` (10%) and `pruning_threshold=0.05` are tightly coupled. Increase one and you should re-tune the other.
- `reactivation_prob=0.01` (Bernoulli per-channel) is what keeps channels from being permanently dead — keep this nonzero.
- `lambda_sparse=0.05` is the L_wreg weighted-sparsity coefficient. The rest (`lambda_align`, `lambda_mu_std`) control L_adp.

### Testing Requirements
- After running adaptation for ~50 batches, verify the actual fraction of pruned channels matches `pruning_rate` within ±1%.
- Smoke-test: `mAP_with_SGP > mAP_source_only` on a SHIFT continuous Faster R-CNN run.

### Common Patterns
- Pruning is implemented via masking γ→0 (and rescaling β) rather than physically removing channels — this keeps the backbone shape unchanged so the rest of the detector keeps working.
- `use_instance_sensitivity=True` activates the RoI-Align path (output spatial size = `roi_output_size=7`) and weights pruning by per-instance sensitivity.

## Dependencies

### Internal
- `.....base` — `AdaptationConfig`, `AdaptationEngine`, `AutoAdaptationEngine{,ForObjectDetection}`.
- `......models` — `FasterRCNNForObjectDetection` (lazy import in `from_preset`).

### External
- `transformers.AutoConfig`, `torch.nn.BatchNorm2d`, `torchvision.ops.roi_align`.

<!-- MANUAL: -->

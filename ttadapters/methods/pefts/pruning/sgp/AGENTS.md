<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-07-08 -->

# sgp

## Purpose
**SGP** (Sensitivity-Guided Pruning) — TTA via channel pruning of domain-sensitive BN channels (using BatchNorm2d γ scaling factors) plus feature alignment for the remaining domain-invariant ones. From "Efficient Test-time Adaptive Object Detection via Sensitivity-Guided Pruning" (CVPR 2025). Includes instance-level (RoI-based) sensitivity to prioritize foreground regions and stochastic channel reactivation to avoid permanent dead channels.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Registers `SGPConfig` / `SGPEngine` with HF Auto classes. |
| `configuration_sgp.py` | `SGPConfig` + `ExcludeLayerPreset` enum. `RESNET_FPN = [r"\.res5\.", r"\.conv3\.norm"]` — excludes the last stage **and** R50-Bottleneck `conv3.norm` (whose bimodal γ distribution poisons the paper's threshold-based prune-rate math; R50-specific since the paper used R18-BasicBlock). Hyperparameters: `optim="Adam"`, `adapt_lr=5e-5` (paper 5e-3, 100× lower for R50+B=1 online TTA), `pruning_rate=0.10`, `pruning_threshold=0.05` (γ < threshold → prune), `lambda_align=1.0`, `lambda_sparse=0.05` (L_wreg weight, Eq.13), `target_ema_momentum=0.5` (EMA α for μ_t in L_adp KL), `source_var_floor=1e-4` (σ²_s division floor), `reactivation_prob=0.01`, `use_instance_sensitivity=True`, `roi_output_size=7`, `fg_confidence_threshold=0.5`, `exclude_layers: list[str] | None`, `source_stats_path: str | None`. |
| `modeling_sgp.py` | `SGPEngine` — main implementation (~26 KB). Pruning + alignment + RoI-Align based instance sensitivity. |

## For AI Agents

### Working In This Directory
- **SGP only supports Faster R-CNN (ResNet-FPN with BN).** `from_preset(SwinRCNN | RTDetr | YOLO11)` raises `NotImplementedError` because Swin uses LayerNorm and the others use different backbone layouts.
- `RESNET_FPN` excludes both `res5` (last stage, task-specific features) and `conv3.norm` (R50-Bottleneck's bimodal-γ layers that break the paper's prune-rate estimation). Both exclusions are load-bearing and documented inline in `ExcludeLayerPreset` — don't drop them or add new layers without the same level of justification; over-excluding kills the pruning advantage.
- `pruning_rate=0.10` (10%) and `pruning_threshold=0.05` are tightly coupled. Increase one and you should re-tune the other.
- `reactivation_prob=0.01` (Bernoulli per-channel) is what keeps channels from being permanently dead — keep this nonzero.
- `lambda_sparse=0.05` is the L_wreg weighted-sparsity coefficient (only applied while prune-rate ρ < `pruning_rate`; dropped once the target is reached, per Eq.13). `lambda_align`, `target_ema_momentum`, and `source_var_floor` control the L_adp KL alignment term.
- Many defaults deliberately deviate from the paper (R18/B=4 → R50-FPN/B=1 online); each such field documents its paper value inline in `configuration_sgp.py`. Read those comments before re-tuning.

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

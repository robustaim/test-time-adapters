<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# pruning

## Purpose
**Pruning-based** PEFT TTA — adapt at test time by *removing* domain-sensitive channels (BN scaling-factor pruning) and updating only the remaining domain-invariant ones via feature alignment. The single concrete method here is SGP (CVPR 2025).

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `sgp/` | SGP: Sensitivity-Guided Pruning with weighted sparsity regularization (see [sgp/AGENTS.md](sgp/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- All pruning methods here target `nn.BatchNorm2d` scaling factors (γ). LayerNorm-only backbones (Swin) are not supported.
- New pruning methods should reuse the `pruning_rate` / `pruning_threshold` / `lambda_sparse` knob convention exposed by SGP.

### Testing Requirements
- See `sgp/AGENTS.md`.

## Dependencies

### Internal
- `....base`.

### External
- `torch.nn.BatchNorm2d`.

<!-- MANUAL: -->

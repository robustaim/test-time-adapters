<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# dynamic

## Purpose
Dynamic batchnorm-replacement methods — recompute BN statistics from a buffer of source samples (or a moving window of target samples), replacing the frozen source running stats entirely.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `norm/` | NORM: Bridging the source and target via batch statistics replacement (see [norm/AGENTS.md](norm/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- The single concrete method here is NORM. Add new dynamic-stat methods as sibling folders.
- "Dynamic" here means "recomputed each step" rather than "interpolated over time" — the latter belongs in `covariate/`.

### Testing Requirements
- See `norm/AGENTS.md`.

## Dependencies

### Internal
- `....base` — `AdaptationConfig`, `AdaptationEngine`.

### External
- `torch.nn.BatchNorm2d`.

<!-- MANUAL: -->

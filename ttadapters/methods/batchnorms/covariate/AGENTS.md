<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# covariate

## Purpose
Covariate-shift batchnorm methods — adjust BN momentum dynamically as a function of how much the input distribution has drifted from source.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `dua/` | DUA: Dynamic Unsupervised Adaptation via momentum decay (see [dua/AGENTS.md](dua/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- The single concrete method here is DUA. New "covariate-aware" BN approaches should add a sibling folder under `covariate/` rather than replacing DUA.

### Testing Requirements
- See `dua/AGENTS.md`.

### Common Patterns
- All methods in this group only mutate BN momentum and running stats — no learnable parameters update.

## Dependencies

### Internal
- `....base` — `AdaptationConfig`, `AdaptationEngine`.

### External
- `torch.nn.BatchNorm2d`.

<!-- MANUAL: -->

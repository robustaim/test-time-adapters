<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# full

## Purpose
**Full-network** deep supervision — TTA methods that align *all* (or many) intermediate feature statistics in the backbone to a pre-collected source distribution. The single concrete method here is ActMAD (Active Means Activation Discrepancy).

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `actmad/` | ActMAD: full-backbone activation-statistic alignment with active sample selection (see [actmad/AGENTS.md](actmad/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- "Full" here means alignment is applied at *every* targeted layer, not just early ones. Methods that target only early/input layers belong in `../input/` or `../local/`.

### Testing Requirements
- See `actmad/AGENTS.md`.

## Dependencies

### Internal
- `....base`.

### External
- `torch.nn`.

<!-- MANUAL: -->

<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# active

## Purpose
**Active-learning** sample selection for TTA — score incoming test samples by informativeness (uncertainty / disagreement / activation discrepancy) so that adaptation only spends gradient steps on the batches that move the model. Currently only the ActMAD active sampler lives here.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `actmad/` | ActMAD active sampler — pairs with `methods/deepsupervisions/full/actmad/` (see [actmad/AGENTS.md](actmad/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- Active samplers are *not* `AdaptationEngine`s — they are helpers consumed by another method's engine. They don't follow the Auto-registration triplet.
- Add new active strategies as sibling folders. Keep the API compatible with the existing usage in `methods/deepsupervisions/full/actmad/modeling_actmad.py`.

### Testing Requirements
- See `actmad/AGENTS.md`.

## Dependencies

### Internal
- Imported by full-network adaptation engines (notably ActMADEngine).

### External
- `torch`.

<!-- MANUAL: -->

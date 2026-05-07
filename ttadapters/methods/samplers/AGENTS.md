<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# samplers

## Purpose
**Sample-selection** strategies for TTA — at test time, decide *which* incoming samples to use for adaptation rather than blindly adapting on every batch. Currently houses ActMAD-style **active learning** sample scoring. Note this directory has no `__init__.py` (intentionally not wildcard-exported from the parent `methods/__init__.py`).

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `active/` | Active-learning samplers (ActMAD) (see [active/AGENTS.md](active/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- Samplers are **helpers**, not full `AdaptationEngine`s — they're meant to be composed with another method (e.g. plug an active sampler into ActMAD's adaptation step to skip uninformative batches).
- Samplers don't follow the Auto-registration triplet because they're not standalone configs/engines. They expose plain functions or small classes to be imported by other methods.
- This is why the parent `methods/__init__.py` does NOT include `from .samplers import *`.

### Testing Requirements
- Sampler correctness is best validated via the parent method's evaluation: confirm that `mAP_with_sampler >= mAP_without_sampler - small_epsilon` on a SHIFT subset.

### Common Patterns
- Sampler API is loose; current convention is a callable `score(batch) -> Tensor` that returns a per-sample score for downstream gating.

## Dependencies

### Internal
- Imported directly from method engines (notably the ActMAD engine in `methods/deepsupervisions/full/actmad/`).

### External
- `torch`.

<!-- MANUAL: -->

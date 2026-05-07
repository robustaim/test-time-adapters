<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# local

## Purpose
**Local** deep supervision — TTA via *region-restricted* alignment (only object regions, RoIs, or salient pixels contribute to the loss). Currently only ObjectNorm lives here, and it's an empty placeholder.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `objectnorm/` | ObjectNorm: per-object normalization alignment — empty stub (see [objectnorm/AGENTS.md](objectnorm/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- "Local" supervision distinguishes itself from `full/` (every layer) and `input/` (input transformation + early layers) by *spatially* restricting the loss to detected object regions.
- Not currently re-exported from `methods/deepsupervisions/__init__.py` — when implementing, add `from .local.objectnorm import *` there.

### Testing Requirements
- Once implemented, the test should verify that the loss is computed *only* over RoI regions (zero-out background pixels and confirm gradient magnitude doesn't change for background-only methods).

## Dependencies

### Internal
- Will depend on `methods.base.AdaptationEngine` and detector RoI heads.

### External
- TBD per method (likely `torchvision.ops.roi_align`).

<!-- MANUAL: -->

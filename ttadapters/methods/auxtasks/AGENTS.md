<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# auxtasks

## Purpose
Auxiliary-task TTA methods — adapting a model at test time by minimizing the loss of a self-supervised auxiliary head (rotation prediction, contrastive learning, masked pixel prediction). Currently a **stub package** with no concrete methods implemented; subdirectories contain only empty Python files awaiting implementation.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Empty — no public exports yet. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `ttt/` | TTT family (rotation prediction, TTT++, Masked TTT) — empty stubs (see [ttt/AGENTS.md](ttt/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- Nothing here is exported. When implementing a method, place it in `ttt/` (or a new family folder), follow the standard triplet (`configuration_*.py`, `modeling_*.py`, leaf `__init__.py` registers Auto classes), and add wildcard imports up the chain.
- The README's checklist marks the entire auxtasks bullet as `[ ]` (none implemented).

### Testing Requirements
- Once a method is implemented, add the standard `from ttadapters.methods import *` smoke check.

### Common Patterns
- Auxiliary-task methods typically subclass `AdaptationEngine` and override `forward` to also run a self-supervised head, then add the auxiliary loss to the optimizer step.

## Dependencies

### Internal
- Will depend on `methods.base.AdaptationEngine` once implemented.

### External
- TBD per method.

<!-- MANUAL: -->

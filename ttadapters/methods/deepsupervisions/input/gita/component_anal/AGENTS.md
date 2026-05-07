<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# component_anal

## Purpose
GITA **component-level analyses** — each subfolder studies the contribution of a particular GITA component (ITM type, alignment loss, target preset, masking, etc.) via a single notebook. No frozen model code here; analyses run against the canonical GITA implementation.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `ca1/` | Component analysis 1 — `analysis1.ipynb` (see [ca1/AGENTS.md](ca1/AGENTS.md)) |
| `ca2/` | Component analysis 2 — `analysis2.ipynb` (see [ca2/AGENTS.md](ca2/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- Like `ablations/`, these are research artifacts and not part of the runtime import path.
- Naming convention: `ca<N>/analysis<N>.ipynb`.

### Testing Requirements
- Not part of CI.

## Dependencies

### External
- Jupyter, `ttadapters` runtime stack.

<!-- MANUAL: -->

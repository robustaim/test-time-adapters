<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# ab1

## Purpose
GITA Ablation 1 — frozen snapshot of `modeling_gita.py` (~20 KB) plus the driver notebook `ablation1.ipynb`. Captures one variant of GITA's design space.

## Key Files
| File | Description |
|------|-------------|
| `ablation1.ipynb` | Driver notebook (~17 KB) that loads the local `modeling_gita.py` and runs the ablation. |
| `modeling_gita.py` | Frozen GITA engine variant (~20 KB) — superseded by the canonical `../../modeling_gita.py`. |

## For AI Agents

### Working In This Directory
- Treat as read-only research artifacts. If you need to update the canonical implementation, edit `methods/deepsupervisions/input/gita/modeling_gita.py` instead and let this snapshot diverge.
- The notebook adjusts `sys.path` (or otherwise loads the local file) so changes to `modeling_gita.py` here only affect this ablation.

### Testing Requirements
- Not part of CI. Reproducing the ablation requires the SHIFT dataset symlinked at the repo root.

## Dependencies

### External
- Jupyter, `ttadapters` runtime stack.

<!-- MANUAL: -->

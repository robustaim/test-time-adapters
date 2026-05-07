<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# ab2

## Purpose
GITA Ablation 2 — driver notebook only (`ablation2.ipynb`, ~24 KB). No `modeling_gita.py` snapshot, so it runs against whatever version of GITA is currently importable from `ttadapters`.

## Key Files
| File | Description |
|------|-------------|
| `ablation2.ipynb` | Driver notebook for the second ablation. Imports the canonical `gita.modeling_gita.GITAEngine`. |

## For AI Agents

### Working In This Directory
- Because there is no local `modeling_gita.py`, the ablation result depends on the *current* canonical implementation. If you change `methods/deepsupervisions/input/gita/modeling_gita.py`, this notebook's numbers will shift — pin a known-good commit before re-running.

### Testing Requirements
- Not part of CI.

## Dependencies

### External
- Jupyter, `ttadapters` runtime stack.

<!-- MANUAL: -->

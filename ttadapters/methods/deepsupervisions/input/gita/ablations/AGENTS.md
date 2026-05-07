<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# ablations

## Purpose
GITA **ablation experiments**: each subfolder freezes a snapshot of `modeling_gita.py` (or a notebook driver) corresponding to one ablation configuration. The four directories cover the original two ablations (`ab1`, `ab2`) plus an "additional" pair (`aab1`, `aab2`) added after initial review. Treat these as research artifacts — not part of the production import path.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `ab1/` | Ablation 1 — variant of GITA + driver notebook (see [ab1/AGENTS.md](ab1/AGENTS.md)) |
| `ab2/` | Ablation 2 — driver notebook only (no model code copy) (see [ab2/AGENTS.md](ab2/AGENTS.md)) |
| `aab1/` | Additional ablation 1 — variant + driver notebook (see [aab1/AGENTS.md](aab1/AGENTS.md)) |
| `aab2/` | Additional ablation 2 — driver notebook only (see [aab2/AGENTS.md](aab2/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- These folders contain **frozen experimental copies** of GITA, not new variants meant to be importable. Do not import from `gita.ablations.*` in production code; instead, port findings into `gita/modeling_gita.py` or `gita/configuration_gita.py`.
- Notebooks may rely on `data/` symlinks at the repo root and on weights cached locally — they are not expected to run from a fresh clone.
- When adding a new ablation, follow the existing pattern: a numbered subdirectory containing exactly one `*.ipynb` plus an optional `modeling_gita.py` if the variant requires diverged code.

### Testing Requirements
- Not part of CI. Validate ablation results by running each notebook's cells end-to-end and recording the metric numbers.

### Common Patterns
- File naming: notebooks use `ablation<N>.ipynb` for the original two and `additional_ab<N>.ipynb` for the follow-ups.

## Dependencies

### Internal
- Each notebook imports `ttadapters` modules at session-time. Snapshot `modeling_gita.py` files override the canonical one when present in the same folder via `sys.path` manipulation in the notebook.

### External
- Jupyter, plus the standard `ttadapters` runtime stack.

<!-- MANUAL: -->

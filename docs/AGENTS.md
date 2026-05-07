<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# docs

## Purpose
Project-level documentation assets: figures used in the README and ad-hoc external reference notebooks pulled from upstream tutorials. There are no Markdown narratives here yet — the canonical docs are in the root `README.md`.

## Key Files
| File | Description |
|------|-------------|
| (none directly) | Top-level `docs/` only contains subdirectories. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `images/` | Figures referenced from the root README (see [images/AGENTS.md](images/AGENTS.md)) |
| `references/` | External tutorial notebooks kept for reproducibility (see [references/AGENTS.md](references/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- Don't add narrative `.md` files here unless coordinated — current convention is to keep documentation in the root README and per-method `README.md` (e.g. `ttadapters/models/yolo11/README.md`).
- When adding figures to `images/`, prefer SVG and reference relative paths from the root README (e.g. `./docs/images/foo.svg`).
- Reference notebooks in `references/` are external in origin — preserve their attribution and link in the local README inside each subfolder.

### Testing Requirements
- None (documentation only).

### Common Patterns
- Each `references/<topic>/` ships a tiny `README.md` crediting the upstream source plus the imported notebook.

## Dependencies

### External
- None.

<!-- MANUAL: -->

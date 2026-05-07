<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# low_rank

## Purpose
**Low-rank adapter** PEFT TTA — adapt at test time by training small low-rank matrices inserted alongside frozen backbone weights (LoRA-style). The single concrete method is WHW.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `whw/` | WHW (When, Where, and How to Adapt?) — adapter-based PEFT TTA with redundancy skip (see [whw/AGENTS.md](whw/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- Add new low-rank PEFT methods as sibling folders. Keep adapter dimension knobs (`adapter_ratio`, `adapter_init_option`, `adapter_dropout`) consistent with existing names so that downstream comparison studies don't need wrappers.

### Testing Requirements
- See `whw/AGENTS.md`.

## Dependencies

### Internal
- `....base`.

### External
- `torch.nn`.

<!-- MANUAL: -->

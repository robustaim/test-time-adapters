<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# ttt

## Purpose
TTT (Test-Time Training) family — auxiliary self-supervised tasks for TTA. **All files here are empty placeholders** awaiting implementation.

## Key Files
| File | Description |
|------|-------------|
| `ttt.py` | Empty — slated for the original TTT (Sun et al., 2020): rotation-prediction auxiliary head. |
| `tttpp.py` | Empty — slated for TTT++: rotation + contrastive auxiliary heads. |
| `masked_ttt.py` | Empty — slated for Masked TTT: masked-pixel prediction auxiliary head. |

## For AI Agents

### Working In This Directory
- The expectation per file is one method = one full triplet (`configuration_<name>.py`, `modeling_<name>.py`, `__init__.py`). The current empty file naming hints at the future split, but you may collapse smaller methods into one file initially.
- TTT requires a small auxiliary head on top of the backbone. Use `BaseModel`'s feature-extraction hooks (see `models/rcnn/hooks.py` for an example).

### Testing Requirements
- After implementation, register and verify `from ttadapters.methods import *` succeeds.

### Common Patterns
- TTT-family typically requires `fit()` to be implemented (offline pretraining of the auxiliary head before any TTA can happen). Plumb that through `AdaptationEngine.fit(source_dataset, batch_size, max_samples, dtype)`.

## Dependencies

### Internal
- Will depend on `methods.base.AdaptationEngine`.

### External
- TBD (likely `torch.nn`, `torchvision.transforms.v2` for rotation augmentation).

<!-- MANUAL: -->

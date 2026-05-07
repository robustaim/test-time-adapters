<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# bn

## Purpose
**BatchNorm-targeted entropy minimization** — methods that minimize prediction entropy on the test stream while updating *only* the BN affine parameters (scale + shift). All files here are currently empty stubs.

## Key Files
| File | Description |
|------|-------------|
| `tent.py` | Empty — slated for TENT (Wang et al., ICLR 2021). |
| `tentpp.py` | Empty — slated for TENT++ (extension). |

## For AI Agents

### Working In This Directory
- TENT's `online_parameters()` should yield only `weight`/`bias` of `nn.BatchNorm2d` modules, with `track_running_stats=False` and `train()` mode for those layers (so per-batch stats are used at inference). Document this clearly in the engine's docstring when implementing.
- TENT++ typically adds something on top (sample weighting, EMA over predictions, etc.) — keep it as a separate engine class that *does not* subclass TENT to avoid implicit-coupling regressions.

### Testing Requirements
- After implementation: confirm only BN affines move and that average prediction entropy decreases over the SHIFT continuous stream.

## Dependencies

### Internal
- Will depend on `methods.base.AdaptationEngine`.

### External
- `torch.nn.BatchNorm2d`, `torch.nn.functional.softmax`, `torch.nn.functional.log_softmax`.

<!-- MANUAL: -->

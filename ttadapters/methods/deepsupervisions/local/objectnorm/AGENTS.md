<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# objectnorm

## Purpose
**ObjectNorm** — TTA via per-object/per-RoI normalization alignment (each detected instance contributes its own statistic to the loss, restricting supervision to foreground regions). **Currently a stub**: `modeling_objectnorm.py` is empty. No `__init__.py`.

## Key Files
| File | Description |
|------|-------------|
| `modeling_objectnorm.py` | Empty placeholder — implementation TBD. |

## For AI Agents

### Working In This Directory
- When implementing, follow the standard triplet (`configuration_objectnorm.py`, `modeling_objectnorm.py`, `__init__.py` registers HF Auto classes).
- Object-localized stats need detection results — either run an initial forward pass, threshold predictions by confidence (mirror `MeanTeacherConfig.conf_threshold=0.3` as a sensible default), and crop features at those RoIs via `torchvision.ops.roi_align`.

### Testing Requirements
- Smoke-test once implemented; confirm the loss responds to mask supports (zero foreground = zero gradient).

## Dependencies

### Internal
- Will depend on `methods.base.AdaptationEngine` and the detector's RoI head.

### External
- TBD (likely `torchvision.ops.roi_align`).

<!-- MANUAL: -->

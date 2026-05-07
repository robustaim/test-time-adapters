<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# teacher

## Purpose
**Teacher-student** consistency regularization — TTA via maintaining a teacher copy of the model (typically EMA-updated) and training the student to agree with the teacher's predictions on augmented views of the same input. Two methods live here: `MeanTeacher` (temporal EMA only) and `TeST` (Teacher-Student Augmentation Consistency, with explicit teacher/student/online stages and entropy minimization in the second stage).

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `mean_teacher/` | MeanTeacher: pure EMA teacher with RandAugment + cutout (see [mean_teacher/AGENTS.md](mean_teacher/AGENTS.md)) |
| `test/` | TeST: two-stage teacher-student with online entropy minimization (see [test/AGENTS.md](test/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- Both methods consume `conf_threshold` (default `0.3`) for pseudo-label filtering — predictions below it are excluded from the consistency loss.
- The teacher is held in the engine but not exposed as a regular nn submodule; `reset()` overrides must explicitly snapshot/restore it.
- The folder name `test/` (the Test-time Self-Training method) is unrelated to pytest test files. Don't be confused by the homonym.

### Testing Requirements
- See `mean_teacher/AGENTS.md` and `test/AGENTS.md`.

## Dependencies

### Internal
- `....base` — `AdaptationConfig`, `AdaptationEngine`.

### External
- `torch.nn`, `torchvision.transforms.v2` (RandAugment + cutout).

<!-- MANUAL: -->

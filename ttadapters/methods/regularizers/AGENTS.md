<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# regularizers

## Purpose
**Consistency-regularization** TTA — adapts at test time by enforcing prediction agreement between a teacher and a student (often the same network at different EMA times, or with different augmentations). Two methods live here: **MeanTeacher** (temporal EMA teacher) and **TeST** (teacher-student augmentation consistency).

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | `from .teacher.mean_teacher import *` and `from .teacher.test import *`. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `teacher/` | Teacher-student style methods: MeanTeacher and TeST (see [teacher/AGENTS.md](teacher/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- Teacher methods maintain an EMA copy of the student. The `reset()` override must restore *both* the student (`base_state`) and the teacher (a separate stored copy) — verify your override does this when adding new teacher variants.
- Pseudo-label confidence threshold (`conf_threshold`) is the standard knob: predictions below threshold are excluded from the consistency loss. Default is `0.3` for both MeanTeacher and TeST.
- `MeanTeacher` uses `ema_alpha=0.999` and adds RandAugment-style data augmentation parameterized by `augment_strength_n=2`, `augment_strength_m=10`, `cutout_size=16`. TeST omits `cutout_size` and adds two-stage gradient-step controls (`n_teacher_steps`, `n_student_steps`).
- TeST runs in three modes: `"teacher"`, `"student"`, `"online"`. Only `"online"` is exercised in the current experiments — `"teacher"` and `"student"` use the offline `fit()` path with `n_teacher_epochs`.

### Testing Requirements
- Smoke-test by running on a SHIFT continuous subset and verifying that the teacher (EMA) typically outperforms the student on later steps (a sanity check that the EMA isn't broken).

### Common Patterns
- Both methods support all four detector classes (`FasterRCNN`, `SwinRCNN`, `RTDetr`, `YOLO11`) in `from_preset`. SwinRCNN gets a 20× lower learning rate (`5e-6` instead of `1e-4`) because of higher sensitivity.

## Dependencies

### Internal
- `..base` — `AdaptationConfig`, `AdaptationEngine`.
- `....models` — all four detector classes (lazily imported in `from_preset`).

### External
- `torch.nn`, `transformers.AutoConfig`, augmentation libs (`torchvision.transforms.v2`, RandAugment).

<!-- MANUAL: -->

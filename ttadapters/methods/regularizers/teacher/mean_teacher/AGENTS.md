<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# mean_teacher

## Purpose
**MeanTeacher** — TTA via temporal-EMA teacher: the teacher tracks the student's parameters with `θ_teacher ← α·θ_teacher + (1−α)·θ_student` (`α = ema_alpha = 0.999`), and the student minimizes a consistency loss between its predictions on a *strongly augmented* view and the teacher's predictions on the *clean* view. Pseudo-labels below `conf_threshold=0.3` are dropped.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Registers `MeanTeacherConfig` / `MeanTeacherEngine` with HF Auto classes. |
| `configuration_mean_teacher.py` | `MeanTeacherConfig`. Hyperparameters: `base_type ∈ {rcnn, swinrcnn, rtdetr, yolo11}`, `optim ∈ {SGD, AdamW}` default `SGD`, `adapt_lr=1e-4`, `momentum=0.9`, `weight_decay=1e-4`, `conf_threshold=0.3`, `ema_alpha=0.999`, `weight_reg=0.0`, `augment_strength_n=2`, `augment_strength_m=10`, `cutout_size=16`. |
| `modeling_mean_teacher.py` | `MeanTeacherEngine` (~16 KB). Holds the teacher network, builds RandAugment+cutout pipeline, computes the consistency loss. |

## For AI Agents

### Working In This Directory
- Supports all four detector classes; SwinRCNN gets a 20× lower learning rate (`5e-6` instead of `1e-4`) per `from_preset`.
- `ema_alpha=0.999` is high — the teacher updates slowly. If you reduce it, also reduce `adapt_lr` proportionally to avoid runaway drift.
- The augmentation strength is RandAugment-style: `n=2` operations sampled per image, magnitude `m=10` (out of 30), plus `cutout_size=16` pixels. Don't change unless you re-tune the consistency-loss weight.
- `weight_reg=0.0` disables explicit weight regularization; treat it as an ablation toggle, not a tuned hyperparameter.

### Testing Requirements
- Verify the EMA actually updates: snapshot `engine.teacher.state_dict()` and check it diverges from `engine.base_model.state_dict()` after a few steps.
- Smoke-test on each of the four backbones.

### Common Patterns
- README marks Mean-Teacher as `[x]`. Reference: "Mean Teachers Are Better Role Models" (Tarvainen & Valpola, NeurIPS 2017) adapted for TTA.

## Dependencies

### Internal
- `.....base`, `......models` (lazy import in `from_preset`).

### External
- `transformers.AutoConfig`, `torch.nn`, `torchvision.transforms.v2` (RandAugment, RandomErasing/Cutout).

<!-- MANUAL: -->

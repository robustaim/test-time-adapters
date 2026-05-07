<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# test

## Purpose
**TeST** (Teacher-Student Augmentation Consistency, "Test-time Self-Training") — two-stage TTA. Stage 1 trains the teacher with a consistency loss against augmented views (`lambda_cons=1.0`); Stage 2 trains the student with entropy minimization (`lambda_ent=0.25` default, `0.5` per-backbone preset). Supports three modes: `"teacher"`, `"student"`, `"online"`. Only the `"online"` path is exercised in current experiments.

> ⚠ **Naming caveat**: this folder is `test/` because TeST is the method's name (Test-time Self-Training). It is NOT a pytest folder. Don't move it without updating the `regularizers/__init__.py` wildcard import.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Registers `TeSTConfig` / `TeSTEngine` with HF Auto classes. |
| `configuration_test.py` | `TeSTConfig`. Hyperparameters: `base_type ∈ {rcnn, swinrcnn, rtdetr, yolo11}`, `optim ∈ {SGD, Adam, AdamW}` default `SGD`, `adapt_lr=1e-4`, `weight_decay=1e-4`, `conf_threshold=0.3`, `weight_reg=0.0`, `augment_strength_n=2`, `lambda_cons=1.0`, `lambda_ent=0.25`, `stage ∈ {teacher, student, online}` default `online`, `n_teacher_epochs=10` (offline `fit()`), `n_teacher_steps=10`, `n_student_steps=1`. |
| `modeling_test.py` | `TeSTEngine` (~29 KB) — implements all three stages plus EMA-style teacher updates and the entropy-minimization step. |

## For AI Agents

### Working In This Directory
- The only stage actually used in current experiments is `"online"`. The `from_preset` overrides set `lambda_ent=0.5`, `n_teacher_epochs=1`, `n_student_steps=1` for all four backbones — i.e. the offline epochs are effectively disabled.
- `n_teacher_steps=10` (per batch in online mode) is the count of teacher gradient updates per incoming test batch. Reducing it speeds up online inference at the cost of teacher quality.
- `weight_reg=0.0` disables weight regularization; this is the same convention as `MeanTeacher`.
- `augment_strength_n=2` matches MeanTeacher's `augment_strength_n` (no separate `m` here — TeST's RandAugment is configured with default magnitude).

### Testing Requirements
- Smoke-test only the `"online"` stage. Verify entropy-minimization is active (Stage-2 entropy should decrease within ~50 batches).
- Confirm pseudo-labels are filtered: log the fraction of detections retained per batch (should be `< 1.0` when `conf_threshold=0.3`).

### Common Patterns
- README marks TeST as `[ ]` (Teacher-Student Augmentation Consistency) — implementation exists but the README checklist hasn't been ticked yet.
- Reference: TeST paper from the self-training/semi-supervised TTA line.

## Dependencies

### Internal
- `.....base`, `......models` (lazy import in `from_preset`).

### External
- `transformers.AutoConfig`, `torch.nn`, `torchvision.transforms.v2` (RandAugment).

<!-- MANUAL: -->

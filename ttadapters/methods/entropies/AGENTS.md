<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# entropies

## Purpose
**Entropy minimization** TTA — adapts at test time by minimizing the prediction entropy of the model on unlabeled target samples. The classic family in TTA (TENT, EATA, MEMO, DeYO). Currently a **stub package**: every method file is empty. The README's checklist marks the entire entropies bullet as `[ ]`.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Empty — no public exports yet. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `bn/` | BatchNorm-targeted entropy methods: TENT, TENT++ (empty stubs) (see [bn/AGENTS.md](bn/AGENTS.md)) |
| `sample/` | Sample-efficient entropy methods: EATA, MEMO, DeYO (empty stubs) (see [sample/AGENTS.md](sample/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- Files exist as empty placeholders so that future PRs land in the right location. When implementing, follow the standard configuration/modeling triplet and register the Auto classes in the leaf `__init__.py`.
- TENT-style methods adapt only BN affine parameters — `online_parameters()` should return only `weight`/`bias` of `nn.BatchNorm2d` modules.
- Sample-selection variants (EATA/DeYO) need a per-sample reliability score; consider plumbing it through `_stats` on the engine so it shows up in `Evaluator` output.

### Testing Requirements
- Once implemented, smoke-test against a SHIFT continuous subset; entropy should monotonically decrease over the first ~100 batches.

### Common Patterns
- The classification entropy loss: `-(p * log p).sum(dim=-1).mean()` where `p = softmax(logits)`. For object detection, apply per-instance after NMS or per-anchor before NMS — pick a convention and document it on the engine.

## Dependencies

### Internal
- Will depend on `methods.base.AdaptationEngine`.

### External
- `torch.nn.functional.softmax`, `torch.nn.functional.log_softmax`.

<!-- MANUAL: -->

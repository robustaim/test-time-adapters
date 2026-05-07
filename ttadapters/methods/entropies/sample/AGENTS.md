<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# sample

## Purpose
**Sample-efficient entropy minimization** — variants of entropy-minimization TTA that decide *which* test samples participate in the gradient update (EATA), augment-and-vote at the marginal level (MEMO), or de-bias the entropy objective (DeYO). All files here are currently empty stubs.

## Key Files
| File | Description |
|------|-------------|
| `eata.py` | Empty — slated for EATA (Niu et al., ICML 2022). |
| `memo.py` | Empty — slated for MEMO (Zhang et al., NeurIPS 2022). |
| `deyo.py` | Empty — slated for DeYO. |

## For AI Agents

### Working In This Directory
- EATA computes per-sample reliability scores and gradient-weighting — implement the score in a separate helper (consider placing it in `methods/samplers/` and importing from there).
- MEMO requires multiple augmented views per sample at test time. Reuse the augmentation strength knobs from `regularizers/teacher/mean_teacher` (`augment_strength_n`, `augment_strength_m`) for consistency.
- DeYO needs to be cited / scoped before implementation; the README's bullet `[ ] De-biasing (DeYO)` lives under "Sample Selection" rather than "Entropy" — be deliberate about the placement.

### Testing Requirements
- Once implemented, confirm that the per-sample gating actually filters batches (e.g. log the fraction of samples retained per step on a SHIFT continuous run).

## Dependencies

### Internal
- Will depend on `methods.base.AdaptationEngine` and possibly `methods.samplers`.

### External
- `torch.nn.functional`, `torchvision.transforms.v2` (for MEMO augmentations).

<!-- MANUAL: -->

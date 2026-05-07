<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# actmad

## Purpose
The **ActMAD active-sampling** half — scores test samples by their activation-mean discrepancy from source statistics so that the ActMAD adaptation engine (`methods/deepsupervisions/full/actmad/`) only updates on batches that meaningfully shift the BN activations. The folder currently contains only `__pycache__` (no source files in version control yet); the active-sampling logic is presently inlined inside `modeling_actmad.py` of the deepsupervisions counterpart.

## Key Files
| File | Description |
|------|-------------|
| (none in source) | Source files have not been split out into this folder yet — the sampler logic lives inside `methods/deepsupervisions/full/actmad/modeling_actmad.py`. The `__pycache__/` here is a stale artifact of an earlier layout. |

## For AI Agents

### Working In This Directory
- This folder is an **intentional placeholder** for when the sampler logic gets factored out of the engine. When extracting it: expose a small callable / module here, import it into `modeling_actmad.py`, and add an `__init__.py`.
- Do NOT add new methods that import `from ttadapters.methods.samplers.active.actmad` until the source files actually exist — the import will fail.
- The stale `__pycache__/` can be safely removed; the build does not depend on it.

### Testing Requirements
- Once the sampler is split out, validate that `mAP_with_active_sampling >= mAP_without_active_sampling - epsilon` on a SHIFT continuous Faster R-CNN run (the active sampler should not hurt; ideally it improves stability).

### Common Patterns
- The naming mirrors the engine: there is exactly one ActMAD method, with two halves (`deepsupervisions/full/actmad/` for the engine, `samplers/active/actmad/` for the sampler).

## Dependencies

### Internal
- (Future) `methods/deepsupervisions/full/actmad/modeling_actmad.py` will import from here.

### External
- `torch`.

<!-- MANUAL: -->

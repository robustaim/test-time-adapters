<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# datasets

## Purpose
Dataset wrappers and **TTA scenarios**. Wraps standard CV datasets (COCO, SHIFT, CityScapes, ACDC, ImageNet-1K, GOT-10k) into a uniform `BaseDataset` interface plus task-typed variants (`*ForObjectDetection`, `*ForPanopticSegmentation`, `*ForSemanticSegmentation`, `*ForObjectTracking`). Adds clean / corrupted / discrete / continuous splits required to study domain shift, and a `scenarios/` sub-package for orchestrating Standard / Gradual / Continual / Universal TTA workflows.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Public API: re-exports all dataset classes plus `BaseDataset`, `DatasetHolder`, `DataLoaderHolder`, `DataPreparation`, and the `scenarios` sub-package. |
| `base.py` | `BaseDataset` (subclass of `torch.utils.data.Dataset`), `DatasetHolder` / `DataLoaderHolder` (train/valid/test holders that print sample/iter counts on init), and `DataPreparation` (per-model transforms / collate / pre/post-process hooks). |
| `coco.py` | `COCODataset`, `COCODatasetForObjectDetection`, `COCOCorruptedDatasetForObjectDetection` (~17 KB). |
| `shift.py` | SHIFT dataset (driving simulation). Discrete and continuous variants at 1×/10×/100× scaling. Includes `patch_fast_download_for_object_detection` (~26 KB). |
| `cityscapes.py` | CityScapes wrappers — clean / corrupted / discrete / continuous (~32 KB). |
| `acdc.py` | ACDC adverse-condition driving — detection / panoptic / semantic variants. |
| `got10k.py` | GOT-10k object-tracking + `PairedGOT10kDataset`. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `imagenet1k/` | ImageNet-1K wrapper backed by HuggingFace `ILSVRC/imagenet-1k` (see [imagenet1k/AGENTS.md](imagenet1k/AGENTS.md)) |
| `scenarios/` | TTA scenarios: Standard / Gradual / Continual / Universal (see [scenarios/AGENTS.md](scenarios/AGENTS.md)) |

## For AI Agents

### Working In This Directory
- Every dataset class must set `dataset_name` and `classes` on the class itself — `BaseModel.__init__` reads `dataset.classes` to compute `num_classes`, and methods key per-dataset behavior on `dataset_name`.
- `DataPreparation` is meant to be subclassed *per model*, not per dataset. Default behavior is identity for `transforms` / `collate_fn` / `pre_process` / `post_process`.
- `DatasetHolder.__post_init__` and `DataLoaderHolder.__post_init__` print summary lines on construction. Don't call them in tight loops.
- New datasets must be added to both this directory AND `__init__.py`'s explicit re-export list — wildcard `*` is intentionally avoided here.

### Testing Requirements
- Smoke-test new dataset wrappers via the `scenarios/` mechanism: instantiate the corresponding scenario, iterate one batch via `DataLoader`, check `len(...)` matches the expected split sizes documented in the root README.

### Common Patterns
- Object-detection variants use the canonical key dict `dict(bboxes="boxes2d", classes="boxes2d_classes", original_size="original_hw")` — match this when adding a new detection dataset so existing `DataPreparation` subclasses keep working.
- Corrupted variants leverage `imagecorruptions` library for synthetic perturbations.
- `force_download` defaults to `True` for HuggingFace-hosted datasets but `False` for local-only ones.

## Dependencies

### Internal
- Used by `ttadapters.models` (`BaseModel.__init__` accepts a `BaseDataset`) and `ttadapters.methods` (via `DataPreparation`).

### External
- `torch.utils.data`, `torchvision.datasets`, `huggingface_hub`, `pycocotools`, `cityscapesscripts`, `shift-dev` (forked), `imagecorruptions`, `tqdm`.

<!-- MANUAL: -->

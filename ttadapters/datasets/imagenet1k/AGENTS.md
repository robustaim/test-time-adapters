<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# imagenet1k

## Purpose
ImageNet-1K (ILSVRC2012) classification dataset wrapper. Pulls the `ILSVRC/imagenet-1k` dataset from HuggingFace Hub at a pinned revision, extracts the per-split tar archives, and reorganizes images into the `ImageFolder`-compatible `<class_id>/<image>.JPEG` layout.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | `ImageNet1K(BaseDataset, datasets.ImageFolder)`. Inherits from `torchvision.datasets.ImageFolder`; downloads via `huggingface_hub.hf_hub_download` at revision `1500f8c59b214ce459c0a593fa1c87993aeb7700`. Hard-coded constants: `img_size=224`, `img_mean=(0.485, 0.456, 0.406)`, `img_std=(0.229, 0.224, 0.225)`. |
| `classes.py` | `IMAGENET2012_CLASSES` ordered dict mapping `n0XXXXXXXX` synset id → human-readable class name (1000 entries, ~46 KB). |

## For AI Agents

### Working In This Directory
- `_download()` is destructive: it extracts archives in-place, *removes* the staging tar files (`remove_finished=True` in `extract_archive`), and renames extracted JPEGs into per-class subfolders. Don't invoke against a directory you care about preserving raw.
- `train` archives are 5 shards (`train_images_0..4.tar.gz`); `val` is one (`val_images.tar.gz`); `test` is one (`test_images.tar.gz`). The download routine picks the right list based on whether `root` ends in `train`/`val`/`test`.
- For `val`, the class id is parsed from the filename suffix (`_n0XXXXXXXX`) — assume HuggingFace val filenames keep that convention.
- `force_download=True` is the default. Pass `force_download=False` to skip download when the directory is already populated.

### Testing Requirements
- The first `__init__` against an empty `root` will download tens of GB. Use a cached `root` with `force_download=False` for quick smoke tests.

### Common Patterns
- Wraps a HF Hub dataset behind a torchvision `ImageFolder` so that the same `transform` / `target_transform` API works for both source-pretrained and TTA pipelines.

## Dependencies

### Internal
- `..base` — extends `BaseDataset`.

### External
- `torchvision.datasets.ImageFolder`, `torchvision.datasets.utils.extract_archive`, `huggingface_hub`.

<!-- MANUAL: -->

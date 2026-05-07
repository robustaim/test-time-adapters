<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# resnet

## Purpose
HuggingFace-backed ResNet wrapped as `BaseModel` for **image-classification** TTA experiments (e.g. ImageNet-1K corruption studies). This is the only image-classification model in the repo today; the other three model packages target object detection.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Public API: `ResNetForImageClassification`. |
| `modeling_resnet.py` | Thin wrapper (~2.5 KB) around `transformers.ResNetForImageClassification` setting `model_provider = ModelProvider.HuggingFace`. |

## For AI Agents

### Working In This Directory
- Not wildcard-exported from `ttadapters.models.__init__` — import explicitly: `from ttadapters.models.resnet import ResNetForImageClassification`.
- Pretraining/eval flow uses `ImageNet1K` from `ttadapters.datasets.imagenet1k`. `BaseModel.from_dataset(dataset)` reads `cls.ModelRegistry.<dataset_name>` for weights — populate that when adding new classification datasets.
- Tasks here are classification, so detection-specific configs (`adaptation_layers="backbone+encoder"`, `mask_value=114`, etc.) don't apply. TTA methods that target this model need their own `from_preset` branch.

### Testing Requirements
- Smoke-test: load a pretrained HF ResNet checkpoint and run a forward pass on a `(B, 3, 224, 224)` tensor.

### Common Patterns
- HuggingFace classification heads expose `logits` — match that contract when writing methods that consume the output.

## Dependencies

### Internal
- `..base` — `BaseModel`, `ModelProvider`.

### External
- `transformers.ResNetForImageClassification`, `torch`.

<!-- MANUAL: -->

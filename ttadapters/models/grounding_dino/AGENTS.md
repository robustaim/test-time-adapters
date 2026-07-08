<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-08 | Updated: 2026-07-08 -->

# grounding_dino

## Purpose
HuggingFace-backed Grounding DINO detector wrapped as `BaseModel`. Grounding DINO (IDEA-Research) is an open-vocabulary / phrase-grounded object detector: detection targets are supplied as text prompts rather than a fixed classifier head. This module exposes both an open-vocabulary variant (prompts supplied at runtime) and a closed-set variant bound to a dataset's fixed class list, plus the data-preparation glue that maps free-text matches back to integer class IDs for mAP evaluation. Because it is open-vocabulary, the same pretrained weights work across every dataset in the harness.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Public API: `GroundingDinoForObjectDetection`, `GroundingDinoForZeroShotObjectDetection`, `GroundingDinoDataPreparation`, `GroundingDinoFixedClassDataPreparation`. Wildcard-exported from `models/__init__.py`. |
| `modeling_grounding_dino.py` | Model + data-prep classes. Detectors inherit `BaseModel` + `GroundingDinoForObjectDetection` (HF) with `model_provider = ModelProvider.HuggingFace`; default checkpoint `IDEA-Research/grounding-dino-tiny`. |

## For AI Agents

### Working In This Directory
- **Two detector variants.** `GroundingDinoForZeroShotObjectDetection` is open-vocabulary — text prompts are supplied at runtime via each sample's `target['text_labels']`. `GroundingDinoForObjectDetection` subclasses it and swaps in `GroundingDinoFixedClassDataPreparation`, which auto-injects `dataset.classes` as a fixed prompt so callers never manage prompts manually.
- **`__init__` skips `BaseModel.__init__`.** The detector calls `super(BaseModel, self).__init__(config=...)` to run the HF `GroundingDinoForObjectDetection` constructor and bypass `BaseModel.__init__`. Preserve this MRO trick when editing — the class multiply-inherits `BaseModel`, the HF model, and `ObjectDetectionMixin`.
- **`forward` must list kwargs explicitly.** The `Trainer` introspects the signature, so `pixel_values`, `input_ids`, `attention_mask`, `labels`, etc. are all spelled out rather than passed via `**kwargs`. Keep them explicit when extending.
- **Two data-prep modes.** `GroundingDinoDataPreparation` supports closed-set (fixed `text_labels` at construction) and zero-shot (`text_labels=None`, per-sample phrases pulled from `target[dataset_key['text_labels']]`). `dataset_key` maps dataset-native field names (`boxes2d`, `boxes2d_classes`, `original_hw`, `text_labels`) — override it for datasets with different keys.
- **Label remapping is required for eval.** `post_process` calls `post_process_grounded_object_detection`, then remaps matched text spans back to integer class IDs via `_match_label_index` (exact → substring → token-overlap fallback). Unmatched phrases map to `-1`. Downstream mAP/`supervision.Detections` consumers expect integer `labels`, so do not remove this remapping.
- **Boxes are converted XYXY→XYWH** in `transforms` before building COCO-style `annotations` (with `area`/`iscrowd`), because the HF processor expects COCO format.

### Testing Requirements
- Smoke-test both variants: load `grounding-dino-tiny` from the HF Hub, run a forward pass, and confirm `post_process` returns integer `labels` aligned to the candidate phrase order (not `-1` for known classes).
- When adding a dataset, verify `_match_label_index` resolves that dataset's class strings against the prompts (watch for casing / phrasing mismatches).

### Common Patterns
- `ModelRegistry` aliases every dataset (`COCO`, `SHIFT`, `SHIFT_SUBSET`, `CityScapes`, `ACDC`) to the canonical `TINY_OFFICIAL` checkpoint — open-vocabulary weights are dataset-agnostic. Add new dataset aliases the same way rather than registering new weights.
- `text_template` (default `"{}"`) lets you wrap each class name in a prompt phrasing (e.g. `"a photo of a {}"`).

## Dependencies

### Internal
- `..base` — `BaseModel`, `ModelProvider`, `WeightsInfo`, `ObjectDetectionMixin`.
- `...datasets` — `BaseDataset`, `DataPreparation`.

### External
- `transformers` (`GroundingDinoConfig`, `GroundingDinoProcessor`, `GroundingDinoForObjectDetection`), `torch`, `torchvision.transforms.v2` / `tv_tensors`.

<!-- MANUAL: -->

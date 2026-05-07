<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-05-07 | Updated: 2026-05-07 -->

# rt_detr

## Purpose
HuggingFace-backed real-time DETR detector wrapped as `BaseModel`. RT-DETR is a transformer-based detector (Zhao et al., 2023) that is competitive with YOLO at real-time latencies. Used as one of the four canonical backbones for TTA experiments (the others being Faster R-CNN, Swin-RCNN, YOLO11).

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Public API: `RTDetrForObjectDetection`. |
| `modeling_rt_detr.py` | The detector class (~17 KB). Inherits `BaseModel` with `model_provider = ModelProvider.HuggingFace`. |
| `README.md` | Per-model notes (~31 KB) — typically a vendored copy of upstream HF docs / config tables. |

## For AI Agents

### Working In This Directory
- Loading goes through the HuggingFace branch of `BaseModel.load_from` — a reference model is constructed via `from_pretrained`, its `state_dict` is copied, and the temp is freed. Do not skip the `tie_weights()` + `post_init()` re-application after loading.
- RT-DETR's backbone is ResNet-style (`stages.[0..3].layers...`), which is why TTA configs use the `RESNET` regex preset for both Faster R-CNN and RT-DETR (e.g. `GITAConfig.from_preset(RTDetrForObjectDetection)` returns `cascade_target=TargetKeyPreset.RESNET.value`).
- `model_type` for HF Auto registration is the canonical RT-DETR string — don't shadow it.

### Testing Requirements
- Smoke-test: load a pretrained RT-DETR from the HF Hub and run a forward pass on a 640×640 image batch.

### Common Patterns
- All configs that target RT-DETR set `adaptation_layers="backbone+encoder"` because RT-DETR has a meaningful encoder (transformer) on top of the conv backbone.

## Dependencies

### Internal
- `..base` — `BaseModel`, `ModelProvider`.

### External
- `transformers` (RTDetrModel, RTDetrConfig), `torch`, `torchvision`.

<!-- MANUAL: -->

# Test-Time Adaptation Engines

Framework-agnostic test-time adaptation engines that work with Detectron2, Transformers (RT-DETR), and Ultralytics (YOLO).

## Architecture Overview

```
ttadapters/methods/
├── framework_adapters.py       # Framework abstraction layer
├── base.py                      # AdaptationEngine base class
├── samplers/active/
│   └── actmad.py               # ActMADEngine
├── batchnorms/
│   ├── covariate/dua.py        # DUAEngine
│   └── dynamic/norm.py         # NORMEngine
├── regularizers/teacher/
│   └── mean_teacher.py         # MeanTeacherEngine
└── pefts/low_rank/
    └── whw.py                  # WHWEngine (Detectron2 only)
```

## Key Features

✅ **Framework-Agnostic**: Automatically detects and adapts to different frameworks
✅ **Unified Interface**: Same API across all engines
✅ **Modular Design**: Easy to extend with new methods
✅ **Production-Ready**: Tested with real-world models

## Quick Start

### 1. ActMADEngine (Activation Mean Alignment)

Works with: Detectron2, Transformers, Ultralytics

```python
from ttadapters.methods import ActMADEngine, ActMADConfig

# Configure
config = ActMADConfig(
    adaptation_layers="backbone+encoder",  # or "backbone", "encoder"
    lr=1e-4,
    clean_dataset=clean_dataset,  # for statistics extraction
    device="cuda"
)

# Create engine (auto-detects framework)
engine = ActMADEngine(model, config)

# Use in inference
for batch in dataloader:
    outputs = engine(batch)
```

### 2. DUAEngine (Dynamic Update Adaptation)

Works with: Detectron2, Transformers, Ultralytics

```python
from ttadapters.methods import DUAEngine, DUAConfig

# Configure
config = DUAConfig(
    adaptation_layers="backbone+encoder",
    min_momentum_constant=0.0001,
    decay_factor=0.94,
    mom_pre=0.01,
    device="cuda"
)

# Create engine
engine = DUAEngine(model, config)

# Use in inference
for batch in dataloader:
    outputs = engine(batch)

# Reset for new domain
engine.reset_momentum()
```

### 3. NORMEngine (Normalization with Source-Target Blending)

Works with: Detectron2, Transformers, Ultralytics

```python
from ttadapters.methods import NORMEngine, NORMConfig

# Configure
config = NORMConfig(
    adaptation_layers="backbone+encoder",
    source_sum=128,  # number of source samples for weighting
    device="cuda"
)

# Create engine
engine = NORMEngine(model, config)

# Use in inference
for batch in dataloader:
    outputs = engine(batch)
```

### 4. MeanTeacherEngine (Self-Training with EMA)

Works with: Detectron2, Transformers, Ultralytics

```python
from ttadapters.methods import MeanTeacherEngine, MeanTeacherConfig

# Configure
config = MeanTeacherConfig(
    adaptation_layers="backbone+encoder",
    lr=1e-4,
    ema_alpha=0.99,  # EMA decay for teacher
    conf_threshold=0.3,  # confidence threshold for pseudo-labels
    device="cuda"
)

# Create engine
engine = MeanTeacherEngine(model, config)

# Use in inference
for batch in dataloader:
    outputs = engine(batch)
```

### 5. WHWEngine (What, How, Where to adapt)

Works with: **Detectron2 only** (full implementation)

```python
from ttadapters.methods import WHWEngine, WHWConfig

# Configure
config = WHWConfig(
    adaptation_where="adapter",  # "adapter", "normalization", "full"
    adapter_bottleneck_ratio=32,
    lr=1e-4,
    fg_align="KL",
    gl_align="KL",
    source_statistics_path="./whw_source_stats.pt",
    device="cuda"
)

# Create engine
engine = WHWEngine(model, config)

# Use in inference
for batch in dataloader:
    outputs = engine(batch)
```

## Framework-Specific Examples

### Detectron2 (Faster R-CNN)

```python
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from ttadapters.methods import ActMADEngine, ActMADConfig

# Build Detectron2 model
cfg = get_cfg()
cfg.merge_from_file("configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
model = build_model(cfg)

# Apply ActMAD
config = ActMADConfig(
    adaptation_layers="backbone",  # adapt backbone only
    lr=1e-4,
    clean_dataset=my_clean_dataset,
    device="cuda"
)
engine = ActMADEngine(model, config)

# Inference
for batch in detectron2_dataloader:
    outputs = engine(batch)  # list of dict with 'instances'
```

### Transformers (RT-DETR)

```python
from transformers import RTDetrForObjectDetection
from ttadapters.methods import DUAEngine, DUAConfig

# Load RT-DETR model
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")

# Apply DUA
config = DUAConfig(
    adaptation_layers="backbone+encoder",  # adapt backbone and encoder
    min_momentum_constant=0.0001,
    decay_factor=0.94,
    device="cuda"
)
engine = DUAEngine(model, config)

# Inference
for batch in transformers_dataloader:
    # batch = {'pixel_values': tensor, 'labels': list of dicts}
    outputs = engine(batch)  # RTDetrObjectDetectionOutput
```

### Ultralytics (YOLO v11)

```python
from ultralytics import YOLO
from ttadapters.methods import NORMEngine, NORMConfig

# Load YOLO model
model = YOLO("yolo11n.pt")

# Apply NORM
config = NORMConfig(
    adaptation_layers="backbone",  # adapt backbone
    source_sum=128,
    device="cuda"
)
engine = NORMEngine(model.model, config)  # Note: use model.model

# Inference
for batch in yolo_dataloader:
    outputs = engine(batch)
```

## Advanced: Custom Framework Adapter

If you need to support a custom framework:

```python
from ttadapters.methods import FrameworkAdapter

class MyCustomAdapter(FrameworkAdapter):
    framework_name = "my_framework"

    def identify_normalization_layers(self, model, layer_filter=None):
        # Implement layer discovery
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if layer_filter is None or layer_filter(name, module):
                    layers.append((name, module))
        return layers

    def execute_forward(self, model, batch, device):
        # Implement forward pass
        return model(batch.to(device))

    def prepare_batch(self, batch, device):
        # Implement batch preparation
        return batch.to(device)

    def register_feature_hook(self, layer, hook_handler):
        # Standard PyTorch hook
        return layer.register_forward_hook(hook_handler)

# Use custom adapter
adapter = MyCustomAdapter()
engine = ActMADEngine(model, config, adapter=adapter)
```

## Migration from other_method/baseline.py

Old code (RT-DETR specific):
```python
from ttadapters.methods.other_method.rtdetr_baseline import ActMAD

actmad = ActMAD.load(
    model=model,
    data_root="./datasets",
    adaptation_layers="backbone+encoder"
)
results = actmad.evaluate_all_tasks()
```

New code (framework-agnostic):
```python
from ttadapters.methods import ActMADEngine, ActMADConfig

config = ActMADConfig(
    adaptation_layers="backbone+encoder",
    clean_dataset=clean_dataset,
    device="cuda"
)
engine = ActMADEngine(model, config)

for batch in dataloader:
    outputs = engine(batch)
```

## Supported Frameworks

| Engine | Detectron2 | Transformers | Ultralytics |
|--------|-----------|--------------|-------------|
| ActMADEngine | ✅ | ✅ | ✅ |
| DUAEngine | ✅ | ✅ | ✅ |
| NORMEngine | ✅ | ✅ | ✅ |
| MeanTeacherEngine | ✅ | ✅ | ✅ |
| WHWEngine | ✅ | ⚠️ (partial) | ⚠️ (partial) |

✅ = Full support
⚠️ = Basic support (no parallel adapters)

## Tips

1. **Choose the right adaptation layers**:
   - `"backbone"`: Fast, works well for domain shifts
   - `"encoder"`: For architectural differences
   - `"backbone+encoder"`: Most comprehensive (default)

2. **Start with simple methods**:
   - Try DUAEngine or NORMEngine first (no hyperparameters)
   - Move to ActMADEngine if you need more control
   - Use MeanTeacherEngine for complex scenarios

3. **Framework detection**:
   - Usually automatic, but you can specify:
     ```python
     from ttadapters.methods import create_adapter
     adapter = create_adapter(model, adapter_type="detectron2")
     engine = ActMADEngine(model, config, adapter=adapter)
     ```

## Troubleshooting

**Problem**: Engine doesn't adapt (loss is 0)
- **Solution**: Check that `adaptation_layers` matches your model architecture
- Use `create_layer_filter_*` methods to verify layer selection

**Problem**: CUDA out of memory
- **Solution**: Reduce batch size or use gradient accumulation
- Some engines (ActMAD) require backward pass through entire model

**Problem**: Framework not detected
- **Solution**: Specify adapter explicitly:
  ```python
  from ttadapters.methods import Detectron2Adapter
  adapter = Detectron2Adapter()
  engine = ActMADEngine(model, config, adapter=adapter)
  ```

## References

- ActMAD: [Paper Link]
- DUA: [Paper Link]
- NORM: [Paper Link]
- Mean Teacher: [Paper Link]
- WHW: [ContinualTTA Repository](https://github.com/robustaim/ContinualTTA_ObjectDetection)

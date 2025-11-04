# APT: Adaptive Plugin for Test-time Adaptation via Temporal Consistency

## Overview

**APT (Adaptive Plugin using Temporal Consistency)** is a test-time adaptation method designed for object detection models operating under continual domain shift scenarios, such as autonomous driving.

### Key Idea

In autonomous driving scenarios, the environment continuously changes (weather, lighting, etc.). APT leverages **temporal consistency** in video sequences to adapt the model without requiring labeled data. By tracking objects across frames using Kalman filters (inspired by SORT), APT creates pseudo-labels from temporally consistent predictions and uses them to adapt the model online.

### How It Works

1. **Track Objects**: Use Kalman filter to track detected objects across frames
2. **Predict Next Frame**: Predict where objects should appear in the next frame based on motion
3. **Compute Loss**: Compare current detections with tracked predictions
4. **Adapt Model**: Update model parameters using temporal consistency loss

## Architecture

```
Frame t-1: Detections → Kalman Filter → Predicted boxes for Frame t
                                              ↓
Frame t:   Current Detections ←--[Loss]--→ Pseudo-labels (Tracked predictions)
                                              ↓
                                        Backpropagation
                                              ↓
                                        Model Adaptation
```

## Files

- `apt_config.py`: Configuration dataclass for APT parameters
- `kalman_tracker.py`: Kalman filter-based tracker (inspired by SORT algorithm)
- `apt_plugin.py`: Main APT adaptation engine
- `apt_example.py`: Complete usage example
- `APT_USAGE_EXAMPLE.md`: Detailed usage documentation
- `__init__.py`: Module initialization

## Installation

### Prerequisites

```bash
# Core dependencies (should already be installed with ttadapters)
pip install torch torchvision
pip install detectron2  # for Faster R-CNN models

# APT-specific dependencies
pip install filterpy scipy numpy
```

### Integration with ttadapters

Option 1: Place files in `ttadapters/methods/`:

```bash
cp apt_config.py ttadapters/methods/
cp kalman_tracker.py ttadapters/methods/
cp apt_plugin.py ttadapters/methods/
```

Then update `ttadapters/methods/__init__.py`:

```python
from .apt_config import APTConfig
from .apt_plugin import APTPlugin
```

Option 2: Use as standalone module (keep files in current directory)

## Quick Start

### Basic Usage

```python
import torch
from ttadapters import datasets, models
from apt_config import APTConfig
from apt_plugin import APTPlugin

# 1. Load base model
base_model = models.FasterRCNNForObjectDetection(
    dataset=datasets.SHIFTDataset
)
base_model.load_from(**vars(base_model.Weights.SHIFT_CLEAR_NATUREYOO))
base_model.to(device)
base_model.eval()

# 2. Create APT configuration
config = APTConfig(
    optim="Adam",
    adapt_lr=1e-4,
    loss_type="smooth_l1",
    conf_threshold=0.5,
)

# 3. Create APT model
adaptive_model = APTPlugin(base_model, config)
adaptive_model.to(device)

# 4. Enable online adaptation
adaptive_model.online()

# 5. Use for inference (adaptation happens automatically)
for batch in dataloader:
    outputs = adaptive_model(batch)
```

### Running the Example

```bash
# Make sure you have the SHIFT dataset downloaded
python apt_example.py
```

## Configuration

### APTConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `optim` | str | "Adam" | Optimizer type (SGD, Adam, AdamW) |
| `adapt_lr` | float | 1e-4 | Learning rate for adaptation |
| `max_age` | int | 3 | Max frames to keep track without detection |
| `min_hits` | int | 1 | Min hits to confirm a track |
| `iou_threshold` | float | 0.3 | IOU threshold for matching |
| `loss_type` | str | "smooth_l1" | Loss function (l1, l2, smooth_l1, giou) |
| `loss_weight` | float | 1.0 | Weight for temporal consistency loss |
| `conf_threshold` | float | 0.5 | Confidence threshold for detections |
| `update_backbone` | bool | True | Whether to update backbone/encoder |
| `update_head` | bool | True | Whether to update detection head |
| `buffer_size` | int | 10 | Frame buffer size |

### Recommended Settings

**For stable adaptation:**
```python
config = APTConfig(
    adapt_lr=1e-4,
    loss_type="smooth_l1",
    conf_threshold=0.5,
    update_backbone=True,
    update_head=True,
)
```

**For aggressive adaptation:**
```python
config = APTConfig(
    adapt_lr=5e-4,
    loss_type="giou",
    conf_threshold=0.3,
    max_age=5,
    iou_threshold=0.2,
)
```

**For conservative adaptation:**
```python
config = APTConfig(
    adapt_lr=5e-5,
    loss_type="smooth_l1",
    conf_threshold=0.7,
    update_backbone=False,  # Only adapt head
    update_head=True,
)
```

## Supported Models

APT is designed to work with various object detection models:

- ✅ **Faster R-CNN** (ResNet-50, Swin Transformer)
- ✅ **RT-DETR** (with HuggingFace integration)
- ✅ **YOLO11** (with Ultralytics integration)
- ⚠️ Other models: May require minor modifications in `extract_detections` method

## Supported Datasets

- ✅ **SHIFT Dataset** (continuous domain shift scenarios)
- ✅ **Cityscapes** (urban driving scenes)
- ⚠️ Custom datasets: Should work with any video-based detection dataset

## Algorithm Details

### Kalman Filter State

The tracker uses a 7-dimensional state vector:
- `[x, y, s, r, vx, vy, vs]`
  - `(x, y)`: Object center position
  - `s`: Scale (area)
  - `r`: Aspect ratio
  - `(vx, vy, vs)`: Velocities

### Temporal Consistency Loss

```python
Loss = Σ Distance(Current_Detection_i, Tracked_Prediction_i)
```

Where:
- Current detections come from the model at frame t
- Tracked predictions are Kalman filter predictions for frame t based on frame t-1
- Distance can be L1, L2, Smooth L1, or GIoU

### Adaptation Strategy

1. **Forward pass**: Model predicts on current frame
2. **Extract detections**: Get high-confidence detections
3. **Update tracker**: Associate detections with tracks
4. **Predict next frame**: Kalman filter predicts object locations
5. **Compute loss**: Match current and predicted boxes
6. **Backward pass**: Update model parameters

## Performance Tips

### 1. Batch Size
- Use `batch_size=1` for online adaptation (sequential frames)
- Larger batches may break temporal consistency

### 2. Learning Rate
- Start with `1e-4`
- Increase to `5e-4` for faster adaptation
- Decrease to `5e-5` for more stable adaptation

### 3. Confidence Threshold
- Higher threshold (0.6-0.7): Cleaner pseudo-labels, slower adaptation
- Lower threshold (0.3-0.4): More noisy labels, faster adaptation
- Recommended: 0.5

### 4. Loss Type
- `smooth_l1`: Most robust (recommended)
- `giou`: Better for box regression but may be unstable
- `l1/l2`: Simpler but less robust

### 5. Update Strategy
- Update both backbone and head for maximum adaptation
- Freeze head to preserve source domain knowledge
- Freeze backbone for faster inference

## Troubleshooting

### Issue: Loss is NaN or exploding
**Solutions:**
- Reduce learning rate (e.g., 5e-5 or 1e-5)
- Increase confidence threshold
- Use `smooth_l1` instead of other losses
- Add gradient clipping in optimizer

### Issue: No adaptation happening
**Solutions:**
- Ensure `no_grad=False` in evaluator
- Verify `adaptive_model.adapting == True`
- Check if detections are above confidence threshold
- Verify gradients are not blocked

### Issue: Model performance degrades
**Solutions:**
- Reduce learning rate
- Decrease loss weight
- Use higher confidence threshold
- Update only backbone (freeze head)

### Issue: Slow inference
**Solutions:**
- Disable adaptation: `adaptive_model.offline()`
- Reduce buffer size
- Use smaller backbone

## Comparison with Other Methods

| Method | Requires Labels | Temporal Info | Online | Continual |
|--------|----------------|---------------|--------|-----------|
| **APT** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| TENT | ❌ No | ❌ No | ✅ Yes | ✅ Yes |
| TTT | ✅ Yes | ❌ No | ✅ Yes | ❌ No |
| CoTTA | ❌ No | ❌ No | ✅ Yes | ✅ Yes |

## Future Work

- [ ] Support for multi-object tracking metrics
- [ ] Integration with more tracking algorithms (DeepSORT, ByteTrack)
- [ ] Memory-efficient implementation for long sequences
- [ ] Support for panoptic segmentation

## Citation

If you use APT in your research, please cite:

```bibtex
@article{apt2024,
  title={APT: Adaptive Plugin for Test-time Adaptation via Temporal Consistency under Continual Domain Shift},
  author={Your Name},
  journal={Conference/Journal},
  year={2024}
}
```

## References

1. **SORT**: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016
2. **Kalman Filter**: Kalman, R.E., "A New Approach to Linear Filtering and Prediction Problems", 1960
3. **SHIFT Dataset**: Sun et al., "SHIFT: A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation", CVPR 2022

## License

This implementation is provided under MIT License.

## Contact

For questions or issues, please open an issue in the repository or contact [your email].

---

**Note**: This is a research implementation. For production use, additional testing and optimization may be required.

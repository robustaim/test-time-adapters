# APT: Adaptive Plugin for Test-time Adaptation via Temporal Consistency

## Overview

**APT (Adaptive Plugin using Temporal Consistency)** is a self-supervised test-time adaptation method designed for object detection models operating under continual domain shift scenarios, such as autonomous driving.

### Key Idea

In autonomous driving scenarios, the environment continuously changes (weather, lighting, etc.). APT leverages **temporal consistency** in video sequences to adapt the model without requiring labeled data. By tracking objects across frames using Kalman filters (inspired by SORT), APT creates **motion-based predictions** from temporally consistent detections and uses them as **self-supervised signals** to adapt the model online.

**Important distinction**: APT uses **motion predictions** (from Kalman filter physics model), NOT pseudo-labels (from model self-predictions). This makes it a **self-supervised** method, not semi-supervised.

### How It Works

1. **Track Objects**: Use Kalman filter to track detected objects across frames
2. **Predict Next Frame**: Predict where objects should appear based on motion physics
3. **Temporal Delay**: Use predictions from frame t-1 as supervision for frame t
4. **Compute Loss**: Compare current detections with motion-based predictions
5. **Adapt Model**: Update model parameters using temporal consistency loss

## Architecture

```
Frame t-1: Detections → Kalman Filter → Motion predictions (stored)
                                              ↓
Frame t:   Model predictions ←--[Loss]--→ Motion predictions (from t-1)
                                              ↓
                                        Backpropagation
                                              ↓
                                        Model Adaptation
                                              ↓
           Detections → Kalman Filter → Motion predictions (for t+1)
```

**Key improvement**: Proper temporal delay ensures that motion predictions from frame t-1 supervise frame t, maintaining true temporal consistency.

## Key Improvements

### 1. Temporal Delay (Most Critical)
- **Problem**: Previous implementation computed loss within same frame
- **Solution**: Motion predictions from frame t-1 used as supervision for frame t
- **Impact**: True temporal consistency, not just spatial matching

### 2. Confidence-Weighted Loss
- **Problem**: All matched boxes weighted equally
- **Solution**: Higher confidence detections contribute more to loss
- **Impact**: More reliable adaptation signal, reduced noise

### 3. Dual Confidence Thresholds
- **Tracker update**: Low threshold (0.3) - use more detections for tracking
- **Loss computation**: High threshold (0.7) - only reliable detections for adaptation
- **Impact**: Better tracking stability + cleaner learning signal

### 4. Loss Stabilization
- **Problem**: Loss scale varies greatly across different conditions
- **Solution**: EMA-based normalization
- **Impact**: More stable training, prevents divergence

### 5. Selective Parameter Updates
- **BatchNorm**: Main adaptation target (scale parameters)
- **FPN last layer**: Optional, very low LR (1e-6)
- **Box regressor**: Optional, very low LR (1e-6)
- **Impact**: Prevents catastrophic forgetting while allowing adaptation

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

The improved implementation is in:
```
ttadapters/methods/regularizers/temporal/
├── __init__.py
├── apt_config.py       # Configuration with new parameters
├── apt_plugin.py       # Main implementation with all improvements
├── kalman_tracker.py   # Kalman filter tracking
└── README.md          # This file
```

## Quick Start

### Basic Usage

```python
import torch
from ttadapters import datasets, models, methods

# 1. Load base model
base_model = models.FasterRCNNForObjectDetection(
    dataset=datasets.SHIFTDataset
)
base_model.load_from(**vars(base_model.Weights.SHIFT_CLEAR_NATUREYOO))
base_model.to(device)
base_model.eval()

# 2. Create APT configuration (IMPROVED)
config = methods.APTConfig(
    # Optimization
    optim="SGD",
    adapt_lr=1e-5,           # Main learning rate
    backbone_lr=1e-6,        # For selective backbone updates
    head_lr=1e-6,            # For selective head updates
    
    # Tracking
    max_age=3,
    min_hits=1,
    iou_threshold=0.8,
    
    # Loss
    loss_type="smooth_l1",
    loss_weight=1.0,
    use_confidence_weighting=True,     # NEW: Weight by confidence
    conf_threshold=0.7,                 # High threshold for loss
    min_confidence_for_update=0.3,     # NEW: Low threshold for tracking
    
    # Update strategy
    update_backbone=False,
    update_head=False,
    update_bn=True,                           # Main adaptation
    update_fpn_last_layer=False,              # NEW: Optional
    update_box_regressor_last_layer=False,    # NEW: Optional
    
    # Stabilization
    buffer_size=500,
    loss_ema_decay=0.9,                # NEW: Loss normalization
)

# 3. Create APT model
adaptive_model = methods.APTPlugin(base_model, config)
adaptive_model.to(device)

# 4. Enable online adaptation
adaptive_model.online()

# 5. Use for inference (adaptation happens automatically)
for batch in dataloader:
    outputs = adaptive_model(batch)
```

### Running the Test Script

```bash
python test_improved_apt.py
```

## Configuration

### APTConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **Optimization** |
| `optim` | str | "SGD" | Optimizer type |
| `adapt_lr` | float | 1e-5 | Main learning rate |
| `backbone_lr` | float | 1e-6 | LR for backbone (if updated) |
| `head_lr` | float | 1e-6 | LR for head (if updated) |
| **Tracking** |
| `max_age` | int | 3 | Max frames without detection |
| `min_hits` | int | 1 | Min hits to confirm track |
| `iou_threshold` | float | 0.8 | IOU threshold for matching |
| **Loss** |
| `loss_type` | str | "smooth_l1" | Loss function |
| `loss_weight` | float | 1.0 | Loss weight |
| `use_confidence_weighting` | bool | True | Weight loss by confidence |
| `conf_threshold` | float | 0.7 | Confidence for loss computation |
| `min_confidence_for_update` | float | 0.3 | Confidence for tracker update |
| **Update Strategy** |
| `update_backbone` | bool | False | Update backbone |
| `update_head` | bool | False | Update detection head |
| `update_bn` | bool | True | Update BatchNorm scale |
| `update_fpn_last_layer` | bool | False | Update FPN last layer |
| `update_box_regressor_last_layer` | bool | False | Update box regressor |
| **Stabilization** |
| `buffer_size` | int | 500 | Frame buffer size |
| `loss_ema_decay` | float | 0.9 | EMA decay for loss scaling |

### Recommended Settings

**For SHIFT dataset (tested):**
```python
config = methods.APTConfig(
    optim="SGD",
    adapt_lr=1e-5,
    iou_threshold=0.8,
    loss_type="smooth_l1",
    conf_threshold=0.7,
    min_confidence_for_update=0.3,
    use_confidence_weighting=True,
    update_bn=True,
    update_backbone=False,
    update_head=False,
)
```

**For more aggressive adaptation (experimental):**
```python
config = methods.APTConfig(
    adapt_lr=1e-5,
    head_lr=1e-6,
    update_bn=True,
    update_fpn_last_layer=True,      # Enable FPN adaptation
    update_box_regressor_last_layer=True,  # Enable box reg adaptation
)
```

## Technical Details

### Self-Supervised Learning (NOT Semi-Supervised)

APT is a **self-supervised** method because:
- Uses **motion predictions** from Kalman filter (physics model)
- NOT pseudo-labels from model predictions (which would be semi-supervised)
- Leverages **temporal structure** as supervision signal
- No human labels required

### Temporal Delay Mechanism

```python
# Frame t-1
detections_t1 = model(frame_t1)
tracker.update(detections_t1)
motion_pred_for_t = tracker.predict()  # Predict frame t

# Frame t
detections_t = model(frame_t)
loss = consistency_loss(detections_t, motion_pred_for_t)  # Use t-1 predictions
loss.backward()
optimizer.step()
tracker.update(detections_t)
motion_pred_for_t1 = tracker.predict()  # Predict frame t+1
```

### Confidence Weighting

```python
loss = Σ (confidence_i × distance(detection_i, prediction_i))
```

Higher confidence detections contribute more to the loss, providing a more reliable adaptation signal.

### Loss Stabilization

```python
# EMA of loss scale
loss_scale_ema = α × loss_scale_ema + (1-α) × current_loss_scale

# Normalized loss
normalized_loss = loss / (1 + n_matched) × loss_weight
```

## Supported Models

APT is designed to work with various object detection models:

- ✅ **Faster R-CNN** (ResNet-50, Swin Transformer) - Tested
- ✅ **RT-DETR** (with HuggingFace integration) - Compatible
- ✅ **YOLO11** (with Ultralytics integration) - Compatible
- ⚠️ Other models: May require modifications in `extract_detections`

## Performance Tips

### 1. Batch Size
- **MUST use `batch_size=1`** for online adaptation
- Larger batches break temporal consistency

### 2. Learning Rate
- **1e-5**: Tested and recommended for BatchNorm-only updates
- 1e-6: If updating FPN or box regressor
- 1e-4: Too high, causes divergence

### 3. Confidence Thresholds
- **High threshold (0.7-0.8)**: For loss computation
- **Low threshold (0.3)**: For tracker update
- This dual-threshold approach balances tracking stability and learning quality

### 4. IOU Threshold
- **0.8**: Tested and recommended
- Higher = stricter matching, cleaner signal
- Lower = more matches, but noisier

### 5. Update Strategy
- **Start with**: `update_bn=True` only
- **If stable**: Add `update_fpn_last_layer=True`
- **If still stable**: Add `update_box_regressor_last_layer=True`
- **Never**: Update full backbone/head (causes catastrophic forgetting)

## Troubleshooting

### Issue: Loss is NaN or exploding
**Solutions:**
- Verify temporal delay is working (check adaptation_steps)
- Ensure confidence thresholds are appropriate
- Try lower learning rate (5e-6)
- Check gradient clipping is enabled

### Issue: No adaptation happening
**Solutions:**
- Verify `no_grad=False` in evaluator
- Check `adaptive_model.adapting == True`
- Ensure sufficient high-confidence detections
- Verify tracker is producing predictions

### Issue: Model performance degrades
**Solutions:**
- Reduce learning rate
- Use stricter confidence threshold (0.8)
- Disable selective updates (only BatchNorm)
- Check loss stabilization (EMA should be stable)

### Issue: Adaptation too slow
**Solutions:**
- Lower confidence threshold for loss (0.6)
- Increase IOU threshold tolerance (0.7)
- Enable confidence weighting
- Check if enough tracks are being maintained

## Comparison with Other Methods

| Method | Supervision | Temporal Info | Online | Continual |
|--------|-------------|---------------|--------|-----------|
| **APT** | Motion predictions | ✅ Yes | ✅ Yes | ✅ Yes |
| TENT | Entropy | ❌ No | ✅ Yes | ✅ Yes |
| TTT | Auxiliary task | ❌ No | ✅ Yes | ❌ No |
| CoTTA | Teacher-Student | ❌ No | ✅ Yes | ✅ Yes |
| MeanTeacher | Pseudo-labels | ❌ No | ❌ No | ❌ No |

**Key distinction**: APT is the only method that uses **motion-based predictions** as self-supervised signals.

## Citation

If you use APT in your research, please cite:

```bibtex
@article{apt2024,
  title={APT: Self-Supervised Test-Time Adaptation via Temporal Consistency under Continual Domain Shift},
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

---

**Note**: This is an improved research implementation with all identified issues fixed. Key improvements include temporal delay, confidence weighting, dual thresholds, loss stabilization, and selective parameter updates.

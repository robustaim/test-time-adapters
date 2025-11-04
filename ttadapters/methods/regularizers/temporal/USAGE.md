# APT: Adaptive Plugin using Temporal Consistency - Usage Example

This notebook demonstrates how to use APT (Adaptive Plugin for Test-time Adaptation via Temporal Consistency) with Faster R-CNN on the SHIFT dataset.

## Overview

APT adapts object detection models at test-time by leveraging temporal consistency between consecutive video frames. It uses Kalman filter-based tracking (inspired by SORT) to predict where objects should appear in the next frame, then uses the discrepancy between predictions and tracked estimates as a self-supervised adaptation signal.

## Key Features

- **Temporal Consistency Loss**: Uses Kalman filter to track objects and predict their locations
- **Self-Supervised Adaptation**: No labels needed during test-time
- **Continual Adaptation**: Designed for continuous domain shift scenarios
- **Model-Agnostic**: Works with any object detection model (Faster R-CNN, YOLO, RT-DETR, etc.)

## Installation

Required packages:
```bash
pip install filterpy scipy
```

## Basic Usage

### 1. Import Required Modules

```python
import torch
from ttadapters import datasets, models
from ttadapters.datasets import DatasetHolder
from ttadapters.utils import visualizer, validator

# Import APT
from apt_config import APTConfig
from apt_plugin import APTPlugin
```

### 2. Load Dataset

```python
DATA_ROOT = "./data"
SOURCE_DOMAIN = datasets.SHIFTDataset

# Load SHIFT dataset
dataset = DatasetHolder(
    train=datasets.SHIFTClearDatasetForObjectDetection(root=DATA_ROOT, train=True),
    valid=datasets.SHIFTClearDatasetForObjectDetection(root=DATA_ROOT, valid=True),
    test=datasets.SHIFTCorruptedDatasetForObjectDetection(root=DATA_ROOT, valid=True)
)

CLASSES = dataset.test.classes
NUM_CLASSES = len(CLASSES)
```

### 3. Load Base Model (Faster R-CNN)

```python
# Initialize Faster R-CNN with ResNet-50 backbone
base_model = models.FasterRCNNForObjectDetection(dataset=SOURCE_DOMAIN)

# Load pretrained weights
load_result = base_model.load_from(
    **vars(base_model.Weights.SHIFT_CLEAR_NATUREYOO),
    strict=False
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
base_model.to(device)
base_model.eval()
```

### 4. Initialize APT

```python
# Configure APT
apt_config = APTConfig(
    # Optimization settings
    optim="Adam",
    adapt_lr=1e-4,
    
    # Temporal tracking settings
    max_age=3,              # Keep tracks for 3 frames without detection
    min_hits=1,             # Confirm track after 1 hit
    iou_threshold=0.3,      # IOU threshold for matching
    
    # Loss settings
    loss_type="smooth_l1",  # Options: l1, l2, smooth_l1, giou
    loss_weight=1.0,
    conf_threshold=0.5,     # Confidence threshold for detections
    
    # Update strategy
    update_backbone=True,   # Adapt backbone/FPN
    update_head=True,       # Adapt detection head
    
    # Memory
    buffer_size=10,         # Frame buffer size
)

# Create APT plugin
adaptive_model = APTPlugin(base_model, apt_config)
adaptive_model.to(device)
```

### 5. Set Online Adaptation Mode

```python
# Set base model to eval mode (for BatchNorm, Dropout, etc.)
base_model.eval()

# Enable online adaptation mode
adaptive_model.online()

print(f"Model: {adaptive_model.model_name}")
print(f"Adaptation enabled: {adaptive_model.adapting}")
```

### 6. Evaluation on Scenarios

```python
from ttadapters.datasets import scenarios

# Prepare data
data_preparation = base_model.DataPreparation(
    datasets.base.BaseDataset(), 
    evaluation_mode=True
)

# Load discrete scenario (e.g., different weather/time conditions)
discrete_scenario = scenarios.SHIFTDiscreteScenario(
    root=DATA_ROOT,
    valid=True,
    order=scenarios.SHIFTDiscreteScenario.WHWPAPER,
    transforms=data_preparation.transforms
)

# Load continuous scenario (gradual domain shift)
continuous_scenario = scenarios.SHIFTContinuousScenario(
    root=DATA_ROOT,
    valid=True,
    order=scenarios.SHIFTContinuousScenario.DEFAULT,
    transforms=data_preparation.transforms
)
```

### 7. Run Evaluation

```python
# Setup evaluator
methods = {
    'Source-Only': base_model,      # No adaptation
    'APT': adaptive_model            # With adaptation
}

DATA_TYPE = torch.float32
BATCH_SIZE = 1  # For online adaptation, typically use batch_size=1

evaluator = validator.DetectionEvaluator(
    list(methods.values()),
    classes=CLASSES,
    data_preparation=data_preparation,
    dtype=DATA_TYPE,
    device=device,
    no_grad=False  # Important: Set False to allow adaptation
)

evaluator_loader_params = dict(
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=data_preparation.collate_fn
)

# Run on discrete scenario
print("\\n=== Discrete Scenario Results ===")
discrete_results = visualizer.visualize_metrics(
    discrete_scenario(**evaluator_loader_params).play(
        evaluator, 
        index=methods.keys()
    )
)

# Run on continuous scenario  
print("\\n=== Continuous Scenario Results ===")
continuous_results = visualizer.visualize_metrics(
    continuous_scenario(**evaluator_loader_params).play(
        evaluator,
        index=methods.keys()
    )
)
```

### 8. View Adaptation Statistics

```python
# Get adaptation statistics
stats = adaptive_model.get_adaptation_stats()
print("\\n=== Adaptation Statistics ===")
print(f"Total adaptation steps: {stats['adaptation_steps']}")
print(f"Average loss: {stats['avg_loss']:.4f}")
print(f"Active tracks: {stats['num_tracks']}")
```

## Configuration Options

### APTConfig Parameters

- **optim**: Optimizer type (`"SGD"`, `"Adam"`, `"AdamW"`)
- **adapt_lr**: Learning rate for adaptation (default: 1e-4)
- **max_age**: Maximum frames to keep track without detection (default: 3)
- **min_hits**: Minimum hits to confirm a track (default: 1)
- **iou_threshold**: IOU threshold for matching detections to tracks (default: 0.3)
- **loss_type**: Loss function type (`"l1"`, `"l2"`, `"smooth_l1"`, `"giou"`)
- **loss_weight**: Weight for temporal consistency loss (default: 1.0)
- **conf_threshold**: Confidence threshold for using detections (default: 0.5)
- **update_backbone**: Whether to update backbone/encoder (default: True)
- **update_head**: Whether to update detection head (default: True)
- **buffer_size**: Number of frames to keep in memory (default: 10)

## Advanced Usage

### Custom Loss Function

```python
# Use GIoU loss for better box regression
apt_config = APTConfig(
    loss_type="giou",
    loss_weight=2.0,  # Increase weight if needed
)
```

### Selective Parameter Updates

```python
# Only adapt the FPN/backbone, keep head frozen
apt_config = APTConfig(
    update_backbone=True,
    update_head=False,
)
```

### Tuning Tracking Parameters

```python
# More aggressive tracking
apt_config = APTConfig(
    max_age=5,          # Keep tracks longer
    min_hits=2,         # Require more hits to confirm
    iou_threshold=0.2,  # More lenient matching
)
```

## Tips for Best Performance

1. **Learning Rate**: Start with 1e-4 and adjust based on your model and dataset
2. **Batch Size**: Use batch_size=1 for online adaptation (sequential frames)
3. **Confidence Threshold**: Higher threshold (0.5-0.7) for cleaner pseudo-labels
4. **Loss Type**: 
   - `smooth_l1` is robust to outliers (recommended)
   - `giou` is better for box regression but more sensitive
5. **Update Strategy**: 
   - Update both backbone and head for maximum adaptation
   - Freeze head if you want to preserve source domain knowledge

## Troubleshooting

### Issue: Loss is NaN or too high
- Reduce learning rate
- Increase confidence threshold
- Check if tracker is producing valid predictions

### Issue: No adaptation happening
- Ensure `no_grad=False` in evaluator
- Verify `adaptive_model.adapting == True`
- Check if sufficient high-confidence detections exist

### Issue: Model diverges
- Reduce learning rate
- Use smaller loss weight
- Consider updating only backbone

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

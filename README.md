# test-time-adapters
Pluggable Test-time Adapter Implementations

> This repository suggests a pluggable implementation of Test-Time Adaptation (TTA) module.
> It is designed to be easily integrated into existing Transformer-family models, enhancing their capabilities to adapt during inference time under the distribution shift.


## Academic History
- [] TENT
- [] TTT
- [] When, Where, and How to Adapt?


## Suggestions
### APT: Adaptive Plugin for TTA (Test-time Adaptation)
<img src="./docs/images/apt_structure.svg">

#### Performance Metrics
| Method | Dataset | Metric      | Value |
|--------|---------|-------------|------|
| APT    | SHIFT   | mAP (50-95) | ???  |


## Usage
### Installation (Use this repository as a package for your own project)
```bash
pip install git+https://github.com/robustaim/test-time-adapters.git
```

### Reproduction of Results
#### Environment Setup
```bash
git clone https://github.com/robustaim/test-time-adapters.git ptta
cd ptta
uv sync
```

#### Run Batch Experiments
```bash
python example.py
```

#### Apply to Your Own Model
```python
from ttadapters.methods import APTConfig, AdaptationPlugin
```

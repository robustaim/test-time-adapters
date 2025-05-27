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


## Citation
```
@InProceedings{shift2022,
    author    = {Sun, Tao and Segu, Mattia and Postels, Janis and Wang, Yuxuan and Van Gool, Luc and Schiele, Bernt and Tombari, Federico and Yu, Fisher},
    title     = {{SHIFT:} A Synthetic Driving Dataset for Continuous Multi-Task Domain Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {21371-21382}
}
```
```
@misc{lv2023detrs,
      title={DETRs Beat YOLOs on Real-time Object Detection},
      author={Yian Zhao and Wenyu Lv and Shangliang Xu and Jinman Wei and Guanzhong Wang and Qingqing Dang and Yi Liu and Jie Chen},
      year={2023},
      eprint={2304.08069},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```
@inproceedings{wolf-etal-2020-transformers,
    title = "Transformers: State-of-the-Art Natural Language Processing",
    author = "Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = oct,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-demos.6",
    pages = "38--45"
}
```

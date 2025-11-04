import os
import warnings
from pathlib import Path


try:
    from ultralytics import settings
    from ultralytics.cfg import get_cfg

    from ultralytics.utils import nms, ops, LOGGER
    from ultralytics.utils.instance import Instances

    from ultralytics.data import build_dataloader
    from ultralytics.data.augment import Compose, v8_transforms, LetterBox, Format

    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.models.yolo.detect import DetectionTrainer

    from ultralytics.utils.checks import check_amp
    from ultralytics.engine import trainer as _trainer


    settings['runs_dir'] = str(Path(".") / "results" / "runs")
    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    cache_dir.mkdir(exist_ok=True)


    def new_check_amp(*args, **kwargs):
        original_dir = Path.cwd()
        try:
            os.chdir(cache_dir)
            return check_amp(*args, **kwargs)
        finally:
            os.chdir(original_dir)


    _trainer.check_amp = new_check_amp
except ImportError:
    class DummyDetectionModel:
        """Dummy model that provides helpful installation instructions."""

        def __init__(self, cfg="yolo11n.yaml", *args, **kwargs):
            self.model_name = cfg.replace(".yaml", "")
            msg = (
                f"\n{'='*70}\n"
                f"YOLO model '{self.model_name}' requires Ultralytics library.\n"
                f"{'='*70}\n\n"
                f"To use YOLO models, install Ultralytics:\n"
                f"    pip install ultralytics"
                f"or"
                f"    uv pip install ultralytics\n\n"
                f"Note: Ultralytics is licensed under AGPL-3.0.\n"
                f"By installing it, you agree to comply with AGPL-3.0 terms.\n"
                f"See: https://github.com/ultralytics/ultralytics\n"
                f"{'='*70}\n"
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        def load_from(self, *args, **kwargs):
            """Silently skip loading when Ultralytics is not installed."""
            return {"missing_keys": [], "unexpected_keys": []}

        def __call__(self, *args, **kwargs):
            raise RuntimeError(
                f"Cannot run YOLO model '{self.model_name}'. "
                f"Install ultralytics first: pip install ultralytics"
            )

        def predict(self, *args, **kwargs):
            raise RuntimeError(
                f"Cannot run YOLO model '{self.model_name}'. "
                f"Install ultralytics first: pip install ultralytics"
            )

        def __repr__(self):
            return f"DummyDetectionModel(model_name='{self.model_name}', installed=False)"


    class DummyDetectionTrainer:
        pass


    nms, ops, LOGGER = None, None, None
    get_cfg, build_dataloader, Instances, Compose, v8_transforms, LetterBox, Format = [lambda: None] * 7
    
    DetectionModel = DummyDetectionModel
    DetectionTrainer = DummyDetectionModel

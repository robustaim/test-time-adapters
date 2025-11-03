import os
import warnings
from pathlib import Path

cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
cache_dir.mkdir(exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(cache_dir)

try:
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import nms, ops
    from ultralytics.utils.instance import Instances
    from ultralytics.data.augment import Compose, v8_transforms, LetterBox, Format

    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.models.yolo.detect import DetectionTrainer
except ImportError:
    class DummyDetectionModel:
        """Dummy model that provides helpful installation instructions."""

        def __init__(self, model_name: str = "yolo11n"):
            self.model_name = model_name
            self._show_install_message()

        def _show_install_message(self):
            msg = (
                f"\n{'='*70}\n"
                f"YOLO model '{self.model_name}' requires Ultralytics library.\n"
                f"{'='*70}\n\n"
                f"To use YOLO models, install Ultralytics:\n"
                f"    pip install ultralytics\n\n"
                f"Note: Ultralytics is licensed under AGPL-3.0.\n"
                f"By installing it, you agree to comply with AGPL-3.0 terms.\n"
                f"See: https://github.com/ultralytics/ultralytics\n"
                f"{'='*70}\n"
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

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
            return f"DummyYOLO(model_name='{self.model_name}', installed=False)"


    class DummyDetectionTrainer:
        pass


    get_cfg = None
    nms, ops, Instances, Compose, v8_transforms, LetterBox, Format = [None] * 7
    
    DetectionModel = DummyDetectionModel
    DetectionTrainer = DummyDetectionModel

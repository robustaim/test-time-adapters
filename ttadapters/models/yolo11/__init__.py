import sys
from . import modeling_yolo11 as __modeling_yolo11
from .modeling_yolo11 import YOLO11ForObjectDetection


sys.modules['ttadapters.models.yolo11.modelings'] = __modeling_yolo11

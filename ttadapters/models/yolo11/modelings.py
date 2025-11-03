from ultralytics.nn.tasks import DetectionModel

from ..base import BaseModel, ModelProvider, WeightsInfo
from ...datasets import BaseDataset


class YOLO11ForObjectDetection(DetectionModel, BaseModel):
    model_name = "YOLO11"
    model_config = "yolo11m.yaml"
    model_provider = ModelProvider.Ultralytics
    channel = 3

    class Weights:
        COCO = WeightsInfo("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt", weight_key="model")
        SHIFT_CLEAR = WeightsInfo("")

    def __init__(self, dataset: BaseDataset):
        nc = len(dataset.classes)
        super().__init__(self.model_config, ch=self.channel, nc=nc)

        self.dataset_name = dataset.dataset_name
        self.num_classes = nc

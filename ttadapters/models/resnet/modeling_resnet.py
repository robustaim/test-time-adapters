from transformers import ResNetForImageClassification as _ResNetForImageClassification, ResNetConfig
from transformers import AutoImageProcessor
import torch

from ..base import BaseModel, ModelProvider, WeightsInfo, ImageClassificationMixin
from ...datasets import BaseDataset, DataPreparation


class ResNetDataPreparation(DataPreparation):
    model_id = "microsoft/resnet-50"

    def __init__(
        self,
        dataset: BaseDataset,
        dataset_key: dict = {},
        img_size: int = 224,
        evaluation_mode: bool = False,
    ):
        self.dataset_name = dataset.dataset_name
        self.classes = dataset.classes

        self.dataset = dataset
        self.dataset_key = dataset_key
        self.evaluation_mode = evaluation_mode

        if self.evaluation_mode:
            pass
        else:
            raise NotImplementedError(f"{self.__class__.__name__} is not yet implemented for training mode.")
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, size=img_size, do_resize=True)

    def transforms(self, *args, idx=None, **kwargs):
        return args

    def pre_process(self, batch, **kwargs):
        images = self.image_processor(images=batch[0], return_tensors="pt")["pixel_values"]
        labels = torch.tensor(batch[1])
        return images, labels

    def post_process(self, batch, **kwargs):
        return batch

    def __getitem__(self, idx):
        return self.transforms(self.dataset[idx], idx=idx)

    def collate_fn(self, batch):
        images, labels = zip(*batch)
        return self.pre_process((images, labels))


class ResNetForImageClassification(_ResNetForImageClassification, BaseModel, ImageClassificationMixin):
    model_id = "microsoft/resnet-50"
    model_name = "ResNet-50"
    model_provider = ModelProvider.HuggingFace
    DataPreparation = ResNetDataPreparation

    class ModelRegistry:
        IMAGENET1k = WeightsInfo("microsoft/resnet-50")

    def __init__(self, config: ResNetConfig | None = None, dataset: BaseDataset | str = "", **kwargs):
        if dataset:
            num_classes = len(dataset.classes)
        else:
            num_classes = config.num_labels if config is not None else 1000  # default to IMAGENET-1k

        if config is None:
            config = ResNetConfig.from_pretrained(self.model_id)
        config.num_labels = num_classes  # override num_labels
        super().__init__(**kwargs)

        self.num_classes = num_classes

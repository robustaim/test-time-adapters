import os
from typing import Optional, Union

from transformers import RTDetrConfig, RTDetrForObjectDetection

from ...datasets import MaskedImageList
from ..base import BaseModel


def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return dict(
        pixel_values=MaskedImageList.from_tensors(images, size_divisibility=32),
        labels=[dict(
            class_labels=item['boxes2d_classes'].long(),
            boxes=item["boxes2d"].float()
        ) for item in targets]
    )


class HFRTDetrForObjectDetection(RTDetrForObjectDetection, BaseModel):
    model_id = "PekingU/rtdetr_r50vd"
    model_name = "RT-DETR-R50"

    def __init__(self, config: RTDetrConfig | None = None, num_labels: int = 80):  # 80 is the default for COCO
        if config is None:
            config = RTDetrConfig.from_pretrained(self.model_id)
            config.num_labels = num_labels
        super().__init__(config=config)

    @classmethod
    def from_pretrained(
        cls,
        num_labels: int = 80,  # 80 is the default for COCO
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ) -> 'RTDetr50ForObjectDetection':
        model = RTDetr50ForObjectDetection(num_labels=num_labels)
        state_dict = RTDetrForObjectDetection.from_pretrained(cls.model_id).state_dict()
        model.load_state_dict(state_dict, strict=False)
        return model

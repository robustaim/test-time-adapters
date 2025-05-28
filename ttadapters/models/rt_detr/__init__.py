import os
from typing import Optional, Union

from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrImageProcessorFast

from ..base import BaseModel


class RTDetr50ForObjectDetection(RTDetrForObjectDetection, BaseModel):
    model_id = "PekingU/rtdetr_r50vd"
    model_name = "RT-DETR-R50"
    image_processor = RTDetrImageProcessorFast.from_pretrained(model_id)

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

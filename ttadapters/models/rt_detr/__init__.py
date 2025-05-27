import os
from typing import Optional, Union

from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrImageProcessorFast

from ..base import BaseModel


class RTDetr50ForObjectDetection(RTDetrForObjectDetection, BaseModel):
    model_id = "PekingU/rtdetr_r50vd"
    model_name = "RT-DETR-R50"
    image_processor = RTDetrImageProcessorFast.from_pretrained(model_id)

    def __init__(self, num_labels: int = 80):  # 80 is the default for COCO
        config = RTDetrConfig.from_pretrained(self.model_id)
        config.num_labels = num_labels
        super().__init__(config=config)

    @classmethod
    def from_pretrained(
        cls,
        num_labels: int = 80,  # 80 is the default for COCO
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ) -> 'RTDetr50ForObjectDetection':
        target_model = RTDetr50ForObjectDetection(num_labels)
        source_model = RTDetrForObjectDetection.from_pretrained(cls.model_id)

        source_dict = source_model.state_dict()
        target_dict = target_model.state_dict()

        copied_count = 0
        for name, source_weight in source_dict.items():
            if name in target_dict:
                target_weight = target_dict[name]

                if source_weight.shape == target_weight.shape:
                    target_dict[name] = source_weight.clone()
                    copied_count += 1

        target_model.load_state_dict(target_dict)
        print(f"Total copied layers: {copied_count}/{len(source_dict)}")

        return target_model

from torch.utils.data import DataLoader
from typing import Optional, Callable
import math

from .. import (
    SHIFTDiscreteSubsetForObjectDetection
)


SubsetType = SHIFTDiscreteSubsetForObjectDetection.SubsetType


class SHIFTDiscreteScenario(dict):
    keys = [
        SubsetType.CLEAR_DAYTIME,  # same as NORMAL
        SubsetType.CLEAR_NIGHT,
        SubsetType.CLEAR_DAWN,
        SubsetType.CLOUDY_DAYTIME,
        SubsetType.OVERCAST_DAYTIME,
        SubsetType.FOGGY_DAYTIME,
        SubsetType.RAINY_DAYTIME,
    ]

    def __init__(
        self, root: str, force_download: bool = False,
        train: bool = True, valid: bool = False, exclude_list: Optional[SubsetType] = None,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None
    ):
        super().__init__(self)

        for key in self.keys:
            if key in exclude_list:
                continue

            self[key] = SHIFTDiscreteSubsetForObjectDetection(
                root=root, force_download=force_download,
                train=train, valid=valid, subset_type=key,
                transform=transform, target_transform=target_transform, transforms=transforms
            )

    def load(self, **kwargs):
        self._play_config = kwargs
        return self

    def play(self, script: Callable, **kwargs):
        result = {}
        for key, dataset in self.items():
            loader = DataLoader(dataset, **self._play_config)
            loader_len = math.ceil(len(dataset)/loader.batch_size)
            bench = script(loader, loader_len, **kwargs)
            result[key] = bench

        result_list = list(result.values())
        result_mean = {
            key: sum(d[key] for d in result_list) / len(result_list)
            for key in result_list[0].keys()
        }

        del self._play_config
        return result, result_mean

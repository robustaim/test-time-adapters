from torch.utils.data import DataLoader
from typing import Optional, Callable
from tqdm.auto import tqdm
import math

from .. import (
    SHIFTDiscreteSubsetForObjectDetection,
    SHIFTContinuousSubsetForObjectDetection,
    SHIFTContinuous10SubsetForObjectDetection,
    SHIFTContinuous100SubsetForObjectDetection
)


DiscreteSubsetType = SHIFTDiscreteSubsetForObjectDetection.SubsetType
ContinuousSubsetType = SHIFTContinuousSubsetForObjectDetection.ContinuousSubsetType


class SHIFTDiscreteScenario(dict):
    keys = [
        DiscreteSubsetType.CLEAR_DAYTIME,  # same as NORMAL
        DiscreteSubsetType.CLEAR_NIGHT,
        DiscreteSubsetType.CLEAR_DAWN,
        DiscreteSubsetType.CLOUDY_DAYTIME,
        DiscreteSubsetType.OVERCAST_DAYTIME,
        DiscreteSubsetType.FOGGY_DAYTIME,
        DiscreteSubsetType.RAINY_DAYTIME,
    ]
    description = "SHIFT Discrete Scenario"

    def __init__(
        self, root: str, force_download: bool = False,
        train: bool = True, valid: bool = False, exclude_list: Optional[DiscreteSubsetType] = None,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None
    ):
        super().__init__(self)

        for key in SHIFTDiscreteScenario.keys:
            if exclude_list is not None and key in exclude_list:
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
        for key, dataset in tqdm(self.items(), desc=self.description):
            loader = DataLoader(dataset, **self._play_config)
            loader_len = math.ceil(len(dataset)/loader.batch_size)
            bench = script(key.value, loader, loader_len, **kwargs)
            result[key] = bench

        result_list = list(result.values())
        result_mean = {
            key: sum(d[key] for d in result_list) / len(result_list)
            for key in result_list[0].keys()
        }

        del self._play_config
        return result, result_mean


class SHIFTContinuousScenario(SHIFTDiscreteScenario):
    pass

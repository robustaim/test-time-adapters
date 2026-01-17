from typing import Optional, Callable

from .base import BaseGradualTTAScenario, ScenarioType
from .. import (
    SHIFTContinuousSubsetForObjectDetection,
    SHIFTContinuous10SubsetForObjectDetection,
    SHIFTContinuous100SubsetForObjectDetection
)


ContinuousSubsetType = SHIFTContinuousSubsetForObjectDetection.ContinuousSubsetType


class SHIFTContinuousSubsetAggregationForObjectDetection(tuple):
    subclasses = [
        SHIFTContinuousSubsetForObjectDetection,
        SHIFTContinuous10SubsetForObjectDetection,
        SHIFTContinuous100SubsetForObjectDetection,
    ]

    def __init__(
        self, root: str, force_download: bool = False,
        train: bool = True, valid: bool = False, subset_type: ContinuousSubsetType = ContinuousSubsetType.DAYTIME_TO_NIGHT,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None
    ):
        super().__init__()
        self.datasets = []
        error = None
        for subclass in self.subclasses:
            try:
                print(subset_type)
                self.datasets.append(subclass(
                    root=root, force_download=force_download,
                    train=train, valid=valid, subset_type=subset_type,
                    transform=transform, target_transform=target_transform, transforms=transforms
                ))
            except Exception as e:
                error = e
                self.datasets.append([])
        if error is not None and all(len(d) == 0 for d in self.datasets):
            raise error

        self._lengths = [len(d) for d in self.datasets]

    def __repr__(self):
        return f"{self.__class__.__name__}({self._lengths})"

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return sum(self._lengths)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index):
        total_len = sum(self._lengths)
        index = index % total_len
        if index < self._lengths[0]:
            return self.datasets[0][index]
        elif index < self._lengths[0] + self._lengths[1]:
            return self.datasets[1][index - self._lengths[0]]
        else:
            return self.datasets[2][index - self._lengths[0] - self._lengths[1]]


class SHIFTContinuousScenarioForGradualTTA(BaseGradualTTAScenario):
    DEFAULT = [
        ContinuousSubsetType.DAYTIME_TO_NIGHT,
        ContinuousSubsetType.CLEAR_TO_FOGGY,
        ContinuousSubsetType.CLEAR_TO_RAINY,
    ]
    WHWPAPER = [
        ContinuousSubsetType.CLEAR_TO_FOGGY,
        ContinuousSubsetType.CLEAR_TO_RAINY,
    ]
    description = "SHIFT-Continuous Scenario For GradualTTA"
    dataset_class = SHIFTContinuousSubsetAggregationForObjectDetection

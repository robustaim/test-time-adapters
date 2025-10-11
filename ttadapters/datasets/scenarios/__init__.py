from typing import Optional, Callable
from enum import Enum

from torch.utils.data import DataLoader

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
    DEFAULT = [
        DiscreteSubsetType.CLEAR_DAYTIME,  # same as NORMAL
        DiscreteSubsetType.CLEAR_NIGHT,
        DiscreteSubsetType.CLEAR_DAWN,
        DiscreteSubsetType.CLOUDY_DAYTIME,
        DiscreteSubsetType.OVERCAST_DAYTIME,
        DiscreteSubsetType.FOGGY_DAYTIME,
        DiscreteSubsetType.RAINY_DAYTIME,
        DiscreteSubsetType.CLEAR_DAYTIME  # same as NORMAL
    ]
    WHWPAPER = [
        DiscreteSubsetType.CLOUDY_DAYTIME,
        DiscreteSubsetType.OVERCAST_DAYTIME,
        DiscreteSubsetType.FOGGY_DAYTIME,
        DiscreteSubsetType.RAINY_DAYTIME,
        DiscreteSubsetType.CLEAR_DAWN,
        DiscreteSubsetType.CLEAR_NIGHT,
        DiscreteSubsetType.CLEAR_DAYTIME  # same as NORMAL
    ]
    description = "SHIFT Discrete Scenario"
    dataset_class = SHIFTDiscreteSubsetForObjectDetection

    def __init__(
        self, root: str, force_download: bool = False, order: Optional[list[DiscreteSubsetType]] = None,
        train: bool = True, valid: bool = False, exclude_list: Optional[DiscreteSubsetType] = None,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None
    ):
        super().__init__(self)
        self.order = order if order else self.DEFAULT

        for key in self.order:
            if exclude_list is not None and key in exclude_list:
                continue

            self[key] = self.dataset_class(
                root=root, force_download=force_download,
                train=train, valid=valid, subset_type=key,
                transform=transform, target_transform=target_transform, transforms=transforms
            )

    def __call__(self, *args, **kwargs):
        return self.load(*args, **kwargs)

    def load(self, **kwargs):
        self._play_config = kwargs
        return self

    def play(self, script: Callable, index: list | None = None, **kwargs):
        if index is None:
            index = ["Trial"]  # single model
        result = [{} for _ in range(len(index))]

        for key, dataset in tqdm(self.items(), desc=self.description):
            loader = DataLoader(dataset, **self._play_config)
            loader_len = math.ceil(len(dataset)/loader.batch_size)
            bench = script(key.value, loader, loader_len, **kwargs)
            if not isinstance(bench, list):
                bench = [bench]
            for res, b in zip(result, bench):
                res[key] = b
            yield result, index

        for res in result:
            res_list = list(res.values())
            res_mean = {
                key: sum(d[key] for d in res_list) / len(res_list)
                for key in res_list[0].keys()
            }
            res["avg"] = res_mean

        del self._play_config
        yield result, index


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


class SHIFTContinuousScenario(SHIFTDiscreteScenario):
    DEFAULT = [
        ContinuousSubsetType.DAYTIME_TO_NIGHT,
        ContinuousSubsetType.CLEAR_TO_FOGGY,
        ContinuousSubsetType.CLEAR_TO_RAINY,
    ]
    WHWPAPER = [
        ContinuousSubsetType.CLEAR_TO_FOGGY,
        ContinuousSubsetType.CLEAR_TO_RAINY,
    ]
    description = "SHIFT Continuous Scenario"
    dataset_class = SHIFTContinuousSubsetAggregationForObjectDetection

    def __init__(
        self, root: str, force_download: bool = False, order: Optional[list[ContinuousSubsetType]] = None,
        train: bool = True, valid: bool = False, exclude_list: Optional[ContinuousSubsetType] = None,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None
    ):
        super().__init__(
            root=root, force_download=force_download, order=order,
            train=train, valid=valid, exclude_list=exclude_list,
            transform=transform, target_transform=target_transform, transforms=transforms
        )

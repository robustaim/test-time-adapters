from typing import Optional, Callable
from enum import Enum

from torch.utils.data import DataLoader

from tqdm.auto import tqdm
import math

from .. import BaseDataset


class ScenarioType(Enum):
    STANDARD = "standard"
    GRADUAL = "gradual"
    CONTINUAL = "continual"
    UNIVERSAL = "universal"


class BaseScenario(dict):
    description = "Base Scenario"
    dataset_class = BaseDataset
    scenario_type: ScenarioType = ScenarioType.CONTINUAL

    DEFAULT = []

    def __init__(
        self, root: str, force_download: bool = False, order: Optional[list] = None,
        train: bool = True, valid: bool = False, exclude_list: Optional[list] = None,
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


class BaseContinualTTAScenario(BaseScenario):
    scenario_type = ScenarioType.CONTINUAL


class BaseGradualTTAScenario(BaseScenario):
    scenario_type = ScenarioType.GRADUAL


class BaseStandardTTAScenario(BaseScenario):
    scenario_type = ScenarioType.STANDARD


class BaseUniversalTTAScenario(BaseScenario):
    scenario_type = ScenarioType.UNIVERSAL

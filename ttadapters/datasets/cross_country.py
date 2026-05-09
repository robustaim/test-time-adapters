"""
CrossCountry Dataset Wrapper.

Unified dataset that dispatches to a per-country sub-dataset selected via the
`corruption_type` (here a domain-key) parameter:

    BASE -> CityScapes  (Germany)
    BDD  -> BDD100k     (USA)
    IDD  -> IDD         (India)

The wrapper exists so the `BaseScenario.__init__` machinery can construct
each step of `CrossCountryScenarioForContinualTTA` with the same call shape
used by other scenarios (it passes `corruption_type=key`).

All sub-datasets enforce the CityScapes-compatible label set (person, rider,
car, truck, bus, train, motorcycle, bicycle), so per-step indices align.
"""
from typing import Callable, Optional
from enum import Enum

from cityscapesscripts.helpers.labels import labels as cs_labels

from .base import BaseDataset
from .cityscapes import CityScapesDatasetForObjectDetection
from .bdd100k import BDD100kDatasetForObjectDetection
from .idd import IDDDatasetForObjectDetection


class CrossCountryDataset(BaseDataset):
    dataset_name = "CrossCountry"

    class DomainType(Enum):
        BASE = "base"   # CityScapes (Germany)
        BDD = "bdd"     # BDD100k    (USA)
        IDD = "idd"     # IDD        (India)

    # Subclasses fill this in.
    DOMAIN_DATASETS: dict = {}

    # detection classes (CityScapes things-only) - kept identical across all
    # per-domain sub-datasets so indexing stays consistent.
    categories = [
        {"id": l.id, "name": l.name, "supercategory": l.category}
        for l in cs_labels if l.hasInstances and not l.ignoreInEval
    ]
    classes = [l.name for l in cs_labels if l.hasInstances and not l.ignoreInEval]
    class_ids = [l.id for l in cs_labels if l.hasInstances and not l.ignoreInEval]

    def __init__(
        self, root: str, force_download: bool = False,
        train: bool = True, valid: bool = False,
        corruption_type: Optional["CrossCountryDataset.DomainType"] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__()
        if not self.DOMAIN_DATASETS:
            raise NotImplementedError(
                f"{type(self).__name__} must define DOMAIN_DATASETS. Use "
                "CrossCountryDatasetForObjectDetection (or another concrete "
                "subclass) instead of CrossCountryDataset directly."
            )
        if corruption_type is None:
            corruption_type = self.DomainType.BASE
        if corruption_type not in self.DOMAIN_DATASETS:
            raise ValueError(
                f"Unsupported domain {corruption_type!r} for {type(self).__name__}. "
                f"Available: {list(self.DOMAIN_DATASETS)}"
            )
        self.corruption_type = corruption_type

        ds_cls = self.DOMAIN_DATASETS[corruption_type]
        self._dataset = ds_cls(
            root=root, force_download=force_download,
            train=train, valid=valid,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        return self._dataset[index]


class CrossCountryDatasetForObjectDetection(CrossCountryDataset):
    DOMAIN_DATASETS = {
        CrossCountryDataset.DomainType.BASE: CityScapesDatasetForObjectDetection,
        CrossCountryDataset.DomainType.BDD: BDD100kDatasetForObjectDetection,
        CrossCountryDataset.DomainType.IDD: IDDDatasetForObjectDetection,
    }

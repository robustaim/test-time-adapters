from typing import Callable, Optional
from collections import defaultdict
from enum import Enum
from os import path

from torchvision import datasets
import numpy as np

from .base import BaseDataset


class ACDCDataset(datasets.ImageFolder, BaseDataset):
    download_method = datasets.utils.download_and_extract_archive
    extract_method = datasets.utils.extract_archive
    base_url = "https://acdc.vision.ee.ethz.ch/api/getPackageUri/"
    download_urls = dict(
        detection=dict(
            id="6436eab79880d97633275d1b",
            name="gt_detection_trainval.zip",
            directory="gt_detection",
            description="Ground-truth bounding box annotations for object detection for train and val sets (2006 images)",
        ),
        panoptic_segmentation=dict(
            id="6436ec0f9880d97633275d71",
            name="gt_panoptic_trainval.zip",
            directory="gt_panoptic",
            description="Ground-truth annotations  for panoptic segmentation for train and val sets (2006 images)",
        ),
        semantic_segmentation=dict(
            id="6436eeae9880d97633275dc9",
            name="gt_trainval.zip",
            directory="gt",
            description="Ground-truth annotations for semantic segmentation and uncertainty-aware semantic segmentation for train and val sets (2006 images)",
        ),
        images=dict(
            id="6436f2259880d97633275dfc",
            name="rgb_anon_trainvaltest.zip",
            directory="rgb_anon",
            description="Anonymized adverse-condition images for train, val, and test sets distributed evenly among fog, night, rain, and snow (4006 images) and anonymized corresponding normal-condition images for train, val, and test sets (4006 images)",
        )
    )
    dataset_name = "ACDC"

    class SubsetType(Enum):
        FOG = "fog"
        NIGHT = "night"
        RAIN = "rain"
        SNOW = "snow"
        NORMAL = "normal"

    def __init__(
        self, root: str, force_download: bool = False,
        train: bool = True, valid: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ):
        self.root = path.join(root, self.dataset_name)
        self.download(self.root, force=force_download)
        self.train, self.valid = train, valid

        if train:
            self.root = path.join(self.root, "val") if valid else path.join(self.root, "train")
        else:
            self.root = path.join(self.root, "test")

        super().__init__(root=self.root)
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self._load_data()

    def _load_data(self):
        sequences = defaultdict(list)
        for idx, (img_path, _) in enumerate(self.samples):
            seq_name = path.dirname(img_path)
            sequences[seq_name].append((idx, img_path))

        for seq_name in sequences:
            sequences[seq_name].sort(key=lambda x: x[1])

        targets = []
        for seq_name, frames in sequences.items():
            gt_path = path.join(seq_name, 'groundtruth.txt')
            if path.exists(gt_path):
                targets.extend(np.loadtxt(gt_path, delimiter=','))
        if self.train:
            self.samples = [(s[0], t) for s, t in zip(self.samples, targets)]
        else:
            start = (self.samples[0], targets)
            follows = [(s[0], [0, 0, 0, 0]) for s in self.samples[1:]]
            self.samples = [start] + follows

    @classmethod
    def download(cls, root: str, force: bool = False, download_key=("images", "detection", "panoptic_segmentation", "semantic_segmentation")):
        print(f"INFO: Downloading '{cls.dataset_name}' from https://acdc.vision.ee.ethz.ch to {root}...")
        for key in download_key:
            file_name = cls.download_urls[key]["name"]
            download_url = cls.base_url + cls.download_urls[key]["id"]
            extract_dir = cls.download_urls[key]["directory"]
            downloaded = path.isfile(path.join(root, file_name))
            extracted = path.isdir(path.join(root, extract_dir))
            if force or not (downloaded or extracted):
                cls.download_method(download_url, download_root=root, extract_root=root, filename=file_name)
                print("INFO: Dataset archive downloaded and extracted.")
            else:
                print("INFO: Dataset archive found in the root directory. Skipping download.")
                if not extracted:
                    cls.extract_method(from_path=path.join(root, file_name), to_path=root)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            sample, target = self.transforms(sample, target)

        return sample, target


class ACDCDatasetForObjectDetection(ACDCDataset):
    pass


class ACDCDatasetForPanopticSegmentation(ACDCDataset):
    pass


class ACDCDatasetForSemanticSegmentation(ACDCDataset):
    pass

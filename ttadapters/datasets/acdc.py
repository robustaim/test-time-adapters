from typing import Callable, Optional
from collections import defaultdict
from enum import Enum
from os import path
import json

from torchvision import datasets
from torchvision.io import read_image, ImageReadMode
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from .base import BaseDataset


class ACDCDataset(BaseDataset):
    download_method = datasets.utils.download_and_extract_archive
    extract_method = datasets.utils.extract_archive
    base_url = "https://acdc.vision.ee.ethz.ch/api/getPackageUri/"
    download_urls = dict(
        detection=dict(
            name="gt_detection_trainval.zip",
            directory="gt_detection",
            description="Ground-truth bounding box annotations for object detection for train and val sets (2006 images)",
        ),
        detection_ref=dict(
            name="gt_detection_trainval_ref.zip",
            directory="gt_detection",
            description="Ground-truth bounding box annotations for object detection for half of train_ref and val_ref sets (1003 normal-condition images)",
        ),
        panoptic_segmentation=dict(
            name="gt_panoptic_trainval.zip",
            directory="gt_panoptic",
            description="Ground-truth annotations  for panoptic segmentation for train and val sets (2006 images)",
        ),
        panoptic_segmentation_ref=dict(
            name="gt_panoptic_trainval_ref.zip",
            directory="gt_panoptic",
            description="Ground-truth annotations for panoptic segmentation for half of train_ref and val_ref sets (1003 normal-condition images)",
        ),
        semantic_segmentation=dict(
            name="gt_trainval.zip",
            directory="gt",
            description="Ground-truth annotations for semantic segmentation and uncertainty-aware semantic segmentation for train and val sets (2006 images)",
        ),
        semantic_segmentation_ref=dict(
            name="gt_trainval_ref.zip",
            directory="gt",
            description="Ground-truth annotations for semantic segmentation for half of train_ref and val_ref sets (1003 normal-condition images)",
        ),
        images=dict(
            name="rgb_anon_trainvaltest.zip",
            directory="rgb_anon",
            description="Anonymized adverse-condition images for train, val, and test sets distributed evenly among fog, night, rain, and snow (4006 images) and anonymized corresponding normal-condition images for train, val, and test sets (4006 images)",
        )
    )
    rgb_load_path = "rgb_anon"
    target_load_path = "gt_detection"
    target_json_prefix = "instancesonly"
    dataset_name = "ACDC"

    categories = [
        {'id': 24, 'name': 'person', 'supercategory': 'human'},
        {'id': 25, 'name': 'rider', 'supercategory': 'human'},
        {'id': 26, 'name': 'car', 'supercategory': 'vehicle'},
        {'id': 27, 'name': 'truck', 'supercategory': 'vehicle'},
        {'id': 28, 'name': 'bus', 'supercategory': 'vehicle'},
        {'id': 31, 'name': 'train', 'supercategory': 'vehicle'},
        {'id': 32, 'name': 'motorcycle', 'supercategory': 'vehicle'},
        {'id': 33, 'name': 'bicycle', 'supercategory': 'vehicle'}
    ]  # TODO: dddd
    classes = []
    class_ids = []

    class CorruptionType(Enum):
        FOG = "fog"
        NIGHT = "night"
        RAIN = "rain"
        SNOW = "snow"
        NORMAL = "normal"

    def __init__(
        self, root: str, force_download: bool = False,
        train: bool = True, valid: bool = False, corruption_type: CorruptionType = CorruptionType.FOG,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None
    ):
        super().__init__()
        self.root = path.join(root, self.dataset_name)
        self.download(self.root, force=force_download)
        self.train, self.valid = train, valid

        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        self.loader = lambda path: read_image(path, mode=ImageReadMode.RGB)
        if train:
            if valid:
                self.raw_path = "_".join([self.target_json_prefix, self.root, corruption_type.value, "val", self.target_load_path])
            else:
                self.raw_path = "_".join([self.target_json_prefix, self.root, corruption_type.value, "train", self.target_load_path])
        else:
            self.raw_path = "_".join([self.target_json_prefix, self.root, corruption_type.value, "test", "image_info"])
        self.samples, self.raw = self.load_data(self.raw_path)

    def load_data(self, raw_path) -> tuple[list, list]:
        with open(raw_path, "r", encoding="utf-8") as f:
            targets = json.load(f)

        samples = []

    @classmethod
    def download(
        cls, root: str, force: bool = False, download_key=(
            "images", "detection", "detection_ref",
            "panoptic_segmentation", "panoptic_segmentation_ref",
            "semantic_segmentation", "semantic_segmentation_ref"
        )
    ):
        print(f"INFO: Downloading '{cls.dataset_name}' from https://acdc.vision.ee.ethz.ch to {root}...")
        for key in download_key:
            file_name = cls.download_urls[key]['name']
            download_url = cls.base_url + cls.download_urls[key]['name']
            extract_dir = cls.download_urls[key]['directory']
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
    rgb_load_path = "rgb_anon"
    target_load_path = "gt_detection"
    target_json_prefix = "instancesonly"


class ACDCDatasetForPanopticSegmentation(ACDCDataset):
    rgb_load_path = "rgb_anon"
    target_load_path = "gt_panoptic"
    target_json_prefix = "instancesonly"  # TODO: correct this


class ACDCDatasetForSemanticSegmentation(ACDCDataset):
    rgb_load_path = "rgb_anon"
    target_load_path = "gt"
    target_json_prefix = "instancesonly"  # TODO: correct this

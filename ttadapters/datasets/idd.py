"""
IDD (Indian Driving Dataset) Detection Wrapper.

The IDD project does not provide a programmatic download URL - the official
release at https://idd.insaan.iiit.ac.in/ requires manual registration and
agreement to the dataset terms. This wrapper therefore expects the archive
`idd-detection.tar.gz` to already exist under `<root>/IDD/`, and only handles
extraction + parsing.

Class labels are projected onto the CityScapes instance class set
(person, rider, car, truck, bus, train, motorcycle, bicycle); IDD-only
categories (autorickshaw, animal, vehicle fallback, ...) are dropped, mirroring
the ACDC wrapper's "CityScapes-compatible labels only" rule.
"""
from typing import Callable, Optional
from os import path, makedirs
import xml.etree.ElementTree as ET

import torch
from torchvision import datasets
from torchvision.io import read_image, ImageReadMode
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from cityscapesscripts.helpers.labels import labels as cs_labels

from .base import BaseDataset


# IDD -> CityScapes detection-class mapping.
# Categories outside this dict (autorickshaw, animal, vehicle fallback, traffic
# sign, traffic light, caravan, trailer, ...) are silently dropped.
IDD_TO_CITYSCAPES = {
    "person": "person",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
}


class IDDDataset(BaseDataset):
    extract_method = datasets.utils.extract_archive
    dataset_name = "IDD"
    archive_name = "idd-detection.tar.gz"
    extracted_root = "IDD_Detection"

    categories = [
        {"id": l.id, "name": l.name, "supercategory": l.category}
        for l in cs_labels if l.hasInstances and not l.ignoreInEval
    ]
    classes = [l.name for l in cs_labels if l.hasInstances and not l.ignoreInEval]
    class_ids = [l.id for l in cs_labels if l.hasInstances and not l.ignoreInEval]

    def __init__(
        self, root: str, force_download: bool = False,
        train: bool = True, valid: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__()
        self.root = path.join(root, self.dataset_name)
        self.extract(self.root, force=force_download)

        self.train, self.valid = train, valid
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.loader = lambda p: read_image(p, mode=ImageReadMode.RGB)

        if train:
            split_file = "val.txt" if valid else "train.txt"
        else:
            split_file = "test.txt"
        self.split_file = split_file

        base_dir = path.join(self.root, self.extracted_root)
        self.split_path = path.join(base_dir, split_file)
        self.images_dir = path.join(base_dir, "JPEGImages")
        self.annot_dir = path.join(base_dir, "Annotations")
        self.samples = self._load_samples(self.split_path, self.images_dir, self.annot_dir)

    # ------------------------------------------------------------------
    # Extraction (no download - IDD requires a manual registration flow)
    # ------------------------------------------------------------------
    @classmethod
    def extract(cls, root: str, force: bool = False):
        makedirs(root, exist_ok=True)
        archive_path = path.join(root, cls.archive_name)
        sentinel = path.join(root, f".{cls.archive_name}.done")
        extracted_dir = path.join(root, cls.extracted_root)

        if not force and path.isfile(sentinel) and path.isdir(extracted_dir):
            print(f"INFO: {cls.dataset_name} already extracted. Skipping.")
            return

        if not path.isfile(archive_path):
            raise FileNotFoundError(
                f"IDD archive missing at {archive_path}. IDD does not provide an "
                "automatic download URL. Register and download IDD Detection "
                "(.tar.gz) from https://idd.insaan.iiit.ac.in/, then place the "
                f"archive at {archive_path} and re-run."
            )

        print(f"INFO: Extracting {cls.archive_name} (this may take several minutes)...")
        cls.extract_method(from_path=archive_path, to_path=root)
        open(sentinel, "w").close()
        print(f"INFO: {cls.dataset_name} extraction complete.")

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _load_samples(self, split_path: str, images_dir: str, annot_dir: str) -> list:
        if not path.isfile(split_path):
            raise FileNotFoundError(f"IDD split file missing: {split_path}")

        cs_index = {name: idx for idx, name in enumerate(self.classes)}
        with open(split_path, "r", encoding="utf-8") as f:
            rel_paths = [ln.strip() for ln in f if ln.strip()]

        samples = []
        for img_idx, rel in enumerate(rel_paths):
            img_path = path.join(images_dir, rel + ".jpg")
            xml_path = path.join(annot_dir, rel + ".xml")
            if not (path.isfile(img_path) and path.isfile(xml_path)):
                continue
            parsed = self._parse_voc(xml_path, cs_index)
            if parsed is None:  # malformed XML - skip rather than dilute eval
                continue
            boxes, labels, hw = parsed
            samples.append((img_path, {
                "boxes": boxes, "labels": labels,
                "image_id": img_idx, "hw": hw,
            }))
        return samples

    @staticmethod
    def _parse_voc(xml_path: str, cs_index: dict):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            print(f"WARNING: skipping malformed IDD annotation: {xml_path}")
            return None
        root = tree.getroot()

        size = root.find("size")
        try:
            w = int(size.find("width").text)
            h = int(size.find("height").text)
        except (AttributeError, TypeError, ValueError):
            w = h = 0

        boxes, labels = [], []
        for obj in root.findall("object"):
            name_node = obj.find("name")
            if name_node is None or name_node.text is None:
                continue
            cs_name = IDD_TO_CITYSCAPES.get(name_node.text.strip())
            if cs_name is None:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                x1 = float(bnd.find("xmin").text)
                y1 = float(bnd.find("ymin").text)
                x2 = float(bnd.find("xmax").text)
                y2 = float(bnd.find("ymax").text)
            except (AttributeError, TypeError, ValueError):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(cs_index[cs_name])
        return boxes, labels, (h, w)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, raw_target = self.samples[index]
        sample = self.loader(img_path)
        h, w = raw_target["hw"]
        if h == 0 or w == 0:
            h, w = sample.shape[-2], sample.shape[-1]

        boxes = raw_target["boxes"]
        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            areas = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
            target = {
                "boxes2d": BoundingBoxes(boxes_t, format=BoundingBoxFormat.XYXY, canvas_size=(h, w)),
                "boxes2d_classes": torch.as_tensor(raw_target["labels"], dtype=torch.int64),
                "area": areas,
                "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
            }
        else:
            target = {
                "boxes2d": BoundingBoxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    format=BoundingBoxFormat.XYXY, canvas_size=(h, w),
                ),
                "boxes2d_classes": torch.zeros(0, dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "iscrowd": torch.zeros(0, dtype=torch.int64),
            }
        target["image_id"] = raw_target["image_id"]
        target["original_hw"] = (h, w)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            sample, target = self.transforms(sample, target)
        return sample, target


class IDDDatasetForObjectDetection(IDDDataset):
    """Alias kept for parity with the *ForObjectDetection naming convention."""
    pass

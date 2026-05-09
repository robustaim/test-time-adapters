"""
BDD100K Dataset Wrapper (Kaggle 'solesensei/solesensei_bdd100k' subset).

Auto-download is performed at construction time. The Kaggle public download
endpoint requires authentication; this wrapper tries the Kaggle CLI first
(reads `~/.kaggle/kaggle.json`) and then falls back to a direct download via
`requests.get`. If both fail, it points the user at a manual `curl` command.

Class labels are projected onto the CityScapes instance class set
(person, rider, car, truck, bus, train, motorcycle, bicycle); BDD-only
categories (traffic light, traffic sign, ...) are dropped, mirroring the
ACDC wrapper.
"""
from typing import Callable, Optional
from os import path, makedirs
from pathlib import Path
import subprocess
import shutil
import json

import requests

import torch
from torchvision import datasets
from torchvision.io import read_image, ImageReadMode
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from tqdm.auto import tqdm
from cityscapesscripts.helpers.labels import labels as cs_labels

from .base import BaseDataset


# BDD100K -> CityScapes detection-class mapping.
# Anything outside this dict is dropped (e.g. traffic light/sign, trailer).
BDD_TO_CITYSCAPES = {
    "person": "person",
    "pedestrian": "person",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "bike": "bicycle",
    "bicycle": "bicycle",
}


class BDD100kDataset(BaseDataset):
    extract_method = datasets.utils.extract_archive
    dataset_name = "BDD100k"
    kaggle_id = "solesensei/solesensei_bdd100k"
    archive_name = "solesensei_bdd100k.zip"
    download_url = (
        "https://www.kaggle.com/api/v1/datasets/download/solesensei/solesensei_bdd100k"
    )

    # detection: things only (CityScapes hasInstances + not ignoreInEval)
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
        self.download(self.root, force=force_download)

        self.train, self.valid = train, valid
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.loader = lambda p: read_image(p, mode=ImageReadMode.RGB)

        # BDD100K's public release ships annotations only for train / val.
        if train:
            self.split = "val" if valid else "train"
        else:
            # No public test annotations - reuse val for test-time evaluation.
            print(
                f"WARNING: {self.dataset_name} has no public test annotations; "
                "falling back to the 'val' split for train=False."
            )
            self.split = "val"

        self.label_path = self._find_label_json(self.split)
        self.images_dir = self._find_images_dir(self.split)
        self.samples = self._load_samples(self.label_path, self.images_dir)

    # ------------------------------------------------------------------
    # Filesystem discovery (Kaggle archives ship with a few layout variants)
    # ------------------------------------------------------------------
    def _find_label_json(self, split: str) -> str:
        roots = Path(self.root)
        for pattern in (
            f"bdd100k_labels_images_{split}.json",
            f"det_v2_{split}_release.json",
            f"det_{split}.json",
        ):
            hits = sorted(roots.rglob(pattern))
            if hits:
                return str(hits[0])
        raise FileNotFoundError(
            f"BDD100k labels JSON for split '{split}' not found under {self.root}. "
            f"Check that {self.archive_name} extracted correctly. Expected one of: "
            f"bdd100k_labels_images_{split}.json / det_v2_{split}_release.json"
        )

    def _find_images_dir(self, split: str) -> str:
        roots = Path(self.root)
        # Try canonical Kaggle layouts first (avoids walking ~100k JPEGs via rglob).
        for canonical in (
            roots / "bdd100k" / "images" / "100k" / split,
            roots / "bdd100k_images_100k" / "100k" / split,
            roots / "images" / "100k" / split,
            roots / "100k" / split,
        ):
            if canonical.is_dir() and any(canonical.glob("*.jpg")):
                return str(canonical)
        # Last-resort: locate any '100k/<split>' subtree.
        for parent in roots.rglob("100k"):
            cand = parent / split
            if cand.is_dir() and any(cand.glob("*.jpg")):
                return str(cand)
        raise FileNotFoundError(
            f"BDD100k '{split}' images directory not found under {self.root}. "
            "Expected '<root>/.../100k/{train,val}/*.jpg'."
        )

    def _load_samples(self, label_path: str, images_dir: str) -> list:
        with open(label_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "frames" in raw:
            entries = raw["frames"]
        else:
            entries = raw

        cs_index = {name: idx for idx, name in enumerate(self.classes)}

        samples = []
        for img_idx, entry in enumerate(entries):
            file_name = entry.get("name") or entry.get("file_name")
            if not file_name:
                continue
            img_path = path.join(images_dir, file_name)
            if not path.isfile(img_path):
                continue

            boxes, labels = [], []
            for obj in entry.get("labels", []) or []:
                cat = obj.get("category", "")
                cs_name = BDD_TO_CITYSCAPES.get(cat)
                if cs_name is None:
                    continue
                box = obj.get("box2d")
                if not box:
                    continue
                x1 = box.get("x1")
                y1 = box.get("y1")
                x2 = box.get("x2")
                y2 = box.get("y2")
                if None in (x1, y1, x2, y2) or x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                labels.append(cs_index[cs_name])

            samples.append((img_path, {
                "boxes": boxes, "labels": labels, "image_id": img_idx,
            }))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, raw_target = self.samples[index]
        sample = self.loader(img_path)
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

    # ------------------------------------------------------------------
    # Download + extract
    # ------------------------------------------------------------------
    @classmethod
    def download(cls, root: str, force: bool = False):
        makedirs(root, exist_ok=True)
        archive_path = path.join(root, cls.archive_name)
        sentinel = path.join(root, f".{cls.archive_name}.done")

        already_extracted = path.isfile(sentinel)
        already_archived = path.isfile(archive_path)

        if not force and already_extracted:
            print(f"INFO: {cls.dataset_name} already present. Skipping.")
            return

        if not already_archived:
            print(
                f"INFO: Downloading '{cls.dataset_name}' from Kaggle "
                f"({cls.kaggle_id}) to {root}..."
            )
            cls._fetch_archive(archive_path)

        if not path.isfile(archive_path):
            raise FileNotFoundError(
                f"Expected Kaggle archive at {archive_path}. Manual download:\n"
                f"  curl -L -o {archive_path} {cls.download_url}\n"
                "or use the Kaggle CLI:\n"
                f"  kaggle datasets download -d {cls.kaggle_id} -p {root}"
            )

        print(f"INFO: Extracting {cls.archive_name}...")
        cls.extract_method(from_path=archive_path, to_path=root)
        open(sentinel, "w").close()
        print(f"INFO: {cls.dataset_name} ready under {root}.")

    @classmethod
    def _fetch_archive(cls, archive_path: str):
        # Prefer the Kaggle CLI (handles auth via ~/.kaggle/kaggle.json).
        if shutil.which("kaggle") is not None:
            try:
                subprocess.run(
                    [
                        "kaggle", "datasets", "download",
                        "-d", cls.kaggle_id,
                        "-p", path.dirname(archive_path),
                    ],
                    check=True,
                )
                if path.isfile(archive_path):
                    return
            except subprocess.CalledProcessError as e:
                print(f"WARNING: kaggle CLI download failed: {e}. Trying direct URL.")

        # Direct streaming download (works if the Kaggle URL is publicly
        # fetchable in the caller's environment, e.g. via a cached cookie).
        try:
            with requests.get(cls.download_url, stream=True, allow_redirects=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) or None
                with open(archive_path, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, unit_divisor=1024,
                    desc=cls.archive_name, dynamic_ncols=True, miniters=1, leave=True,
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to download {cls.archive_name} from Kaggle. "
                "Provide credentials at ~/.kaggle/kaggle.json (chmod 600) or "
                f"download manually:\n  curl -L -o {archive_path} {cls.download_url}"
            ) from e


class BDD100kDatasetForObjectDetection(BDD100kDataset):
    """Alias kept for parity with the *ForObjectDetection naming convention."""
    pass

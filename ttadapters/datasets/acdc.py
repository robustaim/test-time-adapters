from typing import Callable, Optional
from collections import defaultdict
from os import path, makedirs
from pathlib import Path
from enum import Enum
import requests
import json

import torch
from torchvision import datasets
from torchvision.io import read_image, ImageReadMode
from torchvision.tv_tensors import BoundingBoxes, BoundingBoxFormat

from tqdm.auto import tqdm
from cityscapesscripts.helpers.labels import labels as cs_labels

from .base import BaseDataset


class ACDCDataset(BaseDataset):
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
            description="Ground-truth annotations for panoptic segmentation for train and val sets (2006 images)",
        ),
        panoptic_segmentation_ref=dict(
            name="gt_panoptic_trainval_ref.zip",
            directory="gt_panoptic",
            description="Ground-truth annotations for panoptic segmentation for half of train_ref and val_ref sets (1003 normal-condition images)",
        ),
        semantic_segmentation=dict(
            name="gt_trainval.zip",
            directory="gt",
            description="Ground-truth annotations for semantic segmentation for train and val sets (2006 images)",
        ),
        semantic_segmentation_ref=dict(
            name="gt_trainval_ref.zip",
            directory="gt",
            description="Ground-truth annotations for semantic segmentation for half of train_ref and val_ref sets (1003 normal-condition images)",
        ),
        images=dict(
            name="rgb_anon_trainvaltest.zip",
            directory="rgb_anon",
            description="Anonymized images for train, val, and test sets (fog/night/rain/snow + normal ref)",
        )
    )
    rgb_load_path = "rgb_anon"
    target_load_path = "gt_detection"
    dataset_name = "ACDC"
    default_download_key = (
        "images", "detection", "detection_ref",
        "panoptic_segmentation", "panoptic_segmentation_ref",
        "semantic_segmentation", "semantic_segmentation_ref"
    )

    # detection: things only (hasInstances)
    categories = [
        {'id': l.id, 'name': l.name, 'supercategory': l.category}
        for l in cs_labels if l.hasInstances and not l.ignoreInEval
    ]
    classes = [l.name for l in cs_labels if l.hasInstances and not l.ignoreInEval]
    class_ids = [l.id for l in cs_labels if l.hasInstances and not l.ignoreInEval]

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
        self.download(self.root, force=force_download, download_key=self.default_download_key)
        self.train, self.valid = train, valid
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.loader = lambda p: read_image(p, mode=ImageReadMode.RGB)

        cond = corruption_type.value
        is_normal = cond == "normal"
        if train:
            split = ("val_ref" if valid else "train_ref") if is_normal else ("val" if valid else "train")
        else:
            split = "test_ref" if is_normal else "test"

        self.raw_path = self._build_json_path(cond, split, is_normal)
        filter_cond = None if is_normal else cond
        self.samples, self.raw = self.load_data(self.raw_path, filter_cond)

    def _build_json_path(self, cond: str, split: str, is_normal: bool) -> str:
        if is_normal:
            base_split = split.replace("_ref", "")
            if "test" in split:
                json_name = f"instancesonly_{base_split}_ref_image_info.json"
            else:
                json_name = f"instancesonly_{base_split}_gt_ref_detection.json"
            return path.join(self.root, self.target_load_path, json_name)
        if "test" in split:
            json_name = f"instancesonly_{cond}_{split}_image_info.json"
        else:
            json_name = f"instancesonly_{cond}_{split}_gt_detection.json"
        return path.join(self.root, self.target_load_path, cond, json_name)

    def load_data(self, raw_path: str, condition: Optional[str]) -> tuple[list, dict]:
        with open(raw_path, 'r', encoding="utf-8") as f:
            raw = json.load(f)

        ann_by_image = defaultdict(list)
        for ann in raw.get('annotations', []):
            ann_by_image[ann['image_id']].append(ann)

        cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(self.class_ids)}

        samples = []
        for img_info in raw['images']:
            if condition is not None and condition not in img_info['file_name']:
                continue
            img_id = img_info['id']
            img_path = path.join(self.root, self.rgb_load_path, img_info['file_name'])
            h, w = img_info['height'], img_info['width']
            boxes, labels, areas, iscrowd = [], [], [], []
            for ann in ann_by_image[img_id]:
                cat_idx = cat_id_to_idx.get(ann['category_id'])
                if cat_idx is None:
                    continue
                x, y, bw, bh = ann['bbox']
                boxes.append([x, y, x + bw, y + bh])
                labels.append(cat_idx)
                areas.append(ann['area'])
                iscrowd.append(ann['iscrowd'])
            samples.append((img_path, {
                'boxes': boxes, 'labels': labels, 'areas': areas,
                'iscrowd': iscrowd, 'image_id': img_id, 'hw': (h, w)
            }))
        return samples, raw

    def __len__(self) -> int:
        return len(self.samples)

    @classmethod
    def download(
        cls, root: str, force: bool = False, download_key=(
            "images", "detection", "detection_ref",
            "panoptic_segmentation", "panoptic_segmentation_ref",
            "semantic_segmentation", "semantic_segmentation_ref"
        )
    ):
        makedirs(root, exist_ok=True)
        print(f"INFO: Downloading '{cls.dataset_name}' from https://acdc.vision.ee.ethz.ch to {root}...")
        for key in download_key:
            file_name = cls.download_urls[key]['name']
            zip_path = path.join(root, file_name)
            sentinel = path.join(root, f".{file_name}.done")
            already_extracted = path.isfile(sentinel)
            already_downloaded = path.isfile(zip_path)
            if force or not (already_downloaded or already_extracted):
                packages = requests.get("https://acdc.vision.ee.ethz.ch/api/packages").json()['packages']
                package_id = next(p['packageId'] for p in packages if p['name'] == file_name)

                resp = requests.get(cls.base_url + package_id)
                resp.raise_for_status()
                dl_token = resp.json()['token']

                # token is single-use — no HEAD request before GET
                dl_url = f"https://acdc.vision.ee.ethz.ch/api/downloadPackage/{dl_token}/{file_name}"
                with requests.get(dl_url, stream=True) as r:
                    r.raise_for_status()
                    total = int(r.headers.get('content-length', 0)) or None
                    with open(zip_path, 'wb') as f, tqdm(
                        total=total, unit='B', unit_scale=True, unit_divisor=1024,
                        desc=file_name, dynamic_ncols=True, miniters=1, leave=True
                    ) as pbar:
                        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                cls.extract_method(from_path=zip_path, to_path=root)
                open(sentinel, 'w').close()
                print(f"INFO: {file_name} downloaded and extracted.")
            else:
                print(f"INFO: {file_name} already present. Skipping.")
                if not already_extracted and already_downloaded:
                    cls.extract_method(from_path=zip_path, to_path=root)
                    open(sentinel, 'w').close()

    def __getitem__(self, index: int):
        img_path, raw_target = self.samples[index]
        sample = self.loader(img_path)
        h, w = raw_target['hw']
        boxes = raw_target['boxes']
        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            target = {
                'boxes2d': BoundingBoxes(boxes_t, format=BoundingBoxFormat.XYXY, canvas_size=(h, w)),
                'boxes2d_classes': torch.as_tensor(raw_target['labels'], dtype=torch.int64),
                'area': torch.as_tensor(raw_target['areas'], dtype=torch.float32),
                'iscrowd': torch.as_tensor(raw_target['iscrowd'], dtype=torch.int64),
            }
        else:
            target = {
                'boxes2d': BoundingBoxes(torch.zeros((0, 4), dtype=torch.float32), format=BoundingBoxFormat.XYXY, canvas_size=(h, w)),
                'boxes2d_classes': torch.zeros(0, dtype=torch.int64),
                'area': torch.zeros(0, dtype=torch.float32),
                'iscrowd': torch.zeros(0, dtype=torch.int64),
            }
        target['image_id'] = raw_target['image_id']
        target['original_hw'] = (h, w)
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
    default_download_key = ("images", "detection", "detection_ref")


class ACDCDatasetForPanopticSegmentation(ACDCDataset):
    rgb_load_path = "rgb_anon"
    target_load_path = "gt_panoptic"
    default_download_key = ("images", "panoptic_segmentation", "panoptic_segmentation_ref")

    # panoptic: stuff + things
    categories = [
        {'id': l.id, 'name': l.name, 'supercategory': l.category}
        for l in cs_labels if not l.ignoreInEval
    ]
    classes = [l.name for l in cs_labels if not l.ignoreInEval]
    class_ids = [l.id for l in cs_labels if not l.ignoreInEval]

    def _build_json_path(self, cond: str, split: str, is_normal: bool) -> str:
        if is_normal:
            base_split = split.replace("_ref", "")
            if "test" in split:
                json_name = f"{base_split}_ref_image_info.json"
            else:
                json_name = f"{base_split}_gt_ref_panoptic.json"
            return path.join(self.root, self.target_load_path, json_name)
        if "test" in split:
            json_name = f"{cond}_{split}_image_info.json"
        else:
            json_name = f"{cond}_{split}_gt_panoptic.json"
        return path.join(self.root, self.target_load_path, cond, json_name)

    def load_data(self, raw_path: str, condition: Optional[str]) -> tuple[list, dict]:
        with open(raw_path, 'r', encoding="utf-8") as f:
            raw = json.load(f)

        ann_by_image = {ann['image_id']: ann for ann in raw.get('annotations', [])}

        samples = []
        for img_info in raw['images']:
            if condition is not None and condition not in img_info['file_name']:
                continue
            img_id = img_info['id']
            ann = ann_by_image.get(img_id)
            if ann is None:
                continue
            samples.append((path.join(self.root, self.rgb_load_path, img_info['file_name']), {
                'mask_path': path.join(self.root, self.target_load_path, ann['file_name']),
                'segments_info': ann.get('segments_info', []),
                'image_id': img_id,
                'hw': (img_info['height'], img_info['width']),
            }))
        return samples, raw

    def __getitem__(self, index: int):
        img_path, raw_target = self.samples[index]
        sample = self.loader(img_path)
        h, w = raw_target['hw']

        mask_rgb = read_image(raw_target['mask_path'], mode=ImageReadMode.RGB)
        panoptic_mask = mask_rgb[0].long() + mask_rgb[1].long() * 256 + mask_rgb[2].long() * 65536

        target = {
            'panoptic_mask': panoptic_mask,
            'segments_info': raw_target['segments_info'],
            'image_id': raw_target['image_id'],
            'original_hw': (h, w),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            sample, target = self.transforms(sample, target)
        return sample, target


class ACDCDatasetForSemanticSegmentation(ACDCDataset):
    rgb_load_path = "rgb_anon"
    target_load_path = "gt"
    default_download_key = ("images", "semantic_segmentation", "semantic_segmentation_ref")

    # semantic: trainId-based, excludes ignored/void classes
    categories = [
        {'id': l.trainId, 'name': l.name, 'supercategory': l.category}
        for l in cs_labels if not l.ignoreInEval
    ]
    classes = [l.name for l in cs_labels if not l.ignoreInEval]
    class_ids = [l.trainId for l in cs_labels if not l.ignoreInEval]

    def __init__(
        self, root: str, force_download: bool = False,
        train: bool = True, valid: bool = False,
        corruption_type: ACDCDataset.CorruptionType = ACDCDataset.CorruptionType.FOG,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None
    ):
        BaseDataset.__init__(self)
        self.root = path.join(root, self.dataset_name)
        self.download(self.root, force=force_download, download_key=self.default_download_key)
        self.train, self.valid = train, valid
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.loader = lambda p: read_image(p, mode=ImageReadMode.RGB)

        cond = corruption_type.value
        is_normal = cond == "normal"

        if is_normal:
            split = ("val_ref" if valid else "train_ref") if train else "test_ref"
            self.samples = self._collect_ref_samples(split)
        else:
            split = ("val" if valid else "train") if train else "test"
            self.samples = self._collect_adverse_samples(cond, split)

    def _collect_adverse_samples(self, cond: str, split: str) -> list:
        rgb_base = Path(self.root) / self.rgb_load_path / cond / split
        gt_base = Path(self.root) / self.target_load_path / cond / split
        samples = []
        for rgb_path in sorted(rgb_base.glob("**/*_rgb_anon.png")):
            gt_name = rgb_path.name.replace("_rgb_anon.png", "_gt_invIds.png")
            gt_path = gt_base / rgb_path.parent.name / gt_name
            if gt_path.exists():
                samples.append((str(rgb_path), str(gt_path)))
        return samples

    def _collect_ref_samples(self, split: str) -> list:
        samples = []
        for cond in ("fog", "night", "rain", "snow"):
            rgb_base = Path(self.root) / self.rgb_load_path / cond / split
            gt_base = Path(self.root) / self.target_load_path / cond / split
            if not rgb_base.exists():
                continue
            for rgb_path in sorted(rgb_base.glob("**/*_rgb_ref_anon.png")):
                gt_name = rgb_path.name.replace("_rgb_ref_anon.png", "_gt_ref_labelTrainIds.png")
                gt_path = gt_base / rgb_path.parent.name / gt_name
                if gt_path.exists():
                    samples.append((str(rgb_path), str(gt_path)))
        return samples

    def load_data(self, raw_path, condition):
        raise NotImplementedError

    def __getitem__(self, index: int):
        img_path, gt_path = self.samples[index]
        sample = self.loader(img_path)
        target = read_image(gt_path, mode=ImageReadMode.GRAY).squeeze(0).long()
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            sample, target = self.transforms(sample, target)
        return sample, target

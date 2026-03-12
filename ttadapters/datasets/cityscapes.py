"""
CityScapes Dataset Wrapper with Auto-Download
Requires: cityscapesscripts

Register at: https://www.cityscapes-dataset.com/register/

Usage:
    dataset = CityScapesForObjectDetection(
        root="./data",
        train=True,
        force_download=False
    )
"""
from typing import Optional, Callable, List, Tuple
from concurrent.futures import ProcessPoolExecutor
from os import makedirs, path, walk, cpu_count
from collections import defaultdict
from functools import partial
from pathlib import Path
from enum import Enum
import json
import shutil

from tqdm.auto import tqdm

from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import DataLoader
from torchvision.datasets import utils
import torch

from PIL import Image
import numpy as np

from imagecorruptions import corrupt
from cityscapesscripts.helpers.labels import labels as cs_labels
from cityscapesscripts.download import downloader
from cityscapesscripts.download.downloader import login, download_packages

from .base import BaseDataset

downloader.tqdm = tqdm  # Use tqdm from this module
utils.tqdm = tqdm


class CityScapesDataset(BaseDataset):
    dataset_name = "CityScapes"

    # CityScapes instance classes
    valid_labels = [l for l in cs_labels if l.hasInstances and not l.ignoreInEval]
    classes = [l.name for l in cs_labels if l.hasInstances and not l.ignoreInEval]
    train_ids = [l.trainId for l in cs_labels if l.hasInstances and not l.ignoreInEval]

    # Packages to download
    REQUIRED_PACKAGES = [
        "leftImg8bit_trainvaltest.zip",  # Train/val/test images
        "gtFine_trainvaltest.zip",       # Fine annotations
    ]

    IMAGE_PACKAGE_PREFIX = "leftImg8bit"
    ANNOTATION_PACKAGE_PREFIX = "gtFine"
    IMAGE_FILE_SUFFIX = "leftImg8bit"
    ANNOTATION_FILE_SUFFIX = "gtFine_polygons"

    __FIRST_LOAD = False

    def __init__(self, root: str, train: bool = True, valid: bool = False, force_download: bool = False):
        """
        Args:
            root: Root directory for the dataset
            train: If True, use training data; if False, use test data
            valid: If True, use validation data
            force_download: Force re-download even if dataset exists
        """
        self.root = Path(root) / self.dataset_name

        if train:
            self.split = "val" if valid else "train"
        else:
            self.split = "test"

        # Download dataset
        self.download(str(self.root), force=force_download)

        # Set image and annotation paths
        self.images_dir = self.root / self.IMAGE_PACKAGE_PREFIX / self.split
        self.targets_dir = self.root / self.ANNOTATION_PACKAGE_PREFIX / self.split

        # Collect file list
        self.images, self.targets = self._collect_files()

        if not CityScapesDataset.__FIRST_LOAD:
            CityScapesDataset.__FIRST_LOAD = True
            self._print_dataset_info()

    @classmethod
    def download(cls, root: str, force: bool = False, silent: bool = False):
        """Automatic download for CityScapes dataset"""
        root_path = Path(root)

        if force:
            if not silent:
                print(f"INFO: Force download enabled. Removing existing dataset at {root}...")
            shutil.rmtree(root_path, ignore_errors=True)

        makedirs(root_path, exist_ok=True)

        # Check which packages need to be downloaded
        packages_to_download = []
        for package in cls.REQUIRED_PACKAGES:
            package_path = root_path / "_".join([p for p in package.replace(".zip", "").split("_") if "train" not in p])
            download_path = root_path / package
            if not package_path.exists() and not download_path.exists():
                packages_to_download.append(package)
            else:
                if not silent:
                    print(f"INFO: Package '{package}' already downloaded. Skipping.")
                if not package_path.exists() and download_path.exists():
                    # Extract if zip file exists but not extracted
                    try:
                        utils.extract_archive(
                            from_path=str(download_path),
                            to_path=str(root_path),
                            remove_finished=True
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to extract {download_path}: {str(e)}"
                        ) from e

        if not packages_to_download:
            if not silent:
                print("INFO: All required packages already downloaded. Skipping download.")
            return

        if not silent:
            print(f"INFO: Downloading '{cls.dataset_name}' dataset to {root}...")
            print(f"INFO: Packages to download: {packages_to_download}")

        # Login to CityScapes
        session = login()

        # Download each package
        for package in packages_to_download:
            if not silent:
                print(f"INFO: Downloading and extracting {package}...")

            try:
                download_packages(
                    session=session,
                    package_names=[package],
                    destination_path=str(root_path),
                    resume=True
                )
                utils.extract_archive(
                    from_path=str(root_path / package),
                    to_path=str(root_path),
                    remove_finished=True
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {package}: Please check your internet connection or try manual download:\n"
                    "https://www.cityscapes-dataset.com/downloads/"
                ) from e

        if not silent:
            print("INFO: Dataset download and extraction completed.")

    def _collect_files(
        self, images_dir: Path | None = None, targets_dir: Path | None = None,
        image_file_suffix: str | None = None, annotation_file_suffix: str | None = None
    ):
        """Collect image and target file pairs"""
        if images_dir is None:
            images_dir = self.images_dir
        if targets_dir is None:
            targets_dir = self.targets_dir
        if image_file_suffix is None:
            image_file_suffix = self.IMAGE_FILE_SUFFIX
        if annotation_file_suffix is None:
            annotation_file_suffix = self.ANNOTATION_FILE_SUFFIX

        if not images_dir.exists():
            raise RuntimeError(
                f"Dataset not found at {images_dir}. "
                f"Please check if download was successful."
            )

        # Collect files for each city
        images, targets = [], []
        for city_dir in sorted(images_dir.iterdir()):
            if not city_dir.is_dir():
                continue

            for img_path in sorted(city_dir.glob(f"*_{image_file_suffix}*png")):
                images.append(img_path)

                # Generate target file path
                city = city_dir.name
                base_name = img_path.stem.split(f"_{image_file_suffix}")[0]
                target_name = f"{base_name}_{annotation_file_suffix}.json"
                target_path = targets_dir / city / target_name

                if target_path.exists():
                    targets.append(target_path)
                else:
                    # Remove image if target doesn't exist
                    images.pop()

        return images, targets

    def _print_dataset_info(self):
        """Print dataset information"""
        print(f"\n{'='*80}")
        print(f"CityScapes Dataset - {self.split.upper()} split")
        print(f"{'='*80}")
        print(f"Total images: {len(self.images)}")
        print(f"Detection classes: {self.classes}")
        print(f"{'='*80}\n")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        """Abstract method - must be implemented by subclass"""
        raise NotImplementedError("Subclass must implement __getitem__")


class CityScapesDatasetForObjectDetection(CityScapesDataset):
    """CityScapes dataset for Object Detection"""

    def __init__(
        self, root: str, force_download: bool = False, train: bool = True, valid: bool = False, min_area: int = 0,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None,
    ):
        """
        Args:
            root: Root directory for the dataset
            force_download: Force re-download even if dataset exists
            train: If True, use training data; if False, use test data
            valid: If True, use validation data
            min_area: Minimum object area for filtering small objects
            transform: Image transformations
            target_transform: Target transformations
            transforms: Joint image and target transformations
        """
        self.min_area = min_area
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        super().__init__(root=root, train=train, valid=valid, force_download=force_download)

    def _print_dataset_info(self):
        print("Loading first batch for inspection...\n")

        loader = DataLoader(self, batch_size=1, shuffle=False)
        for i, (image, target) in enumerate(loader):
            print(f"Batch {i}:\n")
            print(f"{'Item':20} {'Shape':35} {'Min':10} {'Max':10}")
            print("-" * 80)

            # Image info
            print(f"{'images':20} {str(image.shape):35} {image.min():10.2f} {image.max():10.2f}")

            # Target info
            for key, value in target.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() > 0:
                        print(f"{key:20} {str(value.shape):35} {value.min():10.2f} {value.max():10.2f}")
                    else:
                        print(f"{key:20} {str(value.shape):35} {'N/A':10} {'N/A':10}")
                else:
                    print(f"{key:20} {str(value):35}")
            break
        print()

    def _load_polygon_annotations(self, json_path: Path) -> Tuple[List, List]:
        """Extract bounding boxes from polygon annotations"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        boxes = []
        labels = []

        for obj in data['objects']:
            label_name = obj['label']

            # Check if label is in detection classes
            if label_name not in self.classes:
                continue

            # Calculate bounding box from polygon
            polygon = torch.tensor(obj['polygon'])
            x_coords = polygon[:, 0]
            y_coords = polygon[:, 1]

            x1, y1 = x_coords.min(), y_coords.min()
            x2, y2 = x_coords.max(), y_coords.max()

            # Filter by minimum area
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_area:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.classes.index(label_name))

        return boxes, labels

    def __getitem__(self, idx: int) -> Tuple[tv_tensors.Image, dict]:
        """
        Returns:
            image: tv_tensors.Image (C, H, W)
            target: dict with keys:
                - boxes2d: tv_tensors.BoundingBoxes (N, 4) in XYXY format
                - boxes2d_classes: torch.Tensor (N,)
                - image_id: int
                - area: torch.Tensor (N,)
                - iscrowd: torch.Tensor (N,)
        """
        # Load image
        img_path = self.images[idx]
        image_tv = read_image(img_path, mode=ImageReadMode.RGB)

        # Load annotations
        target_path = self.targets[idx]
        boxes, labels = self._load_polygon_annotations(target_path)

        # Create target dictionary
        target = {}

        if len(boxes) > 0:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            target['boxes2d'] = tv_tensors.BoundingBoxes(
                boxes_tensor,
                format="XYXY",
                canvas_size=image_tv.shape[-2:]
            )
            target['boxes2d_classes'] = torch.as_tensor(labels, dtype=torch.int64)

            # Additional info
            areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * \
                    (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            target['area'] = areas
            target['iscrowd'] = torch.zeros(len(boxes), dtype=torch.int64)
        else:
            # Empty tensors when no objects
            target['boxes2d'] = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format="XYXY",
                canvas_size=image_tv.shape[-2:]
            )
            target['boxes2d_classes'] = torch.zeros(0, dtype=torch.int64)
            target['area'] = torch.zeros(0, dtype=torch.float32)
            target['iscrowd'] = torch.zeros(0, dtype=torch.int64)

        target['image_id'] = idx
        target['original_hw'] = image_tv.shape[-2:]

        # Apply transforms
        if self.transform is not None:
            image_tv = self.transform(image_tv)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            image_tv, target = self.transforms(image_tv, target)

        return image_tv, target


class CityScapesDiscreteDatasetForObjectDetection(CityScapesDatasetForObjectDetection):
    REQUIRED_PACKAGES = [
        "leftImg8bit_trainvaltest.zip", "gtFine_trainvaltest.zip",
        "leftImg8bit_trainval_foggyDBF.zip", "leftImg8bit_trainval_rain.zip"
    ]

    class CorruptionType(Enum):
        BASE = "base"
        FOGGY = "foggy"
        RAIN = "rain"
        SNOW = "snow"
        FROST = "frost"
        BRIGHTNESS = "brightness"

    IMAGE_PACKAGE_PREFIX_MAPPING = {
        CorruptionType.BASE: "leftImg8bit",
        CorruptionType.FOGGY: "leftImg8bit_foggyDBF",
        CorruptionType.RAIN: "leftImg8bit_rain",
        CorruptionType.SNOW: "leftImg8bit_snow",
        CorruptionType.FROST: "leftImg8bit_frost",
        CorruptionType.BRIGHTNESS: "leftImg8bit_brightness"
    }

    def __init__(
        self, root: str, force_download: bool = False,
        train: bool = True, valid: bool = False,
        corruption_type: CorruptionType | None = None, min_area: int = 0,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None,
    ):
        super().__init__(
            root=root, force_download=force_download, train=train, valid=valid, min_area=min_area,
            transform=transform, target_transform=target_transform, transforms=transforms
        )
        self.corruption_type = corruption_type
        domains = self._prepare_domains()
        if self.corruption_type == self.CorruptionType.BASE:
            pass  # Do nothing
        elif self.corruption_type is None:  # all domains
            for dm_images, dm_targets in domains.values():
                dm_images, dm_targets = self._gather_collections(dm_images, dm_targets)
                self.images.extend(dm_images)
                self.targets.extend(dm_targets)
        else:
            self.images, self.targets = self._gather_collections(*domains[self.corruption_type.value])
            print(len(self.images), len(self.targets))

    def _gather_collections(self, images, targets):
        if self.corruption_type in [
            self.CorruptionType.FOGGY, self.CorruptionType.SNOW, self.CorruptionType.RAIN,
            self.CorruptionType.FROST, self.CorruptionType.BRIGHTNESS
        ]:
            prefix = self.IMAGE_PACKAGE_PREFIX_MAPPING[self.corruption_type]

            img_groups = defaultdict(list)
            tgt_groups = defaultdict(list)

            for img, tgt in zip(images, targets):
                stem = Path(img).stem
                suffix = stem.split(prefix)[-1]
                if (  # select middle severity images
                    prefix+suffix == "leftImg8bit_foggyDBF_beta_0.01" or
                    (prefix+suffix).startswith("leftImg8bit_rain_alpha_0.01_beta_0.005_dropsize_0.01") or
                    suffix == "_severity_3"
                ):
                    img_groups[suffix].append(img)
                    tgt_groups[suffix].append(tgt)

            print(img_groups.keys())

            result_images, result_targets = [], []
            for suffix in sorted(img_groups.keys()):
                sorted_imgs = sorted(img_groups[suffix])
                sorted_tgts = sorted(tgt_groups[suffix])
                result_images.extend(sorted_imgs)
                result_targets.extend(sorted_tgts)

            return result_images, result_targets
        else:
            return images, targets

    def _prepare_domains(self) -> dict[str, tuple[list, list]]:
        manual_corruptions = [self.CorruptionType.SNOW, self.CorruptionType.FROST, self.CorruptionType.BRIGHTNESS]
        domains, domains_images_dir = {}, {}
        for tp in self.CorruptionType:
            if tp == self.CorruptionType.BASE:  # base domain is not corrupted
                continue

            prefix_mapping = self.IMAGE_PACKAGE_PREFIX_MAPPING[tp]
            domains_images_dir[tp] = self.root / prefix_mapping / self.split
            dir = Path(domains_images_dir[tp])

            # Rename image file for Foggy
            if tp == self.CorruptionType.FOGGY and not any(dir.rglob("*_foggyDBF_*.png")):
                for img_path in dir.rglob("*_foggy_*.png"):
                    new_name = img_path.stem.replace("_foggy_", "_foggyDBF_")
                    new_path = img_path.parent / f"{new_name}.png"
                    img_path.rename(new_path)

            # Manual Corruption for Snow/Frost
            if tp in manual_corruptions and (not dir.exists() or not any(dir.iterdir())):
                dir.mkdir(parents=True, exist_ok=True)
                tasks = []
                for root, _, files in walk(self.images_dir):
                    for file in files:
                        if file.lower().endswith(".png"):
                            tasks.append((root, file))
                total_files = len(tasks)
                workers = cpu_count() // 2
                print(f"INFO: Starting parallel processing for {total_files} files using {workers} workers...")
                with ProcessPoolExecutor(max_workers=workers, mp_context=__import__("multiprocessing").get_context("spawn")) as executor:
                    func = partial(
                        self.process_single_image, input_root=self.images_dir, output_root=dir,
                        original_file_suffix=self.IMAGE_PACKAGE_PREFIX, new_file_suffix=prefix_mapping, corruption_name=tp.value
                    )
                    list(tqdm(executor.map(func, tasks), total=total_files, desc=f"{tp.value.title()} Corrupting"))
                print(f"\nINFO: {tp.value.title()} corrupting completed.")

            domains[tp.value] = self._collect_files(domains_images_dir[tp], image_file_suffix=prefix_mapping)
        return domains

    @staticmethod
    def process_single_image(
        file_info, input_root, output_root, original_file_suffix, new_file_suffix,
        severities=(1, 3, 5), corruption_name="snow"
    ):
        root, file = file_info
        input_path = path.join(root, file)

        try:
            img = Image.open(input_path).convert("RGB")
            img_array = np.array(img, dtype=np.uint8)

            for sev in severities:
                corrupted_arr = corrupt(img_array, severity=sev, corruption_name=corruption_name)
                new_filename = file.replace(original_file_suffix, f"{new_file_suffix}_severity_{sev}")

                rel_path = path.relpath(root, input_root)
                save_dir = path.join(output_root, rel_path)
                makedirs(save_dir, exist_ok=True)
                Image.fromarray(corrupted_arr).save(path.join(save_dir, new_filename))
            return True
        except Exception as e:
            print(f"\nError processing {file}: {e}")
            return False


class CityScapesCorruptedDatasetForObjectDetection(CityScapesDiscreteDatasetForObjectDetection):
    """CityScapes-C Dataset for Object Detection (Discrete Scenario Only)"""
    pass


class CityScapesContinuousDatasetForObjectDetection(CityScapesDiscreteDatasetForObjectDetection):
    class CorruptionType(Enum):
        BASE_TO_FOGGY = "base_to_foggy"
        BASE_TO_RAIN = "base_to_rain"
        BASE_TO_SNOW = "base_to_snow"
        FOG_TO_RAIN = "fog_to_rain"
        RAIN_TO_SNOW = "rain_to_snow"

    # Suffixes (weak → strong)
    FOG_BETAS  = ["_beta_0.005", "_beta_0.01", "_beta_0.02"]
    SNOW_SEVS  = ["_severity_1", "_severity_3", "_severity_5"]
    RAIN_BETAS = [
        "_alpha_0.01_beta_0.005_dropsize_0.01",
        "_alpha_0.02_beta_0.01_dropsize_0.005",
        "_alpha_0.03_beta_0.015_dropsize_0.002",
    ]
    RAIN_PATTERNS = [f"_pattern_{i}" for i in range(1, 13)]

    # Not used directly; kept for interface compatibility
    IMAGE_PACKAGE_PREFIX_MAPPING = {}

    def __init__(
        self, root: str, force_download: bool = False,
        train: bool = True, valid: bool = False,
        corruption_type: CorruptionType = CorruptionType.BASE_TO_FOGGY, min_area: int = 0,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None,
    ):
        # Skip CityScapesDiscreteDatasetForObjectDetection.__init__
        # → calls CityScapesDatasetForObjectDetection.__init__ (collects base images only)
        super(CityScapesDiscreteDatasetForObjectDetection, self).__init__(
            root=root, force_download=force_download, train=train, valid=valid, min_area=min_area,
            transform=transform, target_transform=target_transform, transforms=transforms
        )
        self.corruption_type = corruption_type
        self.images, self.targets = self._build_continuous_sequence()

    def _collect_domain_by_city(self, prefix: str) -> dict:
        """
        Collect images for a domain grouped by city and scene key.

        Returns:
            {city: {scene_key: {suffix: (img_path, tgt_path)}}}
            where suffix is the part of the stem AFTER '_{prefix}'.
            For the base domain (prefix='leftImg8bit'), suffix is ''.
        """
        images_dir = self.root / prefix / self.split
        if not images_dir.exists():
            return {}

        result: dict = defaultdict(lambda: defaultdict(dict))
        for city_dir in sorted(images_dir.iterdir()):
            if not city_dir.is_dir():
                continue
            city = city_dir.name
            for img_path in sorted(city_dir.glob(f"*_{prefix}*.png")):
                stem = img_path.stem
                parts = stem.split(f"_{prefix}")
                scene_key = parts[0]
                suffix = parts[1] if len(parts) > 1 else ""
                tgt_name = f"{scene_key}_gtFine_polygons.json"
                tgt_path = self.targets_dir / city / tgt_name
                if tgt_path.exists():
                    result[city][scene_key][suffix] = (img_path, tgt_path)
        return result

    def _build_continuous_sequence(self):
        ct = self.corruption_type
        if ct == self.CorruptionType.BASE_TO_FOGGY:
            return self._build_base_to_severity("leftImg8bit_foggyDBF", self.FOG_BETAS)
        elif ct == self.CorruptionType.BASE_TO_SNOW:
            return self._build_base_to_severity("leftImg8bit_snow", self.SNOW_SEVS)
        elif ct == self.CorruptionType.BASE_TO_RAIN:
            return self._build_base_to_rain()
        elif ct == self.CorruptionType.FOG_TO_RAIN:
            return self._build_severity_transition(
                src_prefix="leftImg8bit_foggyDBF", src_suffixes=self.FOG_BETAS,
                dst_prefix="leftImg8bit_rain",    dst_suffixes=self.RAIN_BETAS,
                dst_is_rain=True,
            )
        elif ct == self.CorruptionType.RAIN_TO_SNOW:
            return self._build_severity_transition(
                src_prefix="leftImg8bit_rain",  src_suffixes=self.RAIN_BETAS,
                dst_prefix="leftImg8bit_snow",  dst_suffixes=self.SNOW_SEVS,
                src_is_rain=True,
            )
        return [], []

    def _build_base_to_severity(self, corrupted_prefix: str, ordered_suffixes: list):
        """
        Per-city forward + return sequence.

        With N valid scenes per city and q = N // 4:

          Forward  (first q scenes): base → weak → mid → strong
          Return   (next  q scenes): mid  → weak → base   (unused segments)

        Total = 7 * q images per city.
        """
        base_domain = self._collect_domain_by_city("leftImg8bit")
        corr_domain = self._collect_domain_by_city(corrupted_prefix)
        cities = sorted(set(base_domain) & set(corr_domain))

        result_imgs, result_tgts = [], []
        weak_sfx, mid_sfx, strong_sfx = ordered_suffixes

        for city in cities:
            # Scenes that have ALL required corrupted variants
            common = sorted(set(base_domain[city]) & set(corr_domain[city]))
            valid = [sk for sk in common
                     if all(sfx in corr_domain[city][sk] for sfx in ordered_suffixes)]
            N = len(valid)
            q = N // 4
            if q == 0:
                continue

            fwd = valid[:q]    # used in forward pass
            ret = valid[q:2*q] # used in return pass (different scenes → "unused")

            def _add(domain, city, scenes, sfx):
                for sk in scenes:
                    entry = domain[city][sk].get(sfx)
                    if entry:
                        result_imgs.append(entry[0])
                        result_tgts.append(entry[1])

            # Forward: base → weak → mid → strong
            _add(base_domain, city, fwd, "")
            _add(corr_domain, city, fwd, weak_sfx)
            _add(corr_domain, city, fwd, mid_sfx)
            _add(corr_domain, city, fwd, strong_sfx)
            # Return: mid → weak → base  (unused ret scenes)
            _add(corr_domain, city, ret, mid_sfx)
            _add(corr_domain, city, ret, weak_sfx)
            _add(base_domain, city, ret, "")

        return result_imgs, result_tgts

    def _build_base_to_rain(self):
        """
        Use only cities present in rain domain.
        For each city, n = number of scenes (per weak-beta reference).

        Sequence (per city): base[n] → weak_rain[n] → mid_rain[n] → strong_rain[n]
        Pattern 1-12 cycles within each rain severity segment independently.
        """
        base_domain = self._collect_domain_by_city("leftImg8bit")
        rain_domain = self._collect_domain_by_city("leftImg8bit_rain")
        cities = sorted(set(base_domain) & set(rain_domain))

        result_imgs, result_tgts = [], []
        weak_beta, mid_beta, strong_beta = self.RAIN_BETAS

        for city in cities:
            # Scene keys present in rain (use weak beta as reference)
            rain_scenes = sorted(
                sk for sk in rain_domain[city]
                if any(weak_beta + p in rain_domain[city][sk] for p in self.RAIN_PATTERNS)
            )
            # Intersect with base
            rain_scenes = [sk for sk in rain_scenes if sk in base_domain[city]]
            if not rain_scenes:
                continue

            # base segment
            for sk in rain_scenes:
                entry = base_domain[city][sk].get("")
                if entry:
                    result_imgs.append(entry[0])
                    result_tgts.append(entry[1])

            # rain segments (weak → mid → strong), pattern cycling per segment
            for beta in [weak_beta, mid_beta, strong_beta]:
                for i, sk in enumerate(rain_scenes):
                    pattern = self.RAIN_PATTERNS[i % len(self.RAIN_PATTERNS)]
                    suffix = beta + pattern
                    entry = rain_domain[city][sk].get(suffix)
                    if entry:
                        result_imgs.append(entry[0])
                        result_tgts.append(entry[1])

        return result_imgs, result_tgts

    def _build_severity_transition(
        self,
        src_prefix: str, src_suffixes: list,
        dst_prefix: str, dst_suffixes: list,
        src_is_rain: bool = False,
        dst_is_rain: bool = False,
    ):
        """
        Generic builder for cross-domain transitions.

        Sequence (ascending src, descending dst):
          weak_src(n) → mid_src(3n//5) → strong_src(2n//5)
            → strong_dst(2n//5) → mid_dst(3n//5) → weak_dst(n)

        Reference city list comes from rain domain (src or dst).
        n = scenes per city per one reference beta in rain domain.
        For rain segments, patterns 1-12 cycle independently.
        """
        rain_prefix = src_prefix if src_is_rain else dst_prefix
        rain_domain = self._collect_domain_by_city(rain_prefix)
        src_domain  = self._collect_domain_by_city(src_prefix)
        dst_domain  = self._collect_domain_by_city(dst_prefix)

        # Reference beta for counting rain scenes
        rain_betas = self.RAIN_BETAS
        ref_beta = rain_betas[0]

        cities = sorted(set(rain_domain) & set(src_domain) & set(dst_domain))

        result_imgs, result_tgts = [], []
        weak_src, mid_src, strong_src = src_suffixes
        weak_dst, mid_dst, strong_dst = dst_suffixes

        def _add_plain(domain, city, scenes, sfx):
            for sk in scenes:
                entry = domain[city][sk].get(sfx)
                if entry:
                    result_imgs.append(entry[0])
                    result_tgts.append(entry[1])

        def _add_rain(domain, city, scenes, beta, pat_offset=0):
            for i, sk in enumerate(scenes):
                pattern = self.RAIN_PATTERNS[(i + pat_offset) % len(self.RAIN_PATTERNS)]
                suffix = beta + pattern
                entry = domain[city][sk].get(suffix)
                if entry:
                    result_imgs.append(entry[0])
                    result_tgts.append(entry[1])

        for city in cities:
            # Use rain domain to determine scene list
            if src_is_rain:
                all_scenes = sorted(
                    sk for sk in src_domain[city]
                    if any(ref_beta + p in src_domain[city][sk] for p in self.RAIN_PATTERNS)
                    and sk in dst_domain[city]
                )
            else:
                all_scenes = sorted(
                    sk for sk in dst_domain[city]
                    if any(ref_beta + p in dst_domain[city][sk] for p in self.RAIN_PATTERNS)
                    and sk in src_domain[city]
                )

            n = len(all_scenes)
            split = 3 * n // 5  # mid count
            if n == 0 or split == 0:
                continue

            mid_scenes    = all_scenes[:split]      # first 3/5
            strong_scenes = all_scenes[split:]       # last 2/5

            # ------ ascending (src side) ------
            if src_is_rain:
                _add_rain(src_domain, city, all_scenes,    weak_src)
                _add_rain(src_domain, city, mid_scenes,    mid_src,    pat_offset=n)
                _add_rain(src_domain, city, strong_scenes, strong_src, pat_offset=n + split)
            else:
                _add_plain(src_domain, city, all_scenes,    weak_src)
                _add_plain(src_domain, city, mid_scenes,    mid_src)
                _add_plain(src_domain, city, strong_scenes, strong_src)

            # ------ descending (dst side) ------
            if dst_is_rain:
                _add_rain(dst_domain, city, strong_scenes, strong_dst)
                _add_rain(dst_domain, city, mid_scenes,    mid_dst,    pat_offset=n - split)
                _add_rain(dst_domain, city, all_scenes,    weak_dst,   pat_offset=n)
            else:
                _add_plain(dst_domain, city, strong_scenes, strong_dst)
                _add_plain(dst_domain, city, mid_scenes,    mid_dst)
                _add_plain(dst_domain, city, all_scenes,    weak_dst)

        return result_imgs, result_tgts

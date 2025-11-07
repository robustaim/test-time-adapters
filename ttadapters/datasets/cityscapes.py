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
from os import makedirs
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
        """Automatically download CityScapes dataset"""
        root_path = Path(root)

        if force:
            if not silent:
                print(f"INFO: Force download enabled. Removing existing dataset at {root}...")
            shutil.rmtree(root_path, ignore_errors=True)

        makedirs(root_path, exist_ok=True)

        # Check which packages need to be downloaded
        packages_to_download = []
        for package in cls.REQUIRED_PACKAGES:
            package_path = root_path / package.split("_")[0]
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
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download {package}: Please check your internet connection or try manual download:\n"
                    "https://www.cityscapes-dataset.com/downloads/"
                ) from e

        if not silent:
            print("INFO: Dataset download and extraction completed.")

    def _collect_files(self):
        """Collect image and target file pairs"""
        if not self.images_dir.exists():
            raise RuntimeError(
                f"Dataset not found at {self.images_dir}. "
                f"Please check if download was successful."
            )

        # Collect files for each city
        images, targets = [], []
        for city_dir in sorted(self.images_dir.iterdir()):
            if not city_dir.is_dir():
                continue

            for img_path in sorted(city_dir.glob(f"*_{self.IMAGE_FILE_SUFFIX}.png")):
                images.append(img_path)

                # Generate target file path
                city = city_dir.name
                base_name = img_path.stem.replace(f"_{self.IMAGE_FILE_SUFFIX}", "")
                target_name = f"{base_name}_{self.ANNOTATION_FILE_SUFFIX}.json"
                target_path = self.targets_dir / city / target_name

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
        "leftImg8bit_trainvaltest.zip",  # Train/val/test images
        "gtFine_trainvaltest.zip",       # Fine annotations
    ]

    IMAGE_PACKAGE_PREFIX = "leftImg8bit"
    ANNOTATION_PACKAGE_PREFIX = "gtFine"
    IMAGE_FILE_SUFFIX = "leftImg8bit"
    ANNOTATION_FILE_SUFFIX = "gtFine_polygons"

    class ContinuousSubsetType(Enum):
        DAYTIME_TO_NIGHT = "daytime_to_night"
        CLEAR_TO_FOGGY = "clear_to_foggy"
        CLEAR_TO_RAINY = "clear_to_rainy"

    def __init__(
        self, root: str, force_download: bool = False, train: bool = True, valid: bool = False, min_area: int = 0,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None,
    ):
        super().__init__(
            root=root, force_download=force_download, train=train, valid=valid, min_area=min_area,
            transform=transform, target_transform=target_transform, transforms=transforms
        )


class CityScapesCorruptedDatasetForObjectDetection(CityScapesDiscreteDatasetForObjectDetection):
    pass


class CityScapesContinuousDatasetForObjectDetection(CityScapesDiscreteDatasetForObjectDetection):
    pass

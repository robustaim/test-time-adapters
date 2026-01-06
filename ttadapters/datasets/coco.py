# Original code from: https://github.com/bethgelab/imagecorruptions/blob/master/corrupt_images.py
# Copyright (c) 2019 robustaim
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications:
# - Added COCO and COCO-C dataset support
# Modified portions are licensed under the MIT License (see repository root)
import os
import glob
import argparse
import filetype
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Callable

from tqdm.auto import tqdm

from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode
from torchvision.datasets import utils, CocoDetection
import torch

from imagecorruptions import corrupt
from imagecorruptions import get_corruption_names
from imagecorruptions import corruption_dict
from multiprocessing import Pool
from enum import Enum

from .base import BaseDataset

utils.tqdm = tqdm


class OutputType(Enum):
    """How should the generated files be arranged"""
    SUBDIRS = "subdirs"
    FILENAME = "filename"

    def __str__(self) -> str:
        return self.value


# https://github.com/scikit-image/scikit-image/issues/4294
def corrupt_image(
    image_path: str, image_path_base: str, output_directory: str, output_type: OutputType,
    corruptions: list, severity_levels: list
) -> bool:
    """Apply image corruption to all images in a given folder

    Args:
        image_path (str): Path to an image
        input_path_base (str): Base path of input folder, needed to keep directory structure
        output_directory (str): Output folder
        output_type (OutputType): How should the files be arranged, in
            subfolders for each corruption and severity level or should
            the corruption type be added to the filename
        corruptions (list): which corruptions should be applied
        severity_levels (list): List of severity levels

    Returns:
        bool: If succeeded or failed
    """
    kind = filetype.guess(image_path)
    if not kind.mime.startswith('image'):
        # Skip inputs that are not images...
        return False

    if kind.extension == 'png':
        # matplotlib reads png in float format -> convert to uint8
        img_array = plt.imread(image_path) * 255
        img_array = img_array.astype(dtype=np.uint8)
    else:
        # other image formats are already read as uint8
        img_array = plt.imread(image_path)

    output_path_stub = os.path.relpath(os.path.dirname(image_path), image_path_base)

    for corruption in corruptions:  # get_corruption_names(subset=subset):
        for severity in severity_levels:
            if output_type == OutputType.SUBDIRS:
                # Build output_path with subdirectories for each corruption type
                # and severity, e.g., $OUT_DIR/$ORIGINAL_STRUCTURE/snow/1/image.jpg
                output_path = os.path.join(
                    output_directory, output_path_stub, corruption,
                    str(severity), os.path.basename(image_path)
                )

            elif output_type == OutputType.FILENAME:
                # Put corruption type and severity into filename, e.g., $OUT_DIR/$ORIGINAL_STRUCTURE/image_snow_1.jpg
                fname, ext = os.path.splitext(os.path.basename(image_path))
                fn = "{}_{}_{}{}".format(fname, corruption, str(severity), ext)
                output_path = os.path.join(output_directory, output_path_stub, fn)

            else:
                raise ValueError("output_type unsupported")

            out_dir = os.path.dirname(output_path)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # Apply corruptions
            corrupted = corrupt(img_array, corruption_name=corruption, severity=severity)

            plt.imsave(output_path, corrupted)

    return True


class COCODataset(CocoDetection, BaseDataset):
    download_method = utils.download_and_extract_archive
    extract_method = utils.extract_archive
    base_url = "http://images.cocodataset.org/"
    download_urls = dict(
        images_train=dict(name="train2017.zip", directory="train2017"),
        images_val=dict(name="val2017.zip", directory="val2017"),
        annotations=dict(name="annotations_trainval2017.zip", directory="annotations")
    )

    dataset_name = "COCO"
    classes = [  # COCO 80 classes
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(
        self, root: str, force_download: bool = False,
        train: bool = True, valid: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ):
        """
        Args:
            root: Root directory for the dataset
            force_download: Force re-download even if dataset exists
            train: If True, use training data; if False, use validation data
            valid: If True, use validation data (overrides train parameter)
            transform: Image transformations
            target_transform: Target transformations
            transforms: Joint image and target transformations
        """
        from typing import Callable as _Callable
        self.root = os.path.join(root, self.dataset_name)
        self.train = train
        self.valid = valid

        # Download dataset
        self.download(self.root, force=force_download)

        # Set split
        if valid:
            split = "val2017"
        elif train:
            split = "train2017"
        else:
            split = "val2017"

        # Paths
        img_folder = os.path.join(self.root, split)
        ann_file = os.path.join(
            self.root, "annotations",
            f"instances_{split}.json"
        )

        # Initialize parent class
        super().__init__(root=img_folder, annFile=ann_file)

        # Store transforms
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

    @classmethod
    def download(cls, root: str, force: bool = False, silent: bool = False):
        """Download COCO dataset"""
        if not silent:
            print(f"INFO: Downloading '{cls.dataset_name}' dataset to {root}...")

        os.makedirs(root, exist_ok=True)

        # Download images and annotations
        for key in ["train_images", "val_images", "annotations"]:
            file_name = cls.download_urls[key]["name"]
            download_url = cls.base_url + "zips/" + file_name
            extract_dir = cls.download_urls[key]["directory"]

            downloaded = os.path.isfile(os.path.join(root, file_name))
            extracted = os.path.isdir(os.path.join(root, extract_dir))

            if force:
                if not silent:
                    print(f"INFO: Force download enabled for {file_name}...")
                if downloaded:
                    os.remove(os.path.join(root, file_name))
                downloaded = False
                extracted = False

            if not (downloaded or extracted):
                if not silent:
                    print(f"INFO: Downloading {file_name}...")
                cls.download_method(
                    download_url,
                    download_root=root,
                    extract_root=root,
                    filename=file_name
                )
                if not silent:
                    print(f"INFO: {file_name} downloaded and extracted.")
            else:
                if not silent:
                    print(f"INFO: {file_name} already exists. Skipping download.")
                if not extracted and downloaded:
                    if not silent:
                        print(f"INFO: Extracting {file_name}...")
                    cls.extract_method(
                        from_path=os.path.join(root, file_name),
                        to_path=root
                    )

    def _print_dataset_info(self):
        """Print dataset information"""
        print(f"\n{'='*80}")
        print(f"COCO Dataset - {'TRAIN' if self.train and not self.valid else 'VAL'} split")
        print(f"{'='*80}")
        print(f"Total images: {len(self)}")
        print(f"Detection classes ({len(self.classes)}): {self.classes[:10]}...")
        print(f"{'='*80}\n")

        # Print first batch info
        from torch.utils.data import DataLoader
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

    def __getitem__(self, idx: int):
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
        # Load image and annotations from parent class
        img, target = super().__getitem__(idx)

        # Convert PIL image to tensor
        img_array = np.array(img)
        image_tv = tv_tensors.Image(torch.from_numpy(img_array).permute(2, 0, 1))

        # Process annotations
        boxes = []
        labels = []
        areas = []
        iscrowds = []

        for obj in target:
            # Get bbox in COCO format [x, y, width, height]
            bbox = obj['bbox']
            # Convert to XYXY format
            x1, y1, w, h = bbox
            x2, y2 = x1 + w, y1 + h

            boxes.append([x1, y1, x2, y2])
            labels.append(obj['category_id'])
            areas.append(obj['area'])
            iscrowds.append(obj['iscrowd'])

        # Create target dictionary
        new_target = {}

        if len(boxes) > 0:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            new_target['boxes2d'] = tv_tensors.BoundingBoxes(
                boxes_tensor,
                format="XYXY",
                canvas_size=image_tv.shape[-2:]
            )
            new_target['boxes2d_classes'] = torch.as_tensor(labels, dtype=torch.int64)
            new_target['area'] = torch.as_tensor(areas, dtype=torch.float32)
            new_target['iscrowd'] = torch.as_tensor(iscrowds, dtype=torch.int64)
        else:
            # Empty tensors when no objects
            new_target['boxes2d'] = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format="XYXY",
                canvas_size=image_tv.shape[-2:]
            )
            new_target['boxes2d_classes'] = torch.zeros(0, dtype=torch.int64)
            new_target['area'] = torch.zeros(0, dtype=torch.float32)
            new_target['iscrowd'] = torch.zeros(0, dtype=torch.int64)

        new_target['image_id'] = idx
        new_target['original_hw'] = image_tv.shape[-2:]

        # Apply transforms
        if self.transform is not None:
            image_tv = self.transform(image_tv)
        if self.target_transform is not None:
            new_target = self.target_transform(new_target)
        if self.transforms is not None:
            image_tv, new_target = self.transforms(image_tv, new_target)

        return image_tv, new_target


class COCODatasetForObjectDetection(COCODataset):
    pass


class COCOCorruptedDatasetForObjectDetection(COCODatasetForObjectDetection):
    """COCO dataset with image corruptions applied using imagecorruptions library"""

    class SeverityLevel:
        LV1 = 1
        LV2 = 2
        LV3 = 3
        LV4 = 4
        LV5 = 5

    class CorruptionType(Enum):
        GAU = "Gaussian Noise"
        SHT = "Shot Noise"
        IMP = "Impulse Noise"
        DEF = "Defocus Blur"
        GLS = "Glass Blur"
        MTN = "Motion Blur"
        ZM = "Zoom Blur"
        SNW = "Snow"
        FRS = "Frost"
        FOG = "Fog"
        BRT = "Brightness"
        CNT = "Contrast"
        ELS = "Elastic Transform"
        PX = "Pixelate"
        JPG = "JPEG Compression"



# Example call:
# python corrupt_images.py test_images out_images filename -j 10 -c fog snow -su digital -se 1 2 -n 20
# corrupts all images in test_images and puts the results in out_images
# corruption type will be added to the filename
# corruption happens on 10 cores in parallel
# fog, snow and all digital corruptions are applied
# with severity level 1 and 2
# and on a total of 20 images
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("in_path", help="Directory which has to be processed")
    parser.add_argument("out_path", help="Output folder")
    parser.add_argument("output_type", choices=list(OutputType), type=OutputType,
                        help="How should the output be organized")
    parser.add_argument("-su", "--subset", choices=['common', 'validation', 'all', 'noise', 'blur',
                                                    'weather', 'digital'], help="Which subsets of corruptions should be applied")
    parser.add_argument("-c", "--corruptions", type=str, choices=corruption_dict.keys(), nargs="+",
                        help="Kind of corruptions to be applied, can be mixed with subset")
    parser.add_argument("-se", "--severity", type=int, choices=range(1, 5), nargs="*",
                        help="Severity level of corruption, if not provided all 5 levels will be applied")
    parser.add_argument("-j", type=int, default=1,
                        help="Multiprocessing, default is 1 core")
    parser.add_argument("-n", type=int, help="Limit the number of input images to be corrupted")

    opt = parser.parse_args()

    # make severity a list
    severity_levels = list(range(1, 6)) if opt.severity is None else opt.severity

    # Get the total number of images to be corrupted, mainly for progress bar
    total = opt.n if opt.n is not None else sum([len(files) for r, d, files in os.walk(opt.in_path)])

    corruptions = opt.corruptions if opt.corruptions else []
    if opt.subset:
        corruptions.extend(get_corruption_names(opt.subset))
    corruptions = list(set(corruptions))  # remove duplicates
    assert len(corruptions) > 1, ValueError("No corruption types were provided")

    # Spawn multiprocessing pool
    pool = Pool(opt.j)
    progress_bar = tqdm(total=total, ascii=True)

    def update_bar(*args):
        progress_bar.update()

    i = 0
    for filename in glob.glob(os.path.join(opt.in_path, "**"), recursive=True):
        i += 1
        # skip directories
        if os.path.isdir(filename):
            continue

        pool.apply_async(
            corrupt_image,
            args=[filename, opt.in_path, opt.out_path, opt.output_type, opt.corruptions, severity_levels],
            callback=update_bar
        )

        # break when n is reached
        if opt.n and i > opt.n:
            break

    pool.close()
    pool.join()

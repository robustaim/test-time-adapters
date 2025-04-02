from typing import Callable, Optional
from pathlib import Path
from os import path
import shutil
import sys
import os

from torchvision import datasets, transforms
import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import huggingface_hub
import xmltodict


class ImageNetVIDDataset(datasets.ImageFolder):
    """
    ImageNet-VID dataset for Object Detection and Tracking.
    Only works in Linux.

    :ref: https://huggingface.co/datasets/guanxiongsun/imagenetvid
    """

    download_method = huggingface_hub.snapshot_download
    dataset_name = "ILSVRC2015_VID"
    dataset_id = "guanxiongsun/imagenetvid"
    obj_classes = [
        "n02691156", "n02419796", "n02131653", "n02834778", "n01503061",
        "n02924116", "n02958343", "n02402425", "n02084071", "n02121808",
        "n02503517", "n02118333", "n02510455", "n02342885", "n02374451",
        "n02129165", "n01674464", "n02484322", "n03790512", "n02324045",
        "n02509815", "n02411705", "n01726692", "n02355227", "n02129604",
        "n04468005", "n01662784", "n04530566", "n02062744", "n02391049"
    ]
    obj_class_namees = [
        "airplane", "antelope", "bear", "bicycle", "bird", "bus", "car", "cattle",
        "dog", "domestic cat", "elephant", "fox", "giant panda", "hamster", "horse",
        "lion", "lizard", "monkey", "motorcycle", "rabbit", "red panda",
        "sheep", "snake", "squirrel", "tiger", "train", "turtle",
        "watercraft", "whale", "zebra"
    ]

    def __init__(
            self,
            root: str,
            force_download: bool = True,
            train: bool = True,
            valid: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ):
        self.root = path.join(root, self.dataset_name)
        self.download(self.root, force=force_download)
        self.default_target_transform = transforms.Lambda(lambda x: self.query_annotation(x))
        target_transform = transforms.Compose([
            self.default_target_transform, target_transform
        ]) if target_transform is not None else self.default_target_transform

        if train:
            self.root = path.join(self.root, "val") if valid else path.join(self.root, "train")
        else:
            self.root = path.join(self.root, "test")

        super().__init__(root=self.root, transform=transform, target_transform=target_transform)
        self.cached_annotations = [None] * len(self.samples)
        self.samples = [(data[0], (data[1], idx)) for idx, data in enumerate(self.samples)]

    def query_annotation(self, img_info: int):
        img_index = img_info[1]
        cache = self.cached_annotations[img_index]
        if cache is None:
            file_path = self.samples[img_index][0].replace(".jpeg", ".xml").replace(".JPEG", ".xml")

            with open(file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
                cache = self.cached_annotations[img_index] = xmltodict.parse(xml_content)['annotation']

            try:
                objects = cache['object'] if isinstance(cache['object'], list) else [cache['object']]  # Make sure it is a list
                del cache['object']
                cache['labels'] = [self.obj_classes.index(obj['name']) for obj in objects]
                cache['boxes'] = [
                    [float(obj['bndbox'][key]) for key in ['xmin', 'ymin', 'xmax', 'ymax']] for obj in objects
                ]
            except KeyError:
                cache['labels'], cache['boxes'] = [], []
        return cache

    @staticmethod
    def label_transform(img_info: dict, normalize: bool = False):
        bboxes_info, labels_info = img_info['boxes'], img_info['labels']
        if normalize:
            height, width = float(img_info['size']['height']), float(img_info['size']['width'])
            bboxes_info = [[
                               bbox_info[0] / width, bbox_info[1] / height,  # xmin, ymin
                               bbox_info[2] / width, bbox_info[3] / height  # xmax, ymax
                           ] if bbox_info else bbox_info for bbox_info in bboxes_info]
        return dict(boxes=torch.tensor(bboxes_info), labels=torch.tensor(labels_info))

    @classmethod
    def download(cls, root: str, force: bool = False):
        root = Path(root)

        # Clean up the existing dataset if force is flagged
        if force:
            print(f"INFO: Cleaning up the existing dataset at {root} (Force-download is flagged)")
            for item in os.listdir(root):
                item_path = root / item
                if path.isfile(item_path):
                    os.remove(item_path)
                else:
                    shutil.rmtree(item_path)
            print("INFO: Dataset cleaned successfully.")

        # Do download if the dataset does not exist
        print(f"INFO: Downloading '{cls.dataset_id} from huggingface to {root}...")
        dnlod = lambda: cls.download_method(
            repo_id=cls.dataset_id,
            repo_type="dataset",
            local_dir=root,
            ignore_patterns=["*.git*", "*.md", "*ILSVRC2017*", "annotations.tar.gz"],
        )
        if force or not (
                path.exists(root) and any(p for p in Path(root).iterdir() if not p.name.startswith('.'))
        ):  # Check if dataset files already exist in the directory
            dnlod()
            print("INFO: Dataset downloaded successfully.")
        else:
            #dnlod()  # make sure the dataset is up-to-date
            print("INFO: Dataset files found in the root directory. Skipping download.")

        # Combine split archive
        dataset_archive = root / f"{cls.dataset_name}.tar.gz"
        if not path.exists(dataset_archive):
            print("INFO: Combining seperated archives...")
            result = os.system(f"cat {dataset_archive}.a* | dd status=progress of={dataset_archive}")
            #result = os.system(f"cat {dataset_archive}.a* | pv -s $(du -bc {dataset_archive}.a* | tail -1 | cut -f1) > {dataset_archive}")
            if result != 0:
                raise Exception("Failed to combine split archives. Please make sure that you are running on a Linux system.")
            print("INFO: Split archives combined successfully.")
        else:
            print("INFO: Combined archives found in the root directory. Skipping combination.")

        # Extract the dataset
        if path.isdir(root / "train") and any(p for p in Path(root / "train").iterdir() if not p.name.startswith('.')) \
                and path.isdir(root / "val") and any(p for p in Path(root / "val").iterdir() if not p.name.startswith('.')):
            print("INFO: Dataset is already extracted")
        else:
            print("INFO: Extracting the dataset...", flush=True)
            if os.system(f"dd if={dataset_archive} bs=4M status=progress | tar -I pigz -x -C {root}"):
                #os.system(f"pv {dataset_archive} | tar -I pigz -xz -C {root}")
                print("\nERROR: Cannot find pigz in the system, using default tar command instead", file=sys.stderr, flush=True)
                if os.system(f"dd if={dataset_archive} bs=4M status=progress | tar -xz -C {root}"):
                    #os.system(f"pv {dataset_archive} | tar -xz -C {root}")
                    raise Exception(f"Failed to extract {dataset_archive}")
            # ----
            # Move files to the correct directories
            temp_dir = root / cls.dataset_name.replace("_VID", "")
            # ---- metadata
            #os.system(f"mv {temp_dir}/ImageSets/VID/* {root}")
            # ---- datas
            for subdir in os.listdir(f"{temp_dir}/Data/VID/train"):  # flatten the train data directory
                annt = temp_dir / "Annotations" / "VID" / "train"
                dt = temp_dir / "Data" / "VID" / "train"
                os.system(f"mv {annt}/{subdir}/* {annt}/")
                os.system(f"mv {dt}/{subdir}/* {dt}/")
                os.system(f"rmdir {annt}/{subdir}")
                os.system(f"rmdir {dt}/{subdir}")
            os.system(f"mv {temp_dir}/Data/VID/* {root}")  # copy images
            for data_type in ["train", "val"]:
                for subdir in os.listdir(f"{root}/{data_type}"):  # copy lables
                    os.system(f"mv {temp_dir}/Annotations/VID/{data_type}/{subdir}/* {root}/{data_type}/{subdir}")
            os.system(f"rm -r {temp_dir}")

            print("INFO: Dataset is extracted successfully")

    @property
    def df(self) -> pd.DataFrame:
        return pd.DataFrame(dict(path=[d[0] for d in self.samples], label=[self.classes[lb] for lb in self.targets]))

    def output_sampling(self, img_norm: dict, idx: int | None = None, figsize=(7, 5), imgsize=(224, 224)):
        # Get random index if not provided
        if idx is None:
            idx = np.random.randint(len(self))
            if idx == 0:
                idx = 1

        # Get frame pair
        (prev_img, prev_gt), (curr_img, curr_gt) = self[idx-1], self[idx]

        # Convert tensors to numpy arrays and denormalize
        def denormalize(img_tensor):
            # Move channels to last dimension
            img = img_tensor.permute(1, 2, 0).numpy()
            # Denormalize
            img = img * np.array(img_norm['std']) + np.array(img_norm['mean'])
            # Clip values to valid range
            img = np.clip(img, 0, 1)
            return img

        prev_img = denormalize(prev_img)
        curr_img = denormalize(curr_img)

        def draw_bbox(ax, bbox, color='red'):
            """Helper function to draw bounding box"""
            xmin, ymin, xmax, ymax = bbox.numpy()
            xmin *= imgsize[0]  # xmin
            ymin *= imgsize[1]  # ymin
            xmax *= imgsize[0]  # xmax
            ymax *= imgsize[1]  # ymax
            ax.plot([xmin, xmax], [ymin, ymin], color=color, linewidth=2)
            ax.plot([xmin, xmin], [ymin, ymax], color=color, linewidth=2)
            ax.plot([xmax, xmax], [ymin, ymax], color=color, linewidth=2)
            ax.plot([xmin, xmax], [ymax, ymax], color=color, linewidth=2)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot previous frame
        ax1.imshow(prev_img)
        for bbox in prev_gt['boxes']:
            draw_bbox(ax1, bbox)
        ax1.set_title('Previous Frame')
        ax1.axis('off')

        # Plot current frame
        ax2.imshow(curr_img)
        for bbox in curr_gt['boxes']:
            draw_bbox(ax2, bbox)
        ax2.set_title('Current Frame')
        ax2.axis('off')

        plt.tight_layout()
        plt.show()

        return idx

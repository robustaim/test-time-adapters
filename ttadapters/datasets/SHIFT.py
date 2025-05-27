"""
SHIFT Dataset DevKit
python download.py --view "[front]" --group "[img, det_2d]" --split "all" --framerate "[images]" --shift "discrete" ./data/SHIFT
"""

from shift_dev import SHIFTDataset
from shift_dev.types import Keys, DataDict
from shift_dev.utils.backend import ZipBackend

from torch.utils.data import DataLoader
import torch

from typing import Optional, Callable
from os import path, system, makedirs
from sys import executable
from enum import Enum
import shutil


class SHIFTDataset(SHIFTDataset):
    dataset_name = "SHIFT"
    categories = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]

    class Type(Enum):
        DISCRETE = "discrete"
        CONTINUOUS_1X = "continuous/1x"
        CONTINUOUS_10X = "continuous/10x"
        CONTINUOUS_100X = "continuous/100x"

    class View(Enum):
        FRONT = "front"
        CENTER = "center"
        LEFT_45 = "left_45"
        LEFT_90 = "left_90"
        RIGHT_45 = "right_45"
        RIGHT_90 = "right_90"
        LEFT_STEREO = "left_stereo"
        ALL = "all"

    class FrameRate(Enum):
        IMAGES = "images"
        VIDEO = "video"
        ALL = "all"

    keys_to_load = [
        Keys.images,                # note: images, shape (1, 3, H, W), uint8 (RGB)
        Keys.intrinsics,            # note: camera intrinsics, shape (3, 3)
        Keys.boxes2d,               # note: 2D boxes in image coordinate, (x1, y1, x2, y2)
        Keys.boxes2d_classes,       # note: class indices, shape (num_boxes,)
        Keys.boxes2d_track_ids,     # note: object ids, shape (num_ins,)
        Keys.boxes3d,               # note: 3D boxes in camera coordinate, (x, y, z, dim_x, dim_y, dim_z, rot_x, rot_y, rot_z)
        Keys.boxes3d_classes,       # note: class indices, shape (num_boxes,), the same as 'boxes2d_classes'
        Keys.boxes3d_track_ids,     # note: object ids, shape (num_ins,), the same as 'boxes2d_track_ids'
        Keys.segmentation_masks,    # note: semantic masks, shape (1, H, W), long
        Keys.masks,                 # note: instance masks, shape (num_ins, H, W), binary
        Keys.depth_maps,            # note: depth maps, shape (1, H, W), float (meters)
    ]
    views_to_load = [View.FRONT]
    framerate = FrameRate.IMAGES
    shift_type = Type.DISCRETE      # also supports "continuous/1x", "continuous/10x", "continuous/100x"
    backend = ZipBackend()          # also supports HDF5Backend(), FileBackend()

    def __init__(
            self, root: str, force_download: bool = False,
            train: bool = True, valid: bool = False,
            transform: Optional[Callable] = None, target_transform: Optional[Callable] = None
    ):
        self.root = path.join(root, self.dataset_name)
        self.download(path.join(self.root, self.shift_type.value), force=force_download)
        self.transform = transform
        self.target_transform = target_transform
        views = [v.value for v in self.views_to_load]

        if train:
            self.split = "val" if valid else "train"
        else:
            self.split = "test"  # TODO: Test set is not working yet.

        super().__init__(
            data_root=self.root,
            split=self.split,
            keys_to_load=self.keys_to_load,
            views_to_load=views,
            framerate=self.framerate.value,
            shift_type=self.shift_type.value,
            backend=self.backend,
            verbose=True
        )

        # Print the tensor shape of the first batch.
        for i, batch in enumerate(DataLoader(self, shuffle=False)):
            print(f"Batch {i}:\n")
            print(f"{'Item':20} {'Shape':35} {'Min':10} {'Max':10}")
            print("-" * 80)
            for k, data in batch[views[0]].items():
                if isinstance(data, torch.Tensor):
                    print(f"{k:20} {str(data.shape):35} {data.min():10.2f} {data.max():10.2f}")
                else:
                    print(f"{k:20} {data}")
            break
        print()

        # Print the sample indices within a video.
        # The video indices groups frames based on their video sequences. They are useful for training on videos.
        video_to_indices = self.video_to_indices
        for video, indices in video_to_indices.items():
            print(f"Video name: {video}")
            print(f"Sample indices within a video: {indices}")
            break

    @classmethod
    def download(cls, root: str, force: bool = False):
        print(f"INFO: Downloading '{cls.dataset_name}' from file server to {root}...")
        frame_dir = cls.framerate.value if cls.framerate != cls.FrameRate.ALL else "images"
        if "continuous\\" in root:
            root = root.split("continuous\\")
            root = path.join(root[0], "continuous", root[1], frame_dir)
            train_dir = path.join(root, "train")
        else:
            train_dir = path.join(root, frame_dir, "train")
        if force:  # If force is True, remove the existing dataset directory.
            shutil.rmtree(root, ignore_errors=True)
        check_dirs = [path.isdir(path.join(train_dir, view.value)) for view in cls.views_to_load]
        if False in check_dirs:
            raise RuntimeError()
            makedirs(root, exist_ok=True)
            frame_param = f"[{cls.framerate.value}]" if cls.framerate != cls.FrameRate.ALL else "all"
            view_param = ", ".join(f'"{view.value}"' for view in cls.views_to_load)
            view_param = "all" if "all" in view_param else f"[{view_param}]"
            type_param = cls.shift_type.value
            if system(f"{executable} -m shift_dev.download --view \"{view_param}\" --group \"all\" --split \"all\" --framerate \"{frame_param}\" --shift \"{type_param}\" {root}") != 0:
                raise RuntimeError("Failed to download the SHIFT dataset. Please check your internet connection and try again.")
            print("INFO: Dataset archive downloaded and extracted.")
        else:
            print("INFO: Dataset archive found in the root directory. Skipping download.")

    def __getitem__(self, idx: int) -> DataDict:
        queried = super().__getitem__(idx)
        for key in queried:
            data = queried[key]
            images = data.pop('images', None)
            if self.transform is not None and images is not None:
                images = self.transform(images)
            if self.target_transform is not None:
                data = self.target_transform(data)
            if images is not None:
                data['images'] = images
        return queried


class SHIFTDiscreteDatasetForObjectDetection(SHIFTDataset):
    keys_to_load = [
        Keys.images,
        Keys.intrinsics,
        Keys.boxes2d,
        Keys.boxes2d_classes,
        Keys.boxes2d_track_ids,
    ]
    views_to_load = [SHIFTDataset.View.FRONT]
    framerate = SHIFTDataset.FrameRate.IMAGES
    shift_type = SHIFTDataset.Type.DISCRETE


class SHIFTContinuousDatasetForObjectDetection(SHIFTDataset):
    keys_to_load = [
        Keys.images,
        Keys.intrinsics,
        Keys.boxes2d,
        Keys.boxes2d_classes,
        Keys.boxes2d_track_ids,
    ]
    views_to_load = [SHIFTDataset.View.FRONT]
    framerate = SHIFTDataset.FrameRate.IMAGES
    shift_type = SHIFTDataset.Type.CONTINUOUS_1X


class SHIFTContinuous10DatasetForObjectDetection(SHIFTDataset):
    keys_to_load = [
        Keys.images,
        Keys.intrinsics,
        Keys.boxes2d,
        Keys.boxes2d_classes,
        Keys.boxes2d_track_ids,
    ]
    views_to_load = [SHIFTDataset.View.FRONT]
    framerate = SHIFTDataset.FrameRate.IMAGES
    shift_type = SHIFTDataset.Type.CONTINUOUS_10X


class SHIFTContinuous100DatasetForObjectDetection(SHIFTDataset):
    keys_to_load = [
        Keys.images,
        Keys.intrinsics,
        Keys.boxes2d,
        Keys.boxes2d_classes,
        Keys.boxes2d_track_ids,
    ]
    views_to_load = [SHIFTDataset.View.FRONT]
    framerate = SHIFTDataset.FrameRate.IMAGES
    shift_type = SHIFTDataset.Type.CONTINUOUS_100X


class SHIFTDiscreteSubsetForObjectDetection(SHIFTDataset):
    keys_to_load = [
        Keys.images,
        Keys.intrinsics,
        Keys.boxes2d,
        Keys.boxes2d_classes,
        Keys.boxes2d_track_ids,
    ]
    views_to_load = [SHIFTDataset.View.FRONT]
    framerate = SHIFTDataset.FrameRate.IMAGES
    shift_type = SHIFTDataset.Type.DISCRETE

    def __init__(
            self, root: str, force_download: bool = False,
            train: bool = True, valid: bool = False,
            transform: Optional[Callable] = None, target_transform: Optional[Callable] = None
    ):
        super().__init__(root, force_download, train, valid, transform, target_transform)

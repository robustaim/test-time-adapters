"""
SHIFT Dataset DevKit
python download.py --view "[front]" --group "[img, det_2d]" --split "all" --framerate "[images]" --shift "discrete" ./data/SHIFT
"""
from shift_dev import SHIFTDataset as _SHIFTDataset
from shift_dev.dataloader.shift_dataset import _SHIFTScalabelLabels, Scalabel, DataBackend, HDF5Backend, _get_extension
from shift_dev.types import Keys, DataDict
from shift_dev.utils.backend import ZipBackend

from torchvision import tv_tensors
from torch.utils.data import DataLoader
import torch

from typing import Optional, Callable
from os import path, system, makedirs
from json import load, dump
from sys import executable
from enum import Enum
import shutil

from .base import BaseDataset


def create_instant_labelclass(annotation_root_suffix: str = "_SUBSET", subset_name: str = "normal"):
    class SHIFTScalabelLabelsForSubset(_SHIFTScalabelLabels):
        def __init__(
            self,
            data_root: str,
            split: str,
            data_file: str = "",
            annotation_file: str = "",
            view: str = "front",
            framerate: str = "images",
            shift_type: str = "discrete",
            backend: DataBackend = HDF5Backend(),
            verbose: bool = False,
            num_workers: int = 1,
            **kwargs,
        ) -> None:
            self.verbose = verbose
            self.num_workers = num_workers

            # Validate input
            assert split in set(("train", "val", "test")), f"Invalid split '{split}'"
            assert view in _SHIFTScalabelLabels.VIEWS, f"Invalid view '{view}'"

            # Set attributes
            ext = _get_extension(backend)
            if shift_type.startswith("continuous"):
                shift_speed = shift_type.split("/")[-1]
                annotation_path = path.join(
                    data_root+annotation_root_suffix, subset_name, "continuous", framerate, shift_speed, split, view, annotation_file
                )
                data_path = path.join(
                    data_root, "continuous", framerate, shift_speed, split, view, f"{data_file}{ext}"
                )
            else:
                annotation_path = path.join(
                    data_root+annotation_root_suffix, subset_name, "discrete", framerate, split, view, annotation_file
                )
                data_path = path.join(
                    data_root, "discrete", framerate, split, view, f"{data_file}{ext}"
                )
            super(_SHIFTScalabelLabels, self).__init__(data_path, annotation_path, data_backend=backend, **kwargs)
    return SHIFTScalabelLabelsForSubset


class SHIFTDataset(_SHIFTDataset, BaseDataset):
    dataset_name = "SHIFT"
    classes = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]

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
        train: bool = True, valid: bool = False
    ):
        self.root = path.join(root, self.dataset_name)
        self.download(path.join(self.root, self.shift_type.value), force=force_download)
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
    def download(cls, root: str, force: bool = False, silent: bool = False):
        if not silent: print(f"INFO: Downloading '{cls.dataset_name}' from file server to {root}...")
        frame_dir = cls.framerate.value if cls.framerate != cls.FrameRate.ALL else "images"
        if "continuous\\" in root or "continuous/" in root:  # continuous
            root = root.split("continuous\\") if "\\" in root else root.split("continuous/")
            data_root = root[0]
            root = path.join(root[0], "continuous", frame_dir, root[1])
            train_dir = path.join(root, "train")
        else:  # discrete
            if path.basename(path.normpath(root)) == "discrete":
                train_dir = path.join(root, frame_dir, "train")
                data_root = path.dirname(root)
            else:  # not specified => assume that the root path is the dataset directory
                train_dir = path.join(root, "discrete", frame_dir, "train")
                data_root = root
        if force:  # If force is True, remove the existing dataset directory.
            shutil.rmtree(root, ignore_errors=True)
        check_dirs = [path.isdir(path.join(train_dir, view.value)) for view in cls.views_to_load]
        if False in check_dirs:
            makedirs(root, exist_ok=True)
            frame_param = f"[{cls.framerate.value}]" if cls.framerate != cls.FrameRate.ALL else "all"
            view_param = ", ".join(f'"{view.value}"' for view in cls.views_to_load)
            view_param = "all" if "all" in view_param else f"[{view_param}]"
            type_param = cls.shift_type.value
            if system(f"{executable} -m shift_dev.download --view \"{view_param}\" --group \"all\" --split \"all\" --framerate \"{frame_param}\" --shift \"{type_param}\" {data_root}") != 0:
                raise RuntimeError("Failed to download the SHIFT dataset. Please check your internet connection and try again.")
            if not silent: print("INFO: Dataset archive downloaded and extracted.")
        else:
            if not silent: print("INFO: Dataset archive found in the root directory. Skipping download.")


class SHIFTDiscreteDatasetForObjectDetection(SHIFTDataset):
    keys_to_load = [
        Keys.images,
        Keys.intrinsics,
        Keys.boxes2d,
        Keys.boxes2d_classes,
        Keys.boxes2d_track_ids,
    ]
    views_to_load = (SHIFTDataset.View.FRONT,)
    framerate = SHIFTDataset.FrameRate.IMAGES
    shift_type = SHIFTDataset.Type.DISCRETE

    def __init__(
        self, root: str, force_download: bool = False, train: bool = True, valid: bool = False,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None
    ):
        super().__init__(root=root, force_download=force_download, train=train, valid=valid)
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms
        self.view_key = "front"

    def __getitem__(self, idx: int) -> DataDict:
        data = super().__getitem__(idx)[self.view_key].copy()
        image, boxes2d = data.pop('images', None), data.pop('boxes2d', None)
        if image is None or boxes2d is None:
            raise ValueError(f"Images or Bounding boxes not found in the dataset with camera view: {self.view_key}")

        image_tv = tv_tensors.Image(image)
        boxes2d_tv = tv_tensors.BoundingBoxes(
            boxes2d,
            format="XYXY",  # SHIFT uses Pascal VOC format (x1, y1, x2, y2)
            canvas_size=image_tv.shape[-2:]  # H, W
        )
        data['boxes'] = boxes2d_tv

        if self.transform is not None:
            image_tv = self.transform(image_tv)
        if self.target_transform is not None:
            data = self.target_transform(data)
        if self.transforms is not None:
            image_tv, data = self.transforms(image_tv, data)

        return image_tv, data


class SHIFTDiscreteSubsetForObjectDetection(SHIFTDiscreteDatasetForObjectDetection):
    dataset_name = "SHIFT_SUBSET"

    class SubsetType(Enum):
        NORMAL = "normal"
        CORRUPTED = "corrupted"

        CLEAR_DAYTIME = "clear_daytime"
        CLEAR_NIGHT = "clear_night"
        CLEAR_DAWN = "clear_dawn"

        CLOUDY_DAYTIME = "cloudy_daytime"
        CLOUDY_NIGHT = "cloudy_night"
        CLOUDY_DAWM = "cloudy_dawn"

        FOGGY_DAYTIME = "foggy_daytime"
        FOGGY_NIGHT = "foggy_night"
        FOGGY_DAWN = "foggy_dawn"

        OVERCAST_DAYTIME = "overcast_daytime"
        OVERCAST_NIGHT = "overcast_night"
        OVERCAST_DAWN = "overcast_dawn"

        RAINY_DAYTIME = "rainy_daytime"
        RAINY_NIGHT = "rainy_night"
        RAINY_DAWN = "rainy_dawn"

    def __init__(
        self, root: str, force_download: bool = False,
        train: bool = True, valid: bool = False, subset_type: SubsetType = SubsetType.NORMAL,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None
    ):
        # Let the original constructor operate correctly
        new_root = path.join(root, self.dataset_name)
        self.dataset_name = SHIFTDataset.dataset_name  # override the dataset name to use the original one
        self.root = path.join(root, self.dataset_name, self.shift_type.value)

        # Ensure the dataset is downloaded and split correctly
        super().download(self.root, force=force_download)
        self.subset_split(data_root=new_root, origin=self.root, force=force_download)

        # Create an instant _SHIFTScalabelLabels class to create a new one for the subset
        # WARNING: This does not work with multi-threading.
        from shift_dev.dataloader import shift_dataset
        shift_dataset._SHIFTScalabelLabels = create_instant_labelclass(annotation_root_suffix="_SUBSET", subset_name=subset_type.value)
        super().__init__(
            root=root, force_download=force_download,
            train=train, valid=valid,
            transform=transform, target_transform=target_transform, transforms=transforms
        )

        # Set the root directory based on the subset type
        del self.dataset_name  # recover the dataset name
        shift_dataset._SHIFTScalabelLabels = _SHIFTScalabelLabels  # Restore the original class
        self.root = path.join(new_root, subset_type.value)

    @classmethod
    def subset_split(cls, data_root: str, origin: str, force: bool = False):
        if cls.shift_type != SHIFTDataset.Type.DISCRETE:
            raise ValueError("Subset split is only available for the discrete version of the SHIFT dataset.")

        if path.basename(path.normpath(origin)) != cls.shift_type.value:
            origin = path.join(origin, cls.shift_type.value)

        if force or not path.isdir(data_root):
            print(f"INFO: Subset split for '{cls.dataset_name}' dataset is started...")
            for sett in ['train', 'val']:
                print("INFO: Splitting", sett)
                data = load(open(path.join(origin, "images", sett, "front", "det_2d.json")))

                # Simple separation
                normal = dict(config=data['config'], frames=[d for d in data['frames'] if d['attributes']['weather_coarse'] == "clear" and d['attributes']['timeofday_coarse'] == "daytime"])
                corrupted = dict(config=data['config'], frames=[d for d in data['frames'] if d['attributes']['weather_coarse'] != "clear" or d['attributes']['timeofday_coarse'] != "daytime"])
                print(f"INFO: <simple> weather datasets - Normal: {len(normal['frames'])}, Corrupted: {len(corrupted['frames'])}")

                for weather in ["normal", "corrupted"]:
                    save_path = path.join(data_root, weather, cls.shift_type.value, "images", sett, "front")
                    makedirs(save_path)
                    dump(locals()[weather], open(path.join(save_path, "det_2d.json"), "w"))

                # Detailed separation
                for weather in ["clear", "rainy", "cloudy", "foggy", "overcast"]:
                    daytime = dict(config=data['config'], frames=[d for d in data['frames'] if d['attributes']['weather_coarse'] == weather and d['attributes']['timeofday_coarse'] == "daytime"])
                    night = dict(config=data['config'], frames=[d for d in data['frames'] if d['attributes']['weather_coarse'] == weather and d['attributes']['timeofday_coarse'] == "night"])
                    dawn = dict(config=data['config'], frames=[d for d in data['frames'] if d['attributes']['weather_coarse'] == weather and d['attributes']['timeofday_coarse'] == "dawn/dusk"])
                    print(f"INFO: <{weather}> weather datasets - Daytime: {len(daytime['frames'])}, Night: {len(night['frames'])}, Dawn: {len(dawn['frames'])}")

                    for time in ["daytime", "night", "dawn"]:
                        _id = f"{weather}_{time}"
                        save_path = path.join(data_root, _id, cls.shift_type.value, "images", sett, "front")
                        makedirs(save_path)
                        dump(locals()[time], open(path.join(save_path, "det_2d.json"), "w"))
        else:
            print(f"INFO: Subset split for '{cls.dataset_name}' dataset is already done. Skipping...")


class SHIFTClearDatasetForObjectDetection(SHIFTDiscreteSubsetForObjectDetection):
    def __init__(
        self, root: str, force_download: bool = False,
        train: bool = True, valid: bool = False,
        transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None
    ):
        super().__init__(
            root=root, force_download=force_download,
            train=train, valid=valid, subset_type=self.SubsetType.NORMAL,
            transform=transform, target_transform=target_transform
        )


class SHIFTCorruptedDatasetForObjectDetection(SHIFTDiscreteSubsetForObjectDetection):
    def __init__(
            self, root: str, force_download: bool = False,
            train: bool = True, valid: bool = False,
            transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, transforms: Optional[Callable] = None
    ):
        super().__init__(
            root=root, force_download=force_download,
            train=train, valid=valid, subset_type=self.SubsetType.CORRUPTED,
            transform=transform, target_transform=target_transform
        )


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

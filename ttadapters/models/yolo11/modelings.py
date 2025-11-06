from dataclasses import dataclass
from typing import Optional
from copy import copy

import torch
from torchvision.tv_tensors import BoundingBoxFormat, BoundingBoxes
from torchvision.transforms.v2.functional import convert_bounding_box_format

import numpy as np

from ..base import BaseModel, ModelProvider, WeightsInfo
from ...datasets import BaseDataset, DataPreparation

from .wrappers import (
    get_cfg, nms, ops, LOGGER,
    build_dataloader, Instances, Compose, v8_transforms, LetterBox, Format, Results,
    DetectionTrainer, DetectionModel
)


Default = None


@dataclass
class YOLOTrainerArguments:
    # Basic training params
    epochs: int = 100
    batch: int = -1
    val_batch: int = -1

    # Optimizer params
    optimizer: str = "SGD"  # SGD, Adam, AdamW, auto
    lr0: float = 0.01  # initial learning rate
    lrf: float = 0.01  # final learning rate (lr0 * lrf)
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: int = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1

    # Loss params
    box: float = 7.5  # box loss gain
    cls: float = 0.5  # cls loss gain
    dfl: float = 1.5  # dfl loss gain

    # Other params
    amp: bool = True  # automatic mixed precision
    device: str = ""  # cuda device, e.g. 0 or 0,1,2,3 or cpu
    workers: int = 0  # number of worker threads
    project: str = "./results"  # project name
    name: str = "yolo11_training"  # experiment name
    exist_ok: bool = False  # overwrite existing experiment
    seed: int = 0  # random seed
    deterministic: bool = True  # deterministic mode
    single_cls: bool = False  # train as single-class dataset
    rect: bool = False  # rectangular training
    cos_lr: bool = False  # cosine learning rate scheduler
    close_mosaic: int = 10  # disable mosaic augmentation for final N epochs
    save: bool = True  # save checkpoints
    save_period: int = -1  # save checkpoint every N epochs (-1 = disabled)
    cache: bool = False  # cache images for faster training
    val: bool = True  # validate/test during training
    patience: int = 50  # early stopping patience (epochs without improvement)
    plots: bool = True  # save plots during training


class YOLOTrainer(DetectionTrainer):
    def __init__(
        self,
        model: BaseModel,
        classes: list[str],
        train_dataset: DataPreparation | None = None,
        eval_dataset: DataPreparation | None = None,
        args: YOLOTrainerArguments | None = None,
        **kwargs
    ):
        self.classes = classes
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.custom_args = args if args is not None else YOLOTrainerArguments()
        self.train_batch = self.custom_args.batch
        self.eval_batch = self.custom_args.val_batch
        del self.custom_args.val_batch  # yolo does not accept val_batch

        # Convert args to YOLO cfg format
        overrides = {k: v for k, v in vars(self.custom_args).items()}
        overrides['model'] = ""  # placeholder for model path
        overrides['resume'] = False
        overrides['data'] = self.train_dataset.data if self.train_dataset is not None else self.eval_dataset.data
        if eval_dataset is not None:
            overrides['conf'] = eval_dataset.confidence_threshold
            overrides['iou'] = eval_dataset.iou_threshold
            overrides['imgsz'] = eval_dataset.img_size
        else:
            overrides['imgsz'] = train_dataset.img_size
        if train_dataset is None:  # disable saving if no training dataset
            overrides['save'] = False
            overrides['plots'] = False
            overrides['project'] = None
            overrides['name'] = None
            overrides['val'] = True
            overrides['batch'] = 1  # prevent exception cased from auto_batch

        # Initialize parent DetectionTrainer
        self.epoch = 0
        self.loss_items = None
        super().__init__(overrides=overrides)
        self.model = model
        if not self.data:
            self.data = self.args.data

    @property
    def names(self):
        return self.args.data['names']

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        return self.train_dataset if mode == 'train' else self.eval_dataset

    def get_dataset(self):
        return {}

    def get_dataloader(self, dataset_path: str = None, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        dataset = self.train_dataset if mode == 'train' else self.eval_dataset
        _batch_size = self.train_batch if mode == 'train' else self.eval_batch

        return build_dataloader(
            dataset,
            batch=batch_size if _batch_size == -1 else _batch_size,
            workers=self.custom_args.workers,
            shuffle=(mode == 'train')
        )

    def setup_model(self):
        if self.resume:
            ckpt = torch.load(self.args.resume, weights_only=False, map_location="cpu")
            self.model.load(ckpt['model'] if ckpt['model'] else ckpt['ema'])
            return ckpt
        else:
            return None

    def resume_from_checkpoint(self):
        self.args.resume = True
        try:
            self.check_resume(self.args.__dict__)
        except FileNotFoundError as e:
            LOGGER.warning(str(e))
        if self.resume:  # if check_resume changed self.resume to True
            print(f"Resuming configuration is now set to checkpoint {self.args.resume}")

    def validate(self):
        if self.loss_items is None:
            self._setup_train()
            self.loss_items = torch.zeros(len(self.loss_names), device=self.device)
        if not hasattr(self, "loss") or self.loss is None:
            self.loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.validator.args.verbose = True
        return super().validate()

    def final_eval(self):
        """Override final_eval to use trainer-based validation instead of model path"""
        if not self.args.save or not self.best.exists():
            return

        # Load the best checkpoint into current model
        try:
            from ultralytics.utils import LOGGER
            LOGGER.info(f"\nValidating {self.best}...")

            ckpt = torch.load(self.best, weights_only=False, map_location="cpu")
            self.model.load(ckpt.get('model') or ckpt.get('ema'))

            # Use trainer-based validation (not model path validation)
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False

            # Call validator with trainer (self), not model path
            metrics = self.validator(self)

            if metrics:
                self.metrics = metrics
                self.metrics.pop("fitness", None)
                self.run_callbacks("on_fit_epoch_end")

        except Exception as e:
            LOGGER.warning(f"Final evaluation failed: {e}")


class YOLODataPreparation(DataPreparation):
    def __init__(
        self,
        dataset: BaseDataset,
        dataset_key: dict = dict(bboxes="boxes2d", classes="boxes2d_classes", original_size="original_hw"),
        img_size: int = 800,
        evaluation_mode: bool = False,
        confidence_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        max_detection: int = 300,
        train_strong_transforms: Optional[Compose] = Default,  # Use YOLO's pre-configured v8_transforms as augmentation
        train_weak_transforms: Optional[Compose] = Default,
        valid_transforms: Optional[Compose] = Default
    ):
        self.dataset_name = dataset.dataset_name
        self.classes = dataset.classes
        self.names = {i: name for i, name in enumerate(dataset.classes)}
        self.data = dict(nc=len(self.classes), names=self.names, channels=3, train="")  # yolo trainer compatibility
        self.buffer = []  # yolo trainer compatibility

        self.dataset = dataset
        self.dataset_key = dataset_key
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.evaluation_mode = evaluation_mode
        self.enable_strong_augment = False
        self.max_detection = max_detection

        if evaluation_mode:
            self.strong_transforms = None
            if valid_transforms is Default:
                self.default_transforms = Compose([LetterBox(new_shape=(img_size, img_size))])
            else:
                self.default_transforms = valid_transforms
        else:
            # For compatibility with YOLO v8_transforms
            dataset = copy(dataset)  # copy temporary dataset
            dataset.get_image_and_label = lambda idx: self.convert_to_yolo_label_format(idx, *dataset.__getitem__(idx))
            dataset.cache = "ram"
            dataset.buffer = self.buffer
            dataset.data = dict(flip_idx=[], kpt_shape=None)  # disable keypoint
            dataset.use_keypoints = False  # disable keypoint

            self.enable_strong_augment = True
            if train_strong_transforms is Default:
                hyp = get_cfg()
                hyp.__dict__.update(dict(mosaic=1.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, translate=0.1, scale=0.5))
                self.strong_transforms = v8_transforms(dataset=dataset, imgsz=img_size, hyp=hyp, stretch=False)
            else:
                self.strong_transforms = train_strong_transforms
            self.strong_transforms.append(Format(bbox_format="xywh", normalize=True, batch_idx=True))
            if train_weak_transforms is Default:
                hyp = get_cfg()
                hyp.__dict__.update(dict(mosaic=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, fliplr=0.5, translate=0.1, scale=0.5))
                self.default_transforms = v8_transforms(dataset=dataset, imgsz=img_size, hyp=hyp, stretch=False)
            else:
                self.default_transforms = train_weak_transforms
        self.default_transforms.append(Format(bbox_format="xywh", normalize=True, batch_idx=True))

    def __len__(self):
        return len(self.dataset)

    def close_mosaic(self, *args, **kwargs):
        LOGGER.info(f"Closing mosaic augmentation...")
        self.enable_strong_augment = False

    @property
    def labels(self):
        LOGGER.warning("YOLO Trainer tries to access labels property probably for plotting. If label loading takes too long, you can disable it by setting plots=False in the training arguments.")
        if not hasattr(self, "_labels") or self._labels is None:
            self._labels = []
            for idx in range(len(self.dataset)):
                yolo_data = self.convert_to_yolo_label_format(idx, *self.dataset[idx])
                self._labels.append({
                    'bboxes': yolo_data['instances'].bboxes,  # Instances -> numpy array
                    'cls': yolo_data['cls']
                })
        return self._labels

    @property
    def augmentation(self):
        if self.enable_strong_augment:
            return self.strong_transforms
        else:
            return self.default_transforms

    def convert_to_yolo_label_format(self, idx: int, image: torch.Tensor, target: dict) -> dict:
        bbox = target[self.dataset_key['bboxes']]
        bbox_classes = target[self.dataset_key['classes']]
        original_height, original_width = target[self.dataset_key['original_size']]

        # Convert to numpy for YOLO transforms (YOLO uses OpenCV/numpy internally)
        if isinstance(image, torch.Tensor):
            # CHW (RGB) -> CHW (BGR)
            image = image[[2, 1, 0], :, :]  # R(0), G(1), B(2) -> B(2), G(1), R(0)
            # CHW (BGR) -> HWC (BGR)
            image = image.permute(1, 2, 0).mul(255).byte().numpy()

        # Convert bbox to numpy and ensure XYXY format
        if isinstance(bbox, BoundingBoxes):
            if bbox.format != BoundingBoxFormat.XYXY:
                bbox = convert_bounding_box_format(bbox, new_format=BoundingBoxFormat.XYXY)
            bbox = bbox.data.numpy() if isinstance(bbox.data, torch.Tensor) else bbox.data
        elif isinstance(bbox, torch.Tensor):
            LOGGER.warning_once("Assume the bbox is in Pascal VOC format (x1, y1, x2, y2) since it's not a BoundingBoxes instance. Please ensure this is correct.")
            bbox = bbox.numpy()

        # Convert bbox_classes to numpy
        if isinstance(bbox_classes, torch.Tensor):
            bbox_classes = bbox_classes.numpy()

        # Dummy segments for YOLO
        segments = np.zeros((0, 1000, 2), dtype=np.float32)

        return {
            'img': image,
            'instances': Instances(bbox, segments, bbox_format="xyxy", normalized=False),
            'cls': bbox_classes.reshape(-1, 1),
            'batch_idx': np.array([idx if idx is not None else 0]),
            'im_file': str(idx),
            'ori_shape': (original_height, original_width),
            'resized_shape': (original_height, original_width),
            'ratio_pad': None
        }

    def transforms(self, *data, idx=None):
        image, target = data[0] if len(data) == 1 else data
        yolo_data = self.convert_to_yolo_label_format(idx, image, target)

        # Apply YOLO augmentation
        transformed = self.augmentation(yolo_data)

        if len(data) == 1:
            return transformed
        else:
            return transformed['img'], transformed

    def __getitem__(self, idx):
        return self.transforms(self.dataset[idx], idx=idx)

    def collate_fn(self, batch: list[dict] | list[tuple[torch.Tensor, dict]]) -> dict:
        if isinstance(batch[0], tuple):
            batch = [b[1] for b in batch]

        batch = self.pre_process(batch)

        new_batch = {}
        keys = batch[0].keys()

        for key in keys:
            values = [b[key] for b in batch]

            if key == "img":
                new_batch[key] = torch.stack(values, 0)
            elif key in {"bboxes", "cls"}:
                new_batch[key] = torch.cat(values, 0)
            else:
                new_batch[key] = values

        # Add batch index for each sample
        batch_idx = [idx + i for i, idx in enumerate(new_batch["batch_idx"])]
        new_batch["batch_idx"] = torch.cat(batch_idx, 0)
        return new_batch

    def pre_process(self, batch: list[dict]) -> list[dict]:
        for b in batch:
            if isinstance(b['img'], torch.ByteTensor):
                b['img'] = b['img'].float().div(255.0)
        return batch

    def post_process(
        self, outputs: torch.Tensor, ori_shape: list[tuple[int, int]], resized_shape: list[tuple[int, int]],
        names: dict[int, str], xywh: bool = False
    ) -> list[Results]:
        """ Apply nms and rescale bounding boxes to original image size

        Args:
            outputs (torch.Tensor): YOLO model outputs with bbox format (N, 4).
            ori_shape (list[tuple[int, int]]): Shape of the target image (height, width).
            resized_shape (list[tuple[int, int]]): Actual resized shape after LetterBox (height, width).
            names (dict): Dictionary of class names.
            xywh (bool): Whether box format is xywh (True) or xyxy (False).

        Returns:
            (list[Results]): Rescaled bounding boxes in the same format as input.
        """
        filtered = nms.non_max_suppression(
            outputs,
            conf_thres=self.confidence_threshold,
            iou_thres=self.iou_threshold,
            multi_label=False,
            max_det=self.max_detection
        )
        orig_img = np.ndarray(0)
        results = [Results(orig_img=orig_img, path="", names=names, boxes=(
            dets if dets.shape[0] == 0 else torch.cat((
                ops.scale_boxes(
                    img1_shape=resized_shape[i], boxes=dets[:, :4].clone(),
                    img0_shape=ori_shape[i], xywh=xywh
                ), dets[:, 4:]
            ), dim=1)
        )) for i, dets in enumerate(filtered)]
        for r in results:
            r.orig_img = None
            r.orig_shape = ori_shape
        return results


class YOLO11ForObjectDetection(DetectionModel, BaseModel):
    model_name = "YOLO11m"
    model_config = "yolo11m.yaml"
    model_provider = ModelProvider.Ultralytics
    DataPreparation = YOLODataPreparation
    Trainer = YOLOTrainer
    TrainingArguments = YOLOTrainerArguments
    channel = 3

    class Weights:
        COCO_OFFICIAL = WeightsInfo("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt", weight_key="model")
        SHIFT_CLEAR = WeightsInfo("")
        CITYSCAPES = WeightsInfo("")

    def __init__(self, dataset: BaseDataset):
        nc = len(dataset.classes)
        super().__init__(self.model_config, ch=self.channel, nc=nc)

        self.dataset_name = dataset.dataset_name
        self.num_classes = nc
        self.names = {i: name for i, name in enumerate(dataset.classes)}

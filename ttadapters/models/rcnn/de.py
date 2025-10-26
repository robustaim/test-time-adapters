from typing import Callable
import warnings
import random
import gc

import torch
from torch import nn, cuda, from_numpy
from torch.utils.data import DataLoader
from torchvision.tv_tensors import Image, BoundingBoxFormat, BoundingBoxes
from torchvision.transforms.v2.functional import convert_bounding_box_format

from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import PeriodicWriter

from detectron2.structures import Boxes, Instances
from detectron2.data.samplers import TrainingSampler
from detectron2.data.transforms import AugmentationList, AugInput, ResizeShortestEdge, RandomFlip

from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.modeling import GeneralizedRCNN, SwinTransformer
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo

from IPython.display import display
from ipywidgets import Output

import pandas as pd
import numpy as np

from .hook import TqdmProgressHook
from .transforms import PermuteChannels, ConvertRGBtoBGR
from ..base import BaseModel, ModelProvider, WeightsInfo
from ...datasets import BaseDataset, DataPreparation
from ...utils.validator import DetectionEvaluator


DETECTRON_URL_PREFIX = "https://dl.fbaipublicfiles.com/detectron2/"


class DetectronTrainerArgs:
    def __init__(
        self,
        learning_rate: float = 1e-4,
        total_steps: int = 40000,
        eval_period: int = 1000,
        save_period: int = 1000,
        train_batch_for_total: int = 16,
        eval_batch_for_total: int = 32,
        multiple_gpu_world_size: int = 0,  # Set 0 to disable multi-GPU reference
        num_workers: int = 0,  # Should be 0 on Windows
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        weight_decay_norm: float = 0.0,
        lr_scheduler_type: str = "WarmupCosineLR",  # WarmupMultiStepLR, WarmupStepWithFixedGammaLR
        cosine_lr_final: float = 1e-5,
        multi_step_lr_gamma: float = 0.1,
        multi_step_lr_points: list[int] | tuple[int] = (30000, 35000),
        fixed_gamma_lr_num_decays: int = 3,
        lr_warmup_method: str = "linear",
        lr_warmup_iters: int = 1000,
        lr_warmup_factor: float = 1e-3,
        lr_warmup_rescale_interval: bool = False,
        use_gradient_clipping: bool = False,
        gradient_clipping_type: str = "value",
        gradient_clipping_value: float = 1.0,
        gradient_clipping_norm_type: float = 2.0,
        use_amp: bool = False,
        output_dir: str = "./results",
        seed: int = 42
    ):
        # Merge default config
        cfg = get_cfg()
        for key in cfg:
            setattr(self, key, cfg[key])
        self._cfg = cfg

        # Solver settings
        self.SOLVER.LR_SCHEDULER_NAME = lr_scheduler_type
        self.SOLVER.MAX_ITER = total_steps
        self.SOLVER.BASE_LR = learning_rate
        self.SOLVER.BASE_LR_END = cosine_lr_final  # The end lr, only used by WarmupCosineLR
        self.SOLVER.MOMENTUM = momentum
        self.SOLVER.WEIGHT_DECAY = weight_decay
        self.SOLVER.WEIGHT_DECAY_NORM = weight_decay_norm
        self.SOLVER.GAMMA = multi_step_lr_gamma  # WarmupMultiStepLR
        self.SOLVER.STEPS = multi_step_lr_points  # WarmupMultiStepLR
        self.SOLVER.NUM_DECAYS = fixed_gamma_lr_num_decays  # WarmupStepWithFixedGammaLR
        self.SOLVER.WARMUP_FACTOR = lr_warmup_factor
        self.SOLVER.WARMUP_ITERS = lr_warmup_iters
        self.SOLVER.WARMUP_METHOD = lr_warmup_method
        self.SOLVER.RESCALE_INTERVAL = lr_warmup_rescale_interval
        self.SOLVER.CHECKPOINT_PERIOD = save_period
        self.SOLVER.IMS_PER_BATCH = train_batch_for_total
        self.SOLVER.EVAL_IMS_PER_BATCH = eval_batch_for_total
        self.SOLVER.REFERENCE_WORLD_SIZE = multiple_gpu_world_size
        self.SOLVER.CLIP_GRADIENTS['ENABLED'] = use_gradient_clipping
        self.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = gradient_clipping_type
        self.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = gradient_clipping_value
        self.SOLVER.CLIP_GRADIENTS.NORM_TYPE = gradient_clipping_norm_type
        self.SOLVER.AMP['ENABLED'] = use_amp

        # Dataloader settings
        self.DATALOADER.NUM_WORKERS = num_workers
        self.SEED = seed

        # Evaluation settings
        self.TEST.EVAL_PERIOD = eval_period

        # Output settings
        self.OUTPUT_DIR = output_dir

    def __repr__(self):
        args = {key: val for key, val in self.__dict__.items() if key in ["SOLVER", "DATALOADER", "TEST", "OUTPUT_DIR"]}
        return f"{self.__class__.__name}({args})"

    def __str__(self):
        return self.__repr__()

    def clone(self):
        return self  # does not clone

    def freeze(self):
        self._cfg.freeze()

    def defrost(self):
        self._cfg.defrost()


class DetectronTrainer(DefaultTrainer):
    def __init__(
        self,
        model: BaseModel,
        classes: list[str],
        train_dataset: DataPreparation | None = None,
        eval_dataset: DataPreparation | None = None,
        args: DetectronTrainerArgs | None = None
    ):
        self._model = model  # need to do this since self.build_model is called in DefaultTrainer.__init__
        self.args: DetectronTrainerArgs = args if args is not None else DetectronTrainerArgs()
        self.classes = classes

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.eval_dataset_name = "unknown_dataset"
        if hasattr(self.eval_dataset, "dataset"):
            if hasattr(self.eval_dataset, "dataset_name"):
                self.eval_dataset_name = self.eval_dataset.dataset.dataset_name
        elif hasattr(self.eval_dataset, "dataset_name"):
            self.eval_dataset_name = self.eval_dataset.dataset_name

        self._train_loop_running = False
        super().__init__(cfg=self.args)

    def build_model(self, *args, **kwargs):
        model = getattr(self, "model", None)
        if model is None:
            model = self._model
            del self._model
        return model  # model is passed in constructor

    def build_train_loader(self, *args, **kwargs):
        if self.train_dataset is None:
            return None

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        generator = torch.Generator()
        generator.manual_seed(self.args.SEED * 2)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.SOLVER.IMS_PER_BATCH,
            sampler=TrainingSampler(
                size=len(self.train_dataset),
                shuffle=True,
                seed=self.args.SEED
            ),
            num_workers=self.args.DATALOADER.NUM_WORKERS,
            collate_fn=self.train_dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=generator
        )

    def build_test_loader(self, *args, **kwargs):
        if self.eval_dataset is None:
            return None
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.SOLVER.EVAL_IMS_PER_BATCH,
            shuffle=False,
            num_workers=self.args.DATALOADER.NUM_WORKERS,
            collate_fn=self.eval_dataset.collate_fn
        )

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks = [h for h in hooks if not isinstance(h, PeriodicWriter)]
        hooks.append(TqdmProgressHook())
        return hooks

    def train(self):
        cuda.empty_cache()
        gc.collect()

        self._train_loop_running = True
        try:
            result = super().train()
        except Exception as e:
            raise e
        finally:
            self._train_loop_running = False
            if getattr(self, "_temp_output", None):
                del self._temp_output
        return result

    def test(self, *args, **kwargs):
        cuda.empty_cache()
        gc.collect()

        if self._train_loop_running:
            output, df = getattr(self, "_temp_output", [None, None])
            if output is None:
                (output, df) = self._temp_output = [Output(), pd.DataFrame()]
                display(output)
        else:
            output, df = Output(), pd.DataFrame()
            display(output)

        with output:
            loader = self.build_test_loader()
            self.model.eval()
            result = DetectionEvaluator.evaluate(
                model=self.model,
                desc=self.eval_dataset_name,
                loader=loader,
                loader_length=int(len(self.eval_dataset)/loader.batch_size+0.99),
                classes=self.classes,
                data_preparation=self.eval_dataset,
                dtype=self.model.pixel_mean.dtype,
                device=self.model.pixel_mean.device,
                synchronize=False,
                no_grad=True
            )
            self.model.train()

            df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
            if self._train_loop_running:
                self._temp_output[1] = df
            output.clear_output(wait=True)
            display(df)
        return result


class DetectronDataPreparation(DataPreparation):
    def __init__(
        self,
        dataset: BaseDataset,
        dataset_key: dict = dict(bboxes="boxes2d", classes="boxes2d_classes", original_size="original_hw"),
        img_size: int = 800,
        evaluation_mode: bool = False,
        train_transforms: AugmentationList = AugmentationList([  # Detectron2 Faster R-CNN default
            PermuteChannels(),  # (C, H, W) -> (H, W, C)
            ConvertRGBtoBGR(),
            ResizeShortestEdge([640, 672, 704, 736, 768, 800], max_size=1600, sample_style="choice"),  # change max_size to 1600 for cityscapes dataset; original is 1333.
            RandomFlip(prob=0.5),  # Random horizontal flip with 50% probability
            PermuteChannels(reverse=True)  # (H, W, C) -> (C, H, W) - stupid detectron;;
        ]),
        valid_transforms: AugmentationList = AugmentationList([  # Detectron2 Faster R-CNN default
            PermuteChannels(),  # (C, H, W) -> (H, W, C)
            ConvertRGBtoBGR(),
            ResizeShortestEdge(800, max_size=1600),  # change max_size to 1600 for cityscapes dataset; original is 1333.
            PermuteChannels(reverse=True)  # (H, W, C) -> (C, H, W) - stupid detectron;;
        ])
    ):
        self.dataset_name = dataset.dataset_name
        self.classes = dataset.classes

        self.dataset = dataset
        self.dataset_key = dataset_key
        self.img_size = img_size
        self.evaluation_mode = evaluation_mode

        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms
        for tr in valid_transforms.augs:
            if isinstance(tr, ResizeShortestEdge) and tr.short_edge_length[0] != img_size:
                raise ValueError(f"ResizeShortestEdge is not set to {img_size} for valid_transforms")

        self.pre_process: Callable = lambda batch: batch
        self.post_process: Callable = lambda batch, *args, **kwargs: batch

    def transforms(self, *data):
        image, metadata = data[0] if len(data) == 1 else data

        bboxes = metadata[self.dataset_key['bboxes']]
        bbox_classes = metadata[self.dataset_key['classes']]
        original_height, original_width = metadata[self.dataset_key['original_size']]

        if not isinstance(bboxes, BoundingBoxes):
            warnings.warn("Assume the bbox is in Pascal VOC format (x1, y1, x2, y2) since it's not a BoundingBoxes instance. Please ensure this is correct.")
            bboxes = BoundingBoxes(bboxes, format=BoundingBoxFormat.XYXY, canvas_size=(original_height, original_width))

        if bboxes.format != BoundingBoxFormat.XYXY:  # from Pascal VOC format (x1, y1, x2, y2)
            bboxes = convert_bounding_box_format(bboxes, new_format=BoundingBoxFormat.XYXY)

        aug_input = AugInput(image.numpy(), boxes=bboxes.numpy())
        self.train_transforms(aug_input) if not self.evaluation_mode else self.valid_transforms(aug_input)
        transformed_image = from_numpy(aug_input.image.copy())
        transformed_boxes = from_numpy(aug_input.boxes.copy())

        except_keys = list(self.dataset_key.values())
        resized_height, resized_width = transformed_image.shape[-2:]  # HWC
        instances = Instances(image_size=(resized_height, resized_width))
        instances.gt_boxes = Boxes(transformed_boxes)
        instances.gt_classes = bbox_classes

        return transformed_image, {
            'image': transformed_image,
            'instances': instances,
            'height': original_height,
            'width': original_width,
            'metadata': {key: val for key, val in metadata.items() if key not in except_keys}
        }

    def __getitem__(self, idx):
        return self.transforms(self.dataset[idx])

    def collate_fn(self, batch: list[Image, dict]):
        return [metadata for _, metadata in batch]


class FasterRCNNForObjectDetection(GeneralizedRCNN, BaseModel):
    model_name = "Faster_R-CNN-R50"
    model_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    model_provider = ModelProvider.Detectron2
    DataPreparation = DetectronDataPreparation
    Trainer = DetectronTrainer
    TrainingArguments = DetectronTrainerArgs

    class Weights:
        IMAGENET_OFFICIAL = WeightsInfo("detectron2://ImageNetPretrained/MSRA/R-50.pkl")
        COCO_OFFICIAL = WeightsInfo(model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml").replace(DETECTRON_URL_PREFIX, "detectron2://"))
        SHIFT_CLEAR_NATUREYOO = WeightsInfo("https://github.com/robustaim/ContinualTTA_ObjectDetection/releases/download/backbone/Faster_R-CNN_Resnet_50_SHIFT.pth", weight_key="model")
        CITYSCAPES = WeightsInfo("https://github.com/robustaim/test-time-adapters/releases/download/pretrained/Faster_R-CNN_Resnet_50_CityScapes.pth", weight_key="model")

    def __init__(self, dataset: BaseDataset):
        num_classes = len(dataset.classes)

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.model_config))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.FPN.TOP_LEVELS = 2

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*To copy construct from a tensor.*')
            modules = build_model(cfg)
            super().__init__(
                backbone=modules.backbone,
                proposal_generator=modules.proposal_generator,
                roi_heads=modules.roi_heads,
                pixel_mean=modules.pixel_mean,
                pixel_std=modules.pixel_std,
                input_format=modules.input_format,
                vis_period=modules.vis_period,
            )

        self.dataset_name = dataset.dataset_name
        self.num_classes = num_classes


class SwinRCNNForObjectDetection(GeneralizedRCNN, BaseModel):
    model_name = "SwinT_R-CNN-Tiny"
    model_provider = ModelProvider.Detectron2
    DataPreparation = DetectronDataPreparation
    Trainer = DetectronTrainer
    TrainingArguments = DetectronTrainerArgs
    default_params = dict(
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        frozen_stages=-1,
        out_indices=(0, 1, 2, 3)
    )

    class Weights:
        COCO_XIAOHU2015 = WeightsInfo("https://github.com/xiaohu2015/SwinT_detectron2/releases/download/v1.1/faster_rcnn_swint_T.pth", weight_key="model", exclude_keys = [
            "roi_heads.box_predictor.cls_score.weight",
            "roi_heads.box_predictor.cls_score.bias",
            "roi_heads.box_predictor.bbox_pred.weight",
            "roi_heads.box_predictor.bbox_pred.bias"
        ])
        SHIFT_CLEAR_NATUREYOO = WeightsInfo("https://github.com/robustaim/ContinualTTA_ObjectDetection/releases/download/backbone/Faster_R-CNN_SwinT_Tiny_SHIFT.pth", weight_key="model")
        CITYSCAPES = WeightsInfo("https://github.com/robustaim/test-time-adapters/releases/download/pretrained/Faster_R-CNN_SwinT_Tiny_CityScapes.pth", weight_key="model")

    def __init__(self, dataset: BaseDataset):
        num_classes = len(dataset.classes)

        cfg = get_cfg()
        base_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        cfg.merge_from_file(model_zoo.get_config_file(base_config))

        cfg.MODEL.MASK_ON = False
        cfg.MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
        cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395]

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            modules = build_model(cfg)
            cfg.MODEL.FPN.IN_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

            swin_backbone = SwinTransformer(**self.default_params)
            swin_backbone._out_features = ["stage{}".format(i+2) for i in swin_backbone.out_indices]
            swin_backbone._out_feature_channels = {
                "stage{}".format(i+2): swin_backbone.embed_dim * 2**i
                for i in swin_backbone.out_indices
            }
            swin_backbone._out_feature_strides = {
                "stage{}".format(i+2): 2 ** (i + 2)
                for i in swin_backbone.out_indices
            }
            original_forward = swin_backbone.forward

            def patched_forward(x):
                outs_orig = original_forward(x)
                outs = {}
                for i in swin_backbone.out_indices:
                    outs["stage{}".format(i+2)] = outs_orig["p{}".format(i)]
                return outs

            swin_backbone.forward = patched_forward

            super().__init__(
                backbone=FPN(
                    bottom_up=swin_backbone,
                    in_features=cfg.MODEL.FPN.IN_FEATURES,
                    out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
                    norm=cfg.MODEL.FPN.NORM,
                    top_block=LastLevelMaxPool(),
                    fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
                ),
                proposal_generator=modules.proposal_generator,
                roi_heads=modules.roi_heads,
                pixel_mean=modules.pixel_mean,
                pixel_std=modules.pixel_std,
                input_format=modules.input_format,
                vis_period=modules.vis_period,
            )

        self.dataset_name = dataset.dataset_name
        self.num_classes = num_classes

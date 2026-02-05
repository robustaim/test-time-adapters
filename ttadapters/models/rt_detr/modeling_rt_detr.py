from dataclasses import dataclass
from typing import Optional
import random
import gc

from transformers import RTDetrConfig, RTDetrImageProcessorFast, RTDetrForObjectDetection as _RTDetrForObjectDetection
from transformers.trainer import Trainer, TrainingArguments, is_sagemaker_mp_enabled, logger
from transformers.trainer_utils import EvalPrediction

from torchvision.transforms.v2.functional import convert_bounding_box_format
from torchvision.tv_tensors import BoundingBoxFormat, BoundingBoxes
from torchvision.transforms import v2 as T
from torchvision.ops import box_convert
from torch import nn
import torch

from supervision.detection.core import Detections
from supervision.metrics.mean_average_precision import MeanAveragePrecision

from ..base import BaseModel, ModelProvider, WeightsInfo
from ...datasets import BaseDataset, DataPreparation


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


@dataclass
class RTDetrTrainingArguments(TrainingArguments):
    backbone_learning_rate: float = None  # Set to 1/10th of the main learning rate if not specified


class RTDetrTrainer(Trainer):
    def __init__(
        self,
        model: BaseModel,
        classes: list[str],
        train_dataset: DataPreparation | None = None,
        eval_dataset: DataPreparation | None = None,
        args: RTDetrTrainingArguments | None = None,
        **kwargs
    ):
        self.map_metric = MeanAveragePrecision()
        self.eval_post_process = eval_dataset.post_process if eval_dataset is not None else lambda x, *args, **kwargs: x
        self.classes = classes

        data_collator = kwargs.pop("data_collator", None)
        if data_collator is None:
            self.train_collator = train_dataset.collate_fn if train_dataset else None
            self.eval_collator = eval_dataset.collate_fn if eval_dataset else None
        else:
            self.train_collator = data_collator
            self.eval_collator = data_collator

        if "compute_metrics" in kwargs:
            compute_metrics = kwargs.pop("compute_metrics")
        else:
            compute_metrics = self.compute_metrics

        super().__init__(
            model=model, train_dataset=train_dataset, eval_dataset=eval_dataset,
            compute_metrics=compute_metrics, args=args, **kwargs
        )

    def train(self, *args, **kwargs):
        self.map_metric.reset()
        torch.cuda.empty_cache()
        gc.collect()
        return super().train(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        self.map_metric.reset()
        torch.cuda.empty_cache()
        gc.collect()
        return super().evaluate(*args, **kwargs)

    def get_train_dataloader(self):
        self.data_collator = self.train_collator
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        self.data_collator = self.eval_collator
        return super().get_eval_dataloader(eval_dataset)

    def _inner_training_loop(self, *args, **kwargs):
        current_epoch = int(self.state.epoch) if self.state.epoch else 0

        try:
            if current_epoch < self.train_dataset.strong_augment_threshold_epoch:
                if not self.train_dataset.enable_strong_augment:
                    self.train_dataset.enable_strong_augment = True
                    logger.info(f"Enable strong augment starting for epoch {current_epoch}")
        except AttributeError:
            pass

        result = super()._inner_training_loop(*args, **kwargs)

        try:
            if current_epoch >= self.train_dataset.strong_augment_threshold_epoch:
                if self.train_dataset.enable_strong_augment:
                    self.train_dataset.enable_strong_augment = False
                    logger.info(f"Disable strong augment starting for epoch {current_epoch} since threshold reached")
        except AttributeError:
            pass

        return result

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            backbone_params = []
            other_params = []
            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue
                if n.startswith('bert.') or n.startswith('backbone.'):
                    backbone_params.append((n, p))
                else:
                    other_params.append((n, p))
            backbone_learning_rate = getattr(self.args, "backbone_learning_rate", None)
            if backbone_learning_rate is None:
                backbone_learning_rate = self.args.learning_rate * 0.1
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in backbone_params if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                    "lr": backbone_learning_rate,
                },
                {
                    "params": [p for n, p in backbone_params if n not in decay_parameters],
                    "weight_decay": 0.0,
                    "lr": backbone_learning_rate,
                },
                {
                    "params": [p for n, p in other_params if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in other_params if n not in decay_parameters],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if "bitsandbytes" in str(optimizer_cls) and optimizer_kwargs.get("optim_bits", None) == 8:
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def compute_metrics(self, eval_pred: EvalPrediction, compute_result=False):
        if compute_result:
            m_ap = self.map_metric.compute()
            self.map_metric.reset()

            per_class_map = {
                f"{self.classes[idx]}_mAP@0.50:0.95": m_ap.ap_per_class[idx].mean()
                for idx in m_ap.matched_classes
            }

            return {
                "mAP@0.50:0.95": m_ap.map50_95,
                "mAP@0.50": m_ap.map50,
                "mAP@0.75": m_ap.map75,
                **per_class_map
            }
        else:
            preds = ModelOutput(*eval_pred.predictions[1:3])
            labels = eval_pred.label_ids
            sizes = [label['orig_size'].cpu().tolist() for label in labels]

            results = self.eval_post_process(preds, target_sizes=sizes)

            predictions = [Detections.from_transformers(result) for result in results]
            targets = [Detections(
                xyxy=(box_convert(label['boxes'], "cxcywh", "xyxy") * label['orig_size'].flip(0).repeat(2)).cpu().numpy(),
                class_id=label['class_labels'].cpu().numpy(),
            ) for label in labels]  # keep `normalized` xyxy format

            self.map_metric.update(predictions=predictions, targets=targets)
            return {}


class RTDetrDataPreparation(DataPreparation):
    model_id = "PekingU/rtdetr_r50vd"

    def __init__(
        self,
        dataset: BaseDataset,
        dataset_key: dict = dict(bboxes="boxes2d", classes="boxes2d_classes", original_size="original_hw"),
        img_size: int = 800,
        evaluation_mode: bool = False,
        confidence_threshold: float = 0.05,
        strong_augment_threshold_epoch: int = 0,  # duration epoch of strong augment
        multi_scale: list[int] = [480, 640, 800],
        longest_edge: int = 1600,  # change max_size to 1600 for cityscapes dataset; original is 1333.
        strong_augment: T.Compose = T.Compose([
            T.RandomPhotometricDistort(),
            T.RandomZoomOut(),
            T.RandomIoUCrop(),
            T.Resize(size=(800, 1280)),  # required to be overridden with dataset img_size
            T.RandomHorizontalFlip()
        ]),
        default_augment: T.Compose = T.Compose([
            T.RandomHorizontalFlip()
        ])
    ):
        self.dataset_name = dataset.dataset_name
        self.classes = dataset.classes

        self.dataset = dataset
        self.dataset_key = dataset_key
        self.img_size = img_size
        self.longest_edge = longest_edge
        self.confidence_threshold = confidence_threshold
        self.evaluation_mode = evaluation_mode

        if self.evaluation_mode:
            self.enable_strong_augment = False
            self.strong_augment_threshold_epoch = 0
            self.multi_scale = [img_size]
            self.strong_augment = lambda inputs: inputs
            self.default_augment = lambda inputs: inputs
        else:
            self.enable_strong_augment = True
            self.strong_augment_threshold_epoch = strong_augment_threshold_epoch
            self.multi_scale = multi_scale
            self.strong_augment = strong_augment
            self.default_augment = default_augment

        self.image_processor = RTDetrImageProcessorFast.from_pretrained(self.model_id)
        self.image_processor.size = {"shortest_edge": self.img_size, "longest_edge": self.longest_edge}
        self.image_processor.do_resize = True

    def transforms(self, *args, idx=None):
        augmentation = self.strong_augment if self.enable_strong_augment else self.default_augment
        image, target = args[0] if len(args) == 1 else args

        bbox = target[self.dataset_key['bboxes']]
        bbox_classes = target[self.dataset_key['classes']]
        img_size = target[self.dataset_key['original_size']]

        image, bbox = augmentation((image, bbox))

        if not isinstance(bbox, BoundingBoxes):
            logger.warning_once("Assume the bbox is in Pascal VOC format (x1, y1, x2, y2) since it's not a BoundingBoxes instance. Please ensure this is correct.")
            bbox = BoundingBoxes(bbox, format=BoundingBoxFormat.XYXY, canvas_size=img_size)

        if bbox.format != BoundingBoxFormat.XYWH:  # from Pascal VOC format (x1, y1, x2, y2)
            bbox = convert_bounding_box_format(bbox, new_format=BoundingBoxFormat.XYWH)  # to COCO format: [x, y, width, height]

        # Convert to COCO_Detection Format
        annotations = []
        target = dict(image_id=idx, annotations=annotations)
        for box, cls in zip(bbox, bbox_classes):
            width, height = box[2:].tolist()
            annotations.append(dict(
                bbox=box,
                category_id=cls.item(),
                area=width*height,
                iscrowd=0
            ))

        # Following prepare_coco_detection_annotation's expected format
        # RT-DETR ImageProcessor converts the COCO bbox to center format (cx, cy, w, h) during preprocessing
        # But, eventually re-converts the bbox to Pascal VOC (x1, y1, x2, y2) format after post-processing
        if len(args) == 1:
            return dict(image=image, target=target)
        else:
            return image, target

    def pre_process(self, batch):
        images, targets = batch

        none_idx_found = False
        for target in targets:
            if target.get('annotations'):
                for annotation in target['annotations']:
                    if isinstance(annotation.get('bbox'), torch.Tensor):
                        annotation['bbox'] = annotation['bbox'].tolist()
            if not target.get('image_id', None):
                none_idx_found = True

        if none_idx_found:  # override None image_id
            for i, target in enumerate(targets):
                target['image_id'] = i

        return self.image_processor(images=images, annotations=targets, return_tensors="pt")

    def post_process(self, batch, target_sizes=None):
        return self.image_processor.post_process_object_detection(batch, target_sizes=target_sizes, threshold=self.confidence_threshold)

    def __getitem__(self, idx):
        return self.transforms(self.dataset[idx], idx=idx)

    def collate_fn(self, batch):
        target_size = random.choice(self.multi_scale) if self.evaluation_mode else self.img_size

        try:
            images = [item['image'] for item in batch]
            targets = [item['target'] for item in batch]
        except TypeError:  # fallback to simple collate
            images = [item[0] for item in batch]
            targets = [item[1] for item in batch]
        self.image_processor.size = {"shortest_edge": target_size, "longest_edge": self.longest_edge}
        return self.pre_process((images, targets))


class RTDetrForObjectDetection(BaseModel, _RTDetrForObjectDetection):
    model_id = "PekingU/rtdetr_r50vd"
    model_name = "RT-DETR-R50"
    model_provider = ModelProvider.HuggingFace
    DataPreparation = RTDetrDataPreparation
    Trainer = RTDetrTrainer
    TrainingArguments = RTDetrTrainingArguments

    class Weights:
        COCO_OFFICIAL = WeightsInfo("PekingU/rtdetr_r50vd")
        SHIFT_CLEAR = WeightsInfo("b-re-w/rtdetr_r50vd_shift_clear")
        CITYSCAPES = WeightsInfo("b-re-w/rtdetr_r50vd_cityscapes")

    def __init__(self, config: RTDetrConfig | None = None, dataset: BaseDataset | str = "", **kwargs):
        if dataset:
            num_classes = len(dataset.classes)
        else:
            num_classes = config.num_labels if config is not None else 80  # default to COCO

        if config is None:
            config = RTDetrConfig.from_pretrained(self.model_id)
        config.num_labels = num_classes  # override num_labels
        super(BaseModel, self).__init__(config=config, **kwargs)  # skip BaseDataset.__init__

        self.num_classes = num_classes

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[list[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):  # NOTE: Method kwargs must be explicitly listed for Trainer to work properly
        return super(BaseModel, self).forward(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

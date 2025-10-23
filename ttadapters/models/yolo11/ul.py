from typing import Callable, Optional
from dataclasses import dataclass
import warnings
import random
import gc
import os
from pathlib import Path
import tempfile
import yaml

import torch
from torch.utils.data import DataLoader
import numpy as np

from ultralytics.nn.tasks import DetectionModel
from torchvision.tv_tensors import BoundingBoxFormat, BoundingBoxes
from torchvision.transforms.v2.functional import convert_bounding_box_format
from torchvision.transforms import v2 as T

from ..base import BaseModel, ModelProvider, WeightsInfo
from ...datasets import BaseDataset, DataPreparation
from ...utils.validator import DetectionEvaluator


@dataclass
class YOLO11TrainingArguments:
    """Training arguments for YOLO11 models following Ultralytics conventions."""
    learning_rate: float = 1e-3
    total_steps: int = 40000
    epochs: int = 100
    eval_period: int = 1000
    save_period: int = 1000
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_workers: int = 0
    momentum: float = 0.937
    weight_decay: float = 5e-4
    lr_scheduler_type: str = "cosine"  # cosine, linear
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    use_amp: bool = False
    output_dir: str = "./results"
    seed: int = 42
    img_size: int = 640

    def __post_init__(self):
        # Calculate epochs from total_steps if needed
        if self.total_steps and not hasattr(self, '_epochs_set'):
            # This will be adjusted by trainer based on dataset size
            pass


class YOLO11Trainer:
    """Trainer for YOLO11 models that wraps Ultralytics training interface."""

    def __init__(
        self,
        model: BaseModel,
        classes: list[str],
        train_dataset: DataPreparation | None = None,
        eval_dataset: DataPreparation | None = None,
        args: YOLO11TrainingArguments | None = None
    ):
        self.model = model
        self.classes = classes
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args if args is not None else YOLO11TrainingArguments()

        # Set up evaluation dataset name
        self.eval_dataset_name = "unknown_dataset"
        if hasattr(self.eval_dataset, "dataset"):
            if hasattr(self.eval_dataset.dataset, "dataset_name"):
                self.eval_dataset_name = self.eval_dataset.dataset.dataset_name
        elif hasattr(self.eval_dataset, "dataset_name"):
            self.eval_dataset_name = self.eval_dataset.dataset_name

    def train(self):
        """Train the model using PyTorch DataLoader interface."""
        if self.train_dataset is None:
            raise ValueError("Training dataset is required")

        torch.cuda.empty_cache()
        gc.collect()

        # Set random seeds
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        # Build dataloaders
        train_loader = self._build_train_loader()
        eval_loader = self._build_eval_loader() if self.eval_dataset else None

        # Calculate epochs from total_steps if needed
        steps_per_epoch = len(train_loader)
        if self.args.total_steps:
            epochs = max(1, int(self.args.total_steps / steps_per_epoch))
        else:
            epochs = self.args.epochs

        # Training loop
        self.model.train()
        device = next(self.model.parameters()).device

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )

        # Scheduler
        if self.args.lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs * steps_per_epoch
            )
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs * steps_per_epoch
            )

        step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                # YOLO11 expects images and targets
                # batch should contain pixel_values and labels
                loss = self._training_step(batch, device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                step += 1

                # Evaluation
                if eval_loader and step % self.args.eval_period == 0:
                    self.test(eval_loader)
                    self.model.train()

                # Save checkpoint
                if step % self.args.save_period == 0:
                    save_path = Path(self.args.output_dir) / f"checkpoint-{step}.pt"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    self.model.save_to(str(save_path.parent), version=f"step{step}")

            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    def _training_step(self, batch, device):
        """Perform a single training step."""
        # Move batch to device
        images = batch['pixel_values'].to(device)
        targets = batch['labels']

        # YOLO11 forward pass
        # DetectionModel expects batch dict with 'img' and 'batch_idx', 'cls', 'bboxes'
        # We need to convert our format to YOLO format
        batch_idx = []
        cls = []
        bboxes = []

        for i, target in enumerate(targets):
            if isinstance(target, dict):
                n_boxes = len(target.get('class_labels', target.get('gt_classes', [])))
                batch_idx.extend([i] * n_boxes)
                cls.extend(target.get('class_labels', target.get('gt_classes', [])).tolist())
                boxes = target.get('boxes', target.get('gt_boxes', []))
                if hasattr(boxes, 'tensor'):
                    boxes = boxes.tensor
                bboxes.extend(boxes.tolist() if isinstance(boxes, torch.Tensor) else boxes)

        # Create YOLO-format batch
        yolo_batch = {
            'img': images,
            'batch_idx': torch.tensor(batch_idx, device=device),
            'cls': torch.tensor(cls, device=device).unsqueeze(1),
            'bboxes': torch.tensor(bboxes, device=device)
        }

        # Forward pass - DetectionModel returns loss dict
        loss_dict = self.model(yolo_batch)

        # Sum all losses
        if isinstance(loss_dict, dict):
            loss = sum(v for k, v in loss_dict.items() if 'loss' in k.lower())
        else:
            loss = loss_dict

        return loss

    def test(self, loader=None):
        """Evaluate the model."""
        if loader is None:
            loader = self._build_eval_loader()

        if loader is None or self.eval_dataset is None:
            print("No evaluation dataset provided")
            return {}

        torch.cuda.empty_cache()
        gc.collect()

        self.model.eval()
        device = next(self.model.parameters()).device

        result = DetectionEvaluator.evaluate(
            model=self.model,
            desc=self.eval_dataset_name,
            loader=loader,
            loader_length=len(loader),
            classes=self.classes,
            data_preparation=self.eval_dataset,
            dtype=next(self.model.parameters()).dtype,
            device=device,
            synchronize=False,
            no_grad=True
        )

        print(f"\nEvaluation Results: {result}")
        return result

    def _build_train_loader(self):
        """Build training dataloader."""
        if self.train_dataset is None:
            return None

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        generator = torch.Generator()
        generator.manual_seed(self.args.seed)

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            worker_init_fn=seed_worker,
            generator=generator
        )

    def _build_eval_loader(self):
        """Build evaluation dataloader."""
        if self.eval_dataset is None:
            return None

        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=self.eval_dataset.collate_fn
        )


class YOLO11DataPreparation(DataPreparation):
    """Data preparation for YOLO11 models."""

    def __init__(
        self,
        dataset: BaseDataset,
        dataset_key: dict = dict(bboxes="boxes2d", classes="boxes2d_classes", original_size="original_hw"),
        img_size: int = 640,
        evaluation_mode: bool = False,
        train_transforms: T.Compose = None,
        valid_transforms: T.Compose = None
    ):
        self.dataset_name = dataset.dataset_name
        self.classes = dataset.classes

        self.dataset = dataset
        self.dataset_key = dataset_key
        self.img_size = img_size
        self.evaluation_mode = evaluation_mode

        # Default transforms for YOLO11
        if train_transforms is None:
            self.train_transforms = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize(size=(img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.train_transforms = train_transforms

        if valid_transforms is None:
            self.valid_transforms = T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Resize(size=(img_size, img_size)),
            ])
        else:
            self.valid_transforms = valid_transforms

        self.pre_process: Callable = lambda batch: batch
        self.post_process: Callable = lambda batch, *args, **kwargs: batch

    def transforms(self, *data):
        """Transform data for YOLO11."""
        image, metadata = data[0] if len(data) == 1 else data

        bboxes = metadata[self.dataset_key['bboxes']]
        bbox_classes = metadata[self.dataset_key['classes']]
        original_height, original_width = metadata[self.dataset_key['original_size']]

        if not isinstance(bboxes, BoundingBoxes):
            warnings.warn("Assume the bbox is in Pascal VOC format (x1, y1, x2, y2) since it's not a BoundingBoxes instance.")
            bboxes = BoundingBoxes(bboxes, format=BoundingBoxFormat.XYXY, canvas_size=(original_height, original_width))

        if bboxes.format != BoundingBoxFormat.XYXY:
            bboxes = convert_bounding_box_format(bboxes, new_format=BoundingBoxFormat.XYXY)

        # Apply transforms
        if not self.evaluation_mode:
            image, bboxes = self.train_transforms(image, bboxes)
        else:
            image, bboxes = self.valid_transforms(image, bboxes)

        # Convert bboxes to normalized YOLO format (x_center, y_center, width, height)
        h, w = image.shape[-2:]
        xyxy = bboxes if isinstance(bboxes, torch.Tensor) else torch.tensor(bboxes)

        # Convert to cxcywh and normalize
        x1, y1, x2, y2 = xyxy.unbind(-1)
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        ww = (x2 - x1) / w
        hh = (y2 - y1) / h
        normalized_boxes = torch.stack([cx, cy, ww, hh], dim=-1)

        return image, {
            'pixel_values': image,
            'class_labels': bbox_classes,
            'boxes': normalized_boxes,
            'orig_size': torch.tensor([original_height, original_width]),
            'size': torch.tensor([h, w])
        }

    def __getitem__(self, idx):
        return self.transforms(self.dataset[idx])

    def collate_fn(self, batch):
        """Collate function for YOLO11."""
        images = torch.stack([item[1]['pixel_values'] for item in batch])
        labels = [item[1] for item in batch]

        return {
            'pixel_values': images,
            'labels': labels
        }


class YOLO11ForObjectDetection(DetectionModel, BaseModel):
    model_name = "YOLO11"
    model_config = "yolo11m.yaml"
    model_provider = ModelProvider.Ultralytics
    channel = 3
    DataPreparation = YOLO11DataPreparation
    Trainer = YOLO11Trainer
    TrainingArguments = YOLO11TrainingArguments

    class Weights:
        COCO = WeightsInfo("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt", weight_key="model")
        SHIFT_CLEAR = WeightsInfo("")

    def __init__(self, dataset: BaseDataset):
        nc = len(dataset.classes)
        super().__init__(self.model_config, ch=self.channel, nc=nc)

        self.dataset_name = dataset.dataset_name
        self.num_classes = nc

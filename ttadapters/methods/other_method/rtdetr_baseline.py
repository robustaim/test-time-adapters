import os
import math
import copy
from os import path
from pathlib import Path
from tqdm.auto import tqdm
import time
import gc
import random
import numpy as np
from typing import Optional, Callable

import os
os.chdir("/workspace/ptta")

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

from ttadapters.datasets import BaseDataset, DatasetHolder, DataLoaderHolder
from ttadapters.datasets import (
    SHIFTDataset,
    SHIFTClearDatasetForObjectDetection,
    SHIFTDiscreteSubsetForObjectDetection
)
from ttadapters import datasets
from ttadapters.methods.other_method import utils

from supervision.metrics.mean_average_precision import MeanAveragePrecision
from supervision.detection.core import Detections
from detectron2.layers import FrozenBatchNorm2d
from detectron2.structures import Boxes, Instances, ImageList
from detectron2.utils.events import EventStorage
from torchvision.tv_tensors import Image, BoundingBoxes

# PIL imports for FixMatch augmentation
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image as PILImage

# RT-DETR specific imports
from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessorFast,
    RTDetrConfig,
)
from transformers.image_utils import AnnotationFormat

from transformers.models.rt_detr.modeling_rt_detr import RTDetrFrozenBatchNorm2d


def task_to_subset_types(task: str):
    """Convert task string to SHIFT dataset subset type"""
    T = SHIFTDiscreteSubsetForObjectDetection.SubsetType

    # weather
    if task == "cloudy":
        return T.CLOUDY_DAYTIME
    if task == "overcast":
        return T.OVERCAST_DAYTIME
    if task == "rainy":
        return T.RAINY_DAYTIME
    if task == "foggy":
        return T.FOGGY_DAYTIME

    # time
    if task == "night":
        return T.CLEAR_NIGHT
    if task in {"dawn", "dawn/dusk"}:
        return T.CLEAR_DAWN
    if task == "clear":
        return T.CLEAR_DAYTIME

    # simple
    if task == "normal":
        return T.NORMAL
    if task == "corrupted":
        return T.CORRUPTED

    raise ValueError(f"Unknown task: {task}")


class SHIFTCorruptedTaskDatasetForObjectDetection(SHIFTDiscreteSubsetForObjectDetection):
    def __init__(
            self, root: str, force_download: bool = False,
            train: bool = True, valid: bool = False,
            transform: Optional[Callable] = None, task: str = "clear", target_transform: Optional[Callable] = None
    ):
        super().__init__(
            root=root, force_download=force_download,
            train=train, valid=valid, subset_type=task_to_subset_types(task),
            transform=transform, target_transform=target_transform
        )


def collate_fn(batch, preprocessor=None):
    images = [item['image'] for item in batch]
    if preprocessor is not None:
        target = [item['target'] for item in batch]
        return preprocessor(images=images, annotations=target, return_tensors="pt")
    else:
        # If no preprocessor is provided, just assume images are already in tensor format
        return dict(
            pixel_values=dict(pixel_values=torch.stack(images)),
            labels=[dict(
                class_labels=item['boxes2d_classes'].long(),
                boxes=item["boxes2d"].float()
            ) for item in batch]
        )

class LabelDataset(BaseDataset):
    def __init__(self, original_dataset, camera='front'):
        self.dataset = original_dataset
        self.camera = camera

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # For SHIFTDiscreteSubsetForObjectDetection, item is a tuple (image_tv, data)
        if isinstance(item, tuple) and len(item) == 2:
            image_tv, data = item
            return data['boxes2d'], data['boxes2d_classes']
        # Fallback for other dataset formats that return dict with camera keys
        elif isinstance(item, dict) and self.camera in item:
            camera_data = item[self.camera]
            return camera_data['boxes2d'], camera_data['boxes2d_classes']
        else:
            raise ValueError(f"Unexpected dataset format. Got {type(item)}")

def naive_collate_fn(batch):
    return batch

class DatasetAdapterForTransformers(BaseDataset):
    def __init__(self, original_dataset, camera='front'):
        self.dataset = original_dataset
        self.camera = camera

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # For SHIFTDiscreteSubsetForObjectDetection, item is a tuple (image_tv, data)
        if isinstance(item, tuple) and len(item) == 2:
            image_tv, data = item
            image = image_tv  # image_tv is already the processed image tensor
        # Fallback for other dataset formats that return dict with camera keys
        elif isinstance(item, dict) and self.camera in item:
            camera_data = item[self.camera]
            image = camera_data['images'].squeeze(0)
            data = camera_data
        else:
            raise ValueError(f"Unexpected dataset format. Got {type(item)}")

        # Convert to COCO_Detection Format
        annotations = []
        target = dict(image_id=idx, annotations=annotations)
        for box, cls in zip(data['boxes2d'], data['boxes2d_classes']):
            x1, y1, x2, y2 = box.tolist()  # from Pascal VOC format (x1, y1, x2, y2)
            width, height = x2 - x1, y2 - y1
            annotations.append(dict(
                bbox=[x1, y1, width, height],  # to COCO format: [x, y, width, height]
                category_id=cls.item(),
                area=width * height,
                iscrowd=0
            ))

        # Following prepare_coco_detection_annotation's expected format
        # RT-DETR ImageProcessor converts the COCO bbox to center format (cx, cy, w, h) during preprocessing
        # But, eventually re-converts the bbox to Pascal VOC (x1, y1, x2, y2) format after post-processing
        return dict(image=image, target=target)


class DirectMethod:
    """
    Direct evaluation method for RT-DETR (no adaptation)
    """

    def __init__(
        self,
        model,
        data_root,
        device=None,
        batch_size=4,
        image_size=800,
        model_states=None,
        reference_model_id="PekingU/rtdetr_r50vd",
        class_num=6
    ):
        self.model = model
        self.data_root = data_root
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.image_size = image_size
        self.model_states = model_states
        self.reference_model_id = reference_model_id
        self.class_num = class_num

        # Setup image processor
        self.preprocessor = RTDetrImageProcessorFast.from_pretrained(self.reference_model_id)
        self.preprocessor.format = AnnotationFormat.COCO_DETECTION
        self.preprocessor.do_resize = False
        self.preprocessor.size = {"height": self.image_size, "width": self.image_size}

    @classmethod
    def load(cls, model, data_root, **kwargs):
        """
        Factory method to create and initialize DirectMethod
        """
        instance = cls(model=model, data_root=data_root, **kwargs)
        instance._setup()
        return instance

    def _setup(self):
        """Setup the DirectMethod"""
        self.model.to(self.device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def adapt_single_batch(self, batch):
        """
        Process a single batch without adaptation

        Args:
            batch: Input batch from dataloader

        Returns:
            outputs: Model outputs
        """
        self.model.eval()
        with torch.no_grad():
            pixel_values = batch['pixel_values'].to(self.device, non_blocking=True)
            outputs = self.model(pixel_values=pixel_values)
        return outputs

    def evaluate_task(self, task=None, dataset=None, raw_data=None, valid_dataloader=None, threshold=0.0):
        """
        Evaluate on a specific task using baseline.py approach
        """
        if raw_data is None or valid_dataloader is None:
            raise ValueError("Both raw_data and valid_dataloader must be provided")

        task_name = task if task is not None else "unknown"
        print(f"Starting Direct Method evaluation on {task_name}")

        # Get classes info from dataset if provided
        if dataset is not None:
            CLASSES = dataset
        else:
            CLASSES = None

        # Use the same evaluator as baseline.py
        # Use local class definitions instead of utils import
        # Import utils for Evaluator only
        from ttadapters.methods.other_method import utils
        evaluator = utils.Evaluator(class_list=CLASSES, task=task_name, reference_preprocessor=self.preprocessor)

        # Performance measurement
        torch.cuda.empty_cache()
        gc.collect()

        inference_time = 0
        batch_count = 0

        # Process each batch following baseline.py approach exactly
        for batch_i, labels, input in zip(tqdm(range(len(raw_data))), raw_data, valid_dataloader):
            img = input['pixel_values'].to(self.device, non_blocking=True)

            # Start timing
            start_time = time.time()

            with torch.no_grad():
                outputs = self.model(img)

            # End timing
            inference_time += time.time() - start_time

            evaluator.add(outputs, labels)
            batch_count += 1

        result = evaluator.compute()

        # Calculate FPS
        fps = batch_count / inference_time if inference_time > 0 else 0

        print(f"\nTotal Images: {batch_count}")
        print(f"Inference Time: {inference_time:.4f}s")
        print(f"FPS: {fps:.2f}")

        result.update({
            "fps": fps,
            "total_images": batch_count,
            "inference_time": inference_time
        })

        return result

    def prepare_dataloaders(self, tasks=None):
        """
        Prepare dataloaders for all tasks in advance (following baseline.py approach)
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = {}
        print("Preparing dataloaders for all tasks...")

        # Use local class definitions instead of utils import
        from functools import partial

        for task in tasks:
            print(f"Loading dataset for {task}...")

            # Create dataset using utils (same as baseline.py)
            shift_dataset = SHIFTCorruptedTaskDatasetForObjectDetection(
                root=self.data_root, valid=True, task=task
            )

            # Create raw_data dataloader for labels (GT) - use SHIFT dataset directly
            raw_data = DataLoader(
                LabelDataset(shift_dataset),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=naive_collate_fn
            )

            # Create valid_dataloader for preprocessed inputs - use SHIFT dataset directly
            valid_dataloader = DataLoader(
                DatasetAdapterForTransformers(shift_dataset),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=partial(collate_fn, preprocessor=self.preprocessor)
            )

            # Keep shift_dataset as the main dataset reference
            dataset = shift_dataset

            dataloaders[task] = (dataset, raw_data, valid_dataloader)
            print(f"  - {task}: {len(dataset)} samples")

        print(f"All {len(dataloaders)} dataloaders prepared!")
        return dataloaders


    def evaluate_all_tasks(self, tasks=None):
        """
        Evaluate on all tasks (automatically prepares dataloaders and runs evaluation)
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        # Automatically prepare dataloaders
        dataloaders = self.prepare_dataloaders(tasks)

        results = {}
        for task in tasks:
            if task not in dataloaders:
                print(f"Warning: No dataloader found for task {task}, skipping...")
                continue

            dataset, raw_data, valid_dataloader = dataloaders[task]
            results[task] = self.evaluate_task(
                task=task,
                dataset=dataset,
                raw_data=raw_data,
                valid_dataloader=valid_dataloader
            )
            print(f"Results for {task}: {results[task]}")

        return results


class ActMAD:
    """
    ActMAD (Activation Mean Alignment and Discrepancy) method for RT-DETR
    """

    def __init__(
        self,
        model,
        data_root,
        device=None,
        batch_size=4,
        learning_rate=0.0001,
        clean_bn_extract_batch=8,
        image_size=800,
        model_states=None,
        reference_model_id="PekingU/rtdetr_r50vd",
        class_num=6,
        adaptation_layers="backbone+encoder"  # "backbone", "encoder", "backbone+encoder"
    ):
        self.model = model
        self.data_root = data_root
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.clean_bn_extract_batch = clean_bn_extract_batch
        self.image_size = image_size
        self.model_states = model_states
        self.reference_model_id = reference_model_id
        self.class_num = class_num
        self.adaptation_layers = adaptation_layers

        # Statistics storage
        self.clean_mean_list_final = None
        self.clean_var_list_final = None
        self.layer_names = None
        self.chosen_bn_layers = None

        # Optimizer
        self.optimizer = None
        self.l1_loss = nn.L1Loss(reduction="mean")

        # Statistics save path
        self.stats_save_path = Path("/workspace/ptta/ttadapters/methods/other_method") / f"actmad_clean_statistics_rtdetr_{self.adaptation_layers.replace('+', '_')}.pt"

        # Setup image processor
        self.preprocessor = RTDetrImageProcessorFast.from_pretrained(self.reference_model_id)
        self.preprocessor.format = AnnotationFormat.COCO_DETECTION
        self.preprocessor.do_resize = False
        self.preprocessor.size = {"height": self.image_size, "width": self.image_size}

    @classmethod
    def load(cls, model, data_root, **kwargs):
        """
        Factory method to create and initialize ActMAD
        """
        instance = cls(model=model, data_root=data_root, **kwargs)
        instance._setup()
        return instance

    def _setup(self):
        """Setup the ActMAD method"""
        # Move model to device
        self.model.to(self.device)

        # Unfreeze model parameters
        for k, v in self.model.named_parameters():
            v.requires_grad = True

        # Setup optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            # momentum=0.95,
            # weight_decay=5e-4,
            # nesterov=True
        )

        # Extract or load clean statistics
        self._extract_or_load_clean_statistics()

    def extract_activation_alignment(self, method, data_root, batch_size=16):
        """Extract activation alignment from clean data for RT-DETR"""
        from ttadapters.datasets import SHIFTClearDatasetForObjectDetection

        shift_dataset = SHIFTClearDatasetForObjectDetection(root=data_root, train=True)

        # Wrap with DatasetAdapterForTransformers to ensure dict format
        dataset = DatasetAdapterForTransformers(shift_dataset)
        
        from functools import partial
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn, preprocessor=self.preprocessor))
        loader.train_len = math.ceil(len(dataset)/batch_size)

        # model unfreeze
        for k, v in self.model.named_parameters():
            v.requires_grad = True

        chosen_bn_info = []
        if method == "actmad":
            for name, m in self.model.named_modules():
                # RT-DETR uses RTDetrFrozenBatchNorm2d in backbone, BatchNorm2d in encoder, LayerNorm in transformer layers
                if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, RTDetrFrozenBatchNorm2d)):
                    # Filter based on adaptation_layers setting
                    if self.adaptation_layers == "backbone":
                        # Backbone: RTDetrResNetBackbone with RTDetrFrozenBatchNorm2d
                        if ('model.backbone' in name and 'RTDetrFrozenBatchNorm2d' in str(type(m))) or \
                           ('backbone' in name.lower() and isinstance(m, RTDetrFrozenBatchNorm2d)):
                            chosen_bn_info.append((name, m))
                    elif self.adaptation_layers == "encoder":
                        # Encoder: encoder_input_proj, lateral_convs, fpn_blocks, downsample_convs, pan_blocks (BatchNorm2d)
                        # and encoder.layers (LayerNorm)
                        if ('encoder' in name.lower() and not 'decoder' in name.lower()) and \
                           (isinstance(m, (nn.BatchNorm2d, nn.LayerNorm))):
                            chosen_bn_info.append((name, m))
                    elif self.adaptation_layers == "backbone+encoder":
                        # Both backbone and encoder, exclude decoder
                        if 'decoder' not in name.lower():
                            chosen_bn_info.append((name, m))
                    else:
                        # Default behavior: exclude decoder
                        if 'decoder' not in name.lower():
                            chosen_bn_info.append((name, m))

        cutoff = len(chosen_bn_info) // 2
        chosen_bn_info = chosen_bn_info[cutoff:]
        chosen_bn_layers = [module for name, module in chosen_bn_info]
        layer_names = [name for name, module in chosen_bn_info]

        n_chosen_layers = len(chosen_bn_layers)
        print(f"ActMAD: Using {n_chosen_layers} normalization layers")

        save_outputs = [utils.SaveOutputRTDETR() for _ in range(n_chosen_layers)]

        clean_mean_act_list = [utils.AverageMeterRTDETR() for _ in range(n_chosen_layers)]
        clean_var_act_list = [utils.AverageMeterRTDETR() for _ in range(n_chosen_layers)]

        clean_mean_list_final = []
        clean_var_list_final = []

        # extract the activation alignment in train dataset
        print("Start extracting BN statistics from the training dataset")

        with torch.no_grad():
            for batch in tqdm(loader, total=loader.train_len, desc="Evaluation"):
                self.model.eval()

                # Get pixel_values directly from batch (same as other methods)
                pixel_values = batch['pixel_values'].to(self.device)

                hook_list = [chosen_bn_layers[i].register_forward_hook(save_outputs[i]) for i in range(n_chosen_layers)]
                _ = self.model(pixel_values=pixel_values)

                for i in range(n_chosen_layers):
                    clean_mean_act_list[i].update(save_outputs[i].get_out_mean())
                    clean_var_act_list[i].update(save_outputs[i].get_out_var())

                    save_outputs[i].clear()
                    hook_list[i].remove()

            for i in range(n_chosen_layers):
                clean_mean_list_final.append(clean_mean_act_list[i].avg)
                clean_var_list_final.append(clean_var_act_list[i].avg)

            return clean_mean_list_final, clean_var_list_final, layer_names

    def _extract_or_load_clean_statistics(self):
        """Extract or load clean statistics"""
        if self.stats_save_path.exists():
            print(f"Loading saved ActMAD statistics from {self.stats_save_path}")
            saved_stats = torch.load(self.stats_save_path)
            self.clean_mean_list_final = saved_stats["clean_mean_list_final"]
            self.clean_var_list_final = saved_stats["clean_var_list_final"]
            self.layer_names = saved_stats["layer_names"]
        else:
            print("Extracting ActMAD statistics from clean data...")
            (
                self.clean_mean_list_final,
                self.clean_var_list_final,
                self.layer_names
            ) = self.extract_activation_alignment(
                method="actmad",
                data_root=self.data_root,
                batch_size=self.clean_bn_extract_batch
            )

            print(f"Saving ActMAD statistics to {self.stats_save_path}")
            self.stats_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "clean_mean_list_final": self.clean_mean_list_final,
                "clean_var_list_final": self.clean_var_list_final,
                "layer_names": self.layer_names
            }, self.stats_save_path)

        # Setup chosen BN layers
        self._setup_chosen_bn_layers()

    def _setup_chosen_bn_layers(self):
        """Setup chosen BN layers from layer names"""
        current_bn_dict = {
            name: module for name, module in self.model.named_modules()
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, RTDetrFrozenBatchNorm2d))
        }

        self.chosen_bn_layers = []
        for layer_name in self.layer_names:
            if layer_name in current_bn_dict:
                self.chosen_bn_layers.append(current_bn_dict[layer_name])
            else:
                print(f"Warning: Layer {layer_name} not found!")

    def adapt_single_batch(self, batch):
        """
        Adapt the model on a single batch using ActMAD for RT-DETR
        """
        # Ensure model parameters are unfrozen
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.train()
        # Keep normalization layers in eval mode
        for m in self.model.modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                m.eval()

        self.optimizer.zero_grad()

        n_chosen_layers = len(self.chosen_bn_layers)
        save_outputs_tta = [utils.SaveOutputRTDETR() for _ in range(n_chosen_layers)]

        hook_list_tta = [
            self.chosen_bn_layers[x].register_forward_hook(save_outputs_tta[x])
            for x in range(n_chosen_layers)
        ]

        # Forward pass
        pixel_values = batch['pixel_values'].to(self.device, non_blocking=True)
        outputs = self.model(pixel_values=pixel_values)

        # Extract current batch statistics
        batch_mean_tta = [save_outputs_tta[x].get_out_mean() for x in range(n_chosen_layers)]
        batch_var_tta = [save_outputs_tta[x].get_out_var() for x in range(n_chosen_layers)]

        # Compute ActMAD loss
        loss_mean = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(self.device)
        loss_var = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(self.device)

        for i in range(n_chosen_layers):
            loss_mean += self.l1_loss(
                batch_mean_tta[i].to(self.device),
                self.clean_mean_list_final[i].to(self.device)
            )
            loss_var += self.l1_loss(
                batch_var_tta[i].to(self.device),
                self.clean_var_list_final[i].to(self.device)
            )

        loss = loss_mean + loss_var

        # Backward and update
        loss.backward()

        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

        self.optimizer.step()

        # Clean up hooks
        for z in range(n_chosen_layers):
            save_outputs_tta[z].clear()
            hook_list_tta[z].remove()

        return outputs

    def evaluate_task(self, task=None, dataset=None, raw_data=None, valid_dataloader=None, threshold=0.0):
        """
        Evaluate on a specific task
        """
        if raw_data is None or valid_dataloader is None:
            raise ValueError("Both raw_data and valid_dataloader must be provided")

        task_name = task if task is not None else "unknown"
        print(f"Starting ActMAD evaluation on {task_name}")

        # Get classes info from dataset if provided
        if dataset is not None:
            CLASSES = dataset
        else:
            CLASSES = None

        # Use the same evaluator as baseline.py
        # from ttladapters.methods.other_method import utils
        evaluator = utils.Evaluator(class_list=CLASSES, task=task_name, reference_preprocessor=self.preprocessor)

        # Performance measurement
        torch.cuda.empty_cache()
        gc.collect()

        inference_time = 0
        batch_count = 0

        # Process each batch following baseline.py approach exactly
        for batch_i, labels, input in zip(tqdm(range(len(raw_data))), raw_data, valid_dataloader):
            img = input['pixel_values'].to(self.device, non_blocking=True)

            # Start timing
            start_time = time.time()

            # ActMAD adaptation
            outputs = self.adapt_single_batch(input)

            # End timing
            inference_time += time.time() - start_time

            evaluator.add(outputs, labels)
            batch_count += 1

        # Get results from evaluator
        results = evaluator.compute()

        # Add performance metrics
        results['inference_time'] = inference_time
        results['fps'] = batch_count / inference_time if inference_time > 0 else 0
        results['total_samples'] = batch_count

        print(f"ActMAD evaluation completed on {task_name}")
        return results

    def prepare_dataloaders(self, tasks=None):
        """
        Prepare dataloaders for all tasks in advance (following DirectMethod approach)
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = {}
        print("Preparing dataloaders for all tasks...")

        # Use local class definitions instead of utils import
        from functools import partial

        for task in tasks:
            print(f"Loading dataset for {task}...")

            # Create dataset using utils (same as DirectMethod)
            shift_dataset = SHIFTCorruptedTaskDatasetForObjectDetection(
                root=self.data_root, valid=True, task=task
            )

            # Create raw_data dataloader for labels (GT) - use SHIFT dataset directly
            raw_data = DataLoader(
                LabelDataset(shift_dataset),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=naive_collate_fn
            )

            # Create valid_dataloader for preprocessed inputs - use SHIFT dataset directly
            valid_dataloader = DataLoader(
                DatasetAdapterForTransformers(shift_dataset),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=partial(collate_fn, preprocessor=self.preprocessor)
            )

            # Keep shift_dataset as the main dataset reference
            dataset = shift_dataset

            dataloaders[task] = (dataset, raw_data, valid_dataloader)
            print(f"  - {task}: {len(dataset)} samples")

        print(f"All {len(dataloaders)} dataloaders prepared!")
        return dataloaders


    def evaluate_all_tasks(self, tasks=None):
        """
        Evaluate on all tasks (automatically prepares dataloaders and runs evaluation)
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        # Automatically prepare dataloaders
        dataloaders = self.prepare_dataloaders(tasks)

        results = {}
        for task in tasks:
            if task not in dataloaders:
                print(f"Warning: No dataloader found for task {task}, skipping...")
                continue

            dataset, raw_data, valid_dataloader = dataloaders[task]
            results[task] = self.evaluate_task(
                task=task,
                dataset=dataset,
                raw_data=raw_data,
                valid_dataloader=valid_dataloader
            )
            print(f"Results for {task}: {results[task]}")

        return results


class DUA:
    """
    DUA (Dynamic Update Adaptation) method for RT-DETR
    """

    def __init__(
        self,
        model,
        data_root,
        device=None,
        batch_size=4,
        decay_factor=0.94,
        mom_pre=0.01,
        min_momentum_constant=0.0001,
        image_size=800,
        model_states=None,
        reference_model_id="PekingU/rtdetr_r50vd",
        class_num=6,
        adaptation_layers="backbone+encoder"  # "backbone", "encoder", "backbone+encoder"
    ):
        self.model = model
        self.data_root = data_root
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.decay_factor = decay_factor
        self.mom_pre = mom_pre
        self.min_momentum_constant = min_momentum_constant
        self.image_size = image_size
        self.model_states = model_states
        self.reference_model_id = reference_model_id
        self.class_num = class_num
        self.adaptation_layers = adaptation_layers

        # Setup image processor
        self.preprocessor = RTDetrImageProcessorFast.from_pretrained(self.reference_model_id)
        self.preprocessor.format = AnnotationFormat.COCO_DETECTION
        self.preprocessor.do_resize = False
        self.preprocessor.size = {"height": self.image_size, "width": self.image_size}

    @classmethod
    def load(cls, model, data_root, **kwargs):
        """
        Factory method to create and initialize DUA
        """
        instance = cls(model=model, data_root=data_root, **kwargs)
        instance._setup()
        return instance

    def _setup(self):
        """Setup the DUA method"""
        self.model.to(self.device)
        self._apply_dua_adaptation()

    def _apply_dua_adaptation(self):
        """Apply DUA adaptation to BatchNorm layers in RT-DETR"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, RTDetrFrozenBatchNorm2d)):
                # Filter based on adaptation_layers setting
                should_adapt = False
                if self.adaptation_layers == "backbone":
                    # Backbone: RTDetrResNetBackbone with RTDetrFrozenBatchNorm2d
                    if ('model.backbone' in name and isinstance(module, RTDetrFrozenBatchNorm2d)) or \
                       ('backbone' in name.lower() and isinstance(module, RTDetrFrozenBatchNorm2d)):
                        should_adapt = True
                elif self.adaptation_layers == "encoder":
                    # Encoder: encoder_input_proj, lateral_convs, fpn_blocks, etc. (BatchNorm2d, LayerNorm)
                    if ('encoder' in name.lower() and not 'decoder' in name.lower()) and \
                       isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                        should_adapt = True
                elif self.adaptation_layers == "backbone+encoder":
                    # Both backbone and encoder, exclude decoder
                    if 'decoder' not in name.lower():
                        should_adapt = True
                else:
                    # Default behavior: exclude decoder
                    if 'decoder' not in name.lower():
                        should_adapt = True

                if not should_adapt:
                    continue

                module.adapt_type = "DUA"
                module.min_momentum_constant = self.min_momentum_constant
                module.decay_factor = self.decay_factor
                module.mom_pre = self.mom_pre

                if not hasattr(module, 'original_running_mean') and hasattr(module, 'running_mean'):
                    module.original_running_mean = module.running_mean.clone()
                    module.original_running_var = module.running_var.clone()

                def dua_forward(self, x):
                    if hasattr(self, 'adapt_type') and self.adapt_type == "DUA":
                        with torch.no_grad():
                            current_momentum = self.mom_pre + self.min_momentum_constant

                            if isinstance(self, (nn.BatchNorm2d, RTDetrFrozenBatchNorm2d)):
                                batch_mean = x.mean(dim=[0, 2, 3])
                                batch_var = x.var(dim=[0, 2, 3], unbiased=True)
                            elif isinstance(self, nn.LayerNorm):
                                # For LayerNorm, compute statistics over the last dimension(s)
                                normalized_shape = self.normalized_shape
                                dims = tuple(range(-len(normalized_shape), 0))
                                batch_mean = x.mean(dim=dims, keepdim=True)
                                batch_var = x.var(dim=dims, keepdim=True, unbiased=True)

                            if hasattr(self, 'running_mean'):
                                self.running_mean.mul_(1 - current_momentum).add_(batch_mean, alpha=current_momentum)
                                self.running_var.mul_(1 - current_momentum).add_(batch_var, alpha=current_momentum)

                            self.mom_pre *= self.decay_factor

                    # Standard normalization
                    if isinstance(self, nn.BatchNorm2d):
                        if hasattr(self, 'adapt_type') and self.adapt_type == "DUA":
                            scale = self.weight * (self.running_var + self.eps).rsqrt()
                            bias = self.bias - self.running_mean * scale
                            scale = scale.reshape(1, -1, 1, 1)
                            bias = bias.reshape(1, -1, 1, 1)
                        else:
                            scale = self.weight * (self.running_var + self.eps).rsqrt()
                            bias = self.bias - self.running_mean * scale
                            scale = scale.reshape(1, -1, 1, 1)
                            bias = bias.reshape(1, -1, 1, 1)

                        out_dtype = x.dtype
                        out = x * scale.to(out_dtype) + bias.to(out_dtype)

                    elif isinstance(self, RTDetrFrozenBatchNorm2d):
                        # RTDetrFrozenBatchNorm2d has different structure - use eps=1e-5 as default
                        eps = getattr(self, 'eps', 1e-5)
                        if hasattr(self, 'adapt_type') and self.adapt_type == "DUA":
                            scale = self.weight * (self.running_var + eps).rsqrt()
                            bias = self.bias - self.running_mean * scale
                            scale = scale.reshape(1, -1, 1, 1)
                            bias = bias.reshape(1, -1, 1, 1)
                        else:
                            scale = self.weight * (self.running_var + eps).rsqrt()
                            bias = self.bias - self.running_mean * scale
                            scale = scale.reshape(1, -1, 1, 1)
                            bias = bias.reshape(1, -1, 1, 1)

                        out_dtype = x.dtype
                        out = x * scale.to(out_dtype) + bias.to(out_dtype)

                    elif isinstance(self, nn.LayerNorm):
                        # Standard LayerNorm operation
                        out = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

                    else:
                        # Fallback: just return input
                        out = x

                    return out

                module.forward = dua_forward.__get__(module, module.__class__)

    def reset_dua_momentum(self, mom_pre=0.01):
        """Reset DUA momentum for new task"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, RTDetrFrozenBatchNorm2d)) and hasattr(module, 'adapt_type'):
                if module.adapt_type == "DUA":
                    module.mom_pre = mom_pre
                    if hasattr(module, 'original_running_mean'):
                        module.running_mean = module.original_running_mean.clone()
                        module.running_var = module.original_running_var.clone()

    def adapt_single_batch(self, batch):
        """
        Adapt the model on a single batch using DUA for RT-DETR
        """
        self.model.eval()
        pixel_values = batch['pixel_values'].to(self.device, non_blocking=True)
        outputs = self.model(pixel_values=pixel_values)
        return outputs

    def evaluate_task(self, task=None, dataset=None, raw_data=None, valid_dataloader=None, threshold=0.0):
        """
        Evaluate on a specific task
        """
        if raw_data is None or valid_dataloader is None:
            raise ValueError("Both raw_data and valid_dataloader must be provided")

        task_name = task if task is not None else "unknown"
        print(f"Starting DUA evaluation on {task_name}")

        if dataset is not None:
            CLASSES = dataset
        else:
            CLASSES = None

        map_metric = MeanAveragePrecision()
        predictions_list = []
        targets_list = []

        torch.cuda.empty_cache()
        gc.collect()

        inference_time = 0
        # Use the same evaluator as DirectMethod
        # Use local class definitions instead of utils import
        CLASSES = dataset.classes if hasattr(dataset, 'classes') else CLASSES
        # Import utils for Evaluator only
        from ttadapters.methods.other_method import utils
        evaluator = utils.Evaluator(class_list=CLASSES, task=task_name, reference_preprocessor=self.preprocessor)

        batch_count = 0

        # Process each batch following baseline.py approach exactly
        for batch_i, labels, input in zip(tqdm(range(len(raw_data))), raw_data, valid_dataloader):
            img = input['pixel_values'].to(self.device, non_blocking=True)

            # Start timing
            start_time = time.time()

            # Method-specific adaptation
            outputs = self.adapt_single_batch(input)

            # End timing
            inference_time += time.time() - start_time

            evaluator.add(outputs, labels)
            batch_count += 1

        # Get results from evaluator
        results = evaluator.compute()

        # Add performance metrics
        results['inference_time'] = inference_time
        results['fps'] = batch_count / inference_time if inference_time > 0 else 0
        results['total_samples'] = batch_count

        print(f"Evaluation completed on {task_name}")
        return results

    def prepare_dataloaders(self, tasks=None):
        """
        Prepare dataloaders for all tasks in advance (following DirectMethod approach)
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = {}
        print("Preparing dataloaders for all tasks...")

        # Use local class definitions instead of utils import
        from functools import partial

        for task in tasks:
            print(f"Loading dataset for {task}...")

            # Create dataset using utils (same as DirectMethod)
            shift_dataset = SHIFTCorruptedTaskDatasetForObjectDetection(
                root=self.data_root, valid=True, task=task
            )

            # Create raw_data dataloader for labels (GT) - use SHIFT dataset directly
            raw_data = DataLoader(
                LabelDataset(shift_dataset),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=naive_collate_fn
            )

            # Create valid_dataloader for preprocessed inputs - use SHIFT dataset directly
            valid_dataloader = DataLoader(
                DatasetAdapterForTransformers(shift_dataset),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=partial(collate_fn, preprocessor=self.preprocessor)
            )

            # Keep shift_dataset as the main dataset reference
            dataset = shift_dataset

            dataloaders[task] = (dataset, raw_data, valid_dataloader)
            print(f"  - {task}: {len(dataset)} samples")

        print(f"All {len(dataloaders)} dataloaders prepared!")
        return dataloaders


    def evaluate_all_tasks(self, tasks=None):
        """
        Evaluate on all tasks (automatically prepares dataloaders and runs evaluation)
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = self.prepare_dataloaders(tasks)

        results = {}
        for task in tasks:
            if task not in dataloaders:
                print(f"Warning: No dataloader found for task {task}, skipping...")
                continue

            dataset, raw_data, valid_dataloader = dataloaders[task]
            results[task] = self.evaluate_task(
                task=task,
                dataset=dataset,
                raw_data=raw_data,
                valid_dataloader=valid_dataloader
            )
            print(f"Results for {task}: {results[task]}")

        return results


class NORM:
    """
    NORM (Normalization) method for RT-DETR
    """

    def __init__(
        self,
        model,
        data_root,
        device=None,
        batch_size=4,
        source_sum=128,
        image_size=800,
        model_states=None,
        reference_model_id="PekingU/rtdetr_r50vd",
        class_num=6,
        adaptation_layers="backbone+encoder"  # "backbone", "encoder", "backbone+encoder"
    ):
        self.model = model
        self.data_root = data_root
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.source_sum = source_sum
        self.image_size = image_size
        self.model_states = model_states
        self.reference_model_id = reference_model_id
        self.class_num = class_num
        self.adaptation_layers = adaptation_layers

        # Setup image processor
        self.preprocessor = RTDetrImageProcessorFast.from_pretrained(self.reference_model_id)
        self.preprocessor.format = AnnotationFormat.COCO_DETECTION
        self.preprocessor.do_resize = False
        self.preprocessor.size = {"height": self.image_size, "width": self.image_size}

    @classmethod
    def load(cls, model, data_root, **kwargs):
        """
        Factory method to create and initialize NORM
        """
        instance = cls(model=model, data_root=data_root, **kwargs)
        instance._setup()
        return instance

    def _setup(self):
        """Setup the NORM method"""
        self.model.to(self.device)
        self._apply_norm_adaptation()

    def _apply_norm_adaptation(self):
        """Apply NORM adaptation to BatchNorm layers in RT-DETR"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, RTDetrFrozenBatchNorm2d)):
                # Filter based on adaptation_layers setting
                should_adapt = False
                if self.adaptation_layers == "backbone":
                    # Backbone: RTDetrResNetBackbone with RTDetrFrozenBatchNorm2d
                    if ('model.backbone' in name and isinstance(module, RTDetrFrozenBatchNorm2d)) or \
                       ('backbone' in name.lower() and isinstance(module, RTDetrFrozenBatchNorm2d)):
                        should_adapt = True
                elif self.adaptation_layers == "encoder":
                    # Encoder: encoder_input_proj, lateral_convs, fpn_blocks, etc. (BatchNorm2d, LayerNorm)
                    if ('encoder' in name.lower() and not 'decoder' in name.lower()) and \
                       isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                        should_adapt = True
                elif self.adaptation_layers == "backbone+encoder":
                    # Both backbone and encoder, exclude decoder
                    if 'decoder' not in name.lower():
                        should_adapt = True
                else:
                    # Default behavior: exclude decoder
                    if 'decoder' not in name.lower():
                        should_adapt = True

                if not should_adapt:
                    continue

                module.adapt_type = "NORM"
                module.source_sum = self.source_sum

                def norm_forward(self, x):
                    if hasattr(self, 'adapt_type') and self.adapt_type == "NORM":
                        alpha = x.shape[0] / (self.source_sum + x.shape[0])

                        if isinstance(self, nn.BatchNorm2d):
                            running_mean = (1 - alpha) * self.running_mean + alpha * x.mean(dim=[0,2,3])
                            running_var = (1 - alpha) * self.running_var + alpha * x.var(dim=[0,2,3])
                            scale = self.weight * (running_var + self.eps).rsqrt()
                            bias = self.bias - running_mean * scale
                            scale = scale.reshape(1, -1, 1, 1)
                            bias = bias.reshape(1, -1, 1, 1)
                        elif isinstance(self, RTDetrFrozenBatchNorm2d):
                            # RTDetrFrozenBatchNorm2d has different structure - use eps=1e-5 as default
                            eps = getattr(self, 'eps', 1e-5)
                            running_mean = (1 - alpha) * self.running_mean + alpha * x.mean(dim=[0,2,3])
                            running_var = (1 - alpha) * self.running_var + alpha * x.var(dim=[0,2,3])
                            scale = self.weight * (running_var + eps).rsqrt()
                            bias = self.bias - running_mean * scale
                            scale = scale.reshape(1, -1, 1, 1)
                            bias = bias.reshape(1, -1, 1, 1)
                        elif isinstance(self, nn.LayerNorm):
                            # For LayerNorm, use standard operation
                            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

                    else:
                        if isinstance(self, nn.BatchNorm2d):
                            scale = self.weight * (self.running_var + self.eps).rsqrt()
                            bias = self.bias - self.running_mean * scale
                            scale = scale.reshape(1, -1, 1, 1)
                            bias = bias.reshape(1, -1, 1, 1)
                        elif isinstance(self, RTDetrFrozenBatchNorm2d):
                            # RTDetrFrozenBatchNorm2d has different structure - use eps=1e-5 as default
                            eps = getattr(self, 'eps', 1e-5)
                            scale = self.weight * (self.running_var + eps).rsqrt()
                            bias = self.bias - self.running_mean * scale
                            scale = scale.reshape(1, -1, 1, 1)
                            bias = bias.reshape(1, -1, 1, 1)
                        elif isinstance(self, nn.LayerNorm):
                            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

                    out_dtype = x.dtype
                    out = x * scale.to(out_dtype) + bias.to(out_dtype)
                    return out

                module.forward = norm_forward.__get__(module, module.__class__)

    def adapt_single_batch(self, batch):
        """
        Adapt the model on a single batch using NORM for RT-DETR
        """
        self.model.eval()
        pixel_values = batch['pixel_values'].to(self.device, non_blocking=True)
        outputs = self.model(pixel_values=pixel_values)
        return outputs

    def evaluate_task(self, task=None, dataset=None, raw_data=None, valid_dataloader=None, threshold=0.0):
        """
        Evaluate on a specific task
        """
        if raw_data is None or valid_dataloader is None:
            raise ValueError("Both raw_data and valid_dataloader must be provided")

        task_name = task if task is not None else "unknown"
        print(f"Starting NORM evaluation on {task_name}")

        if dataset is not None:
            CLASSES = dataset
        else:
            CLASSES = None

        map_metric = MeanAveragePrecision()
        predictions_list = []
        targets_list = []

        torch.cuda.empty_cache()
        gc.collect()

        inference_time = 0
        # Use the same evaluator as DirectMethod
        # Use local class definitions instead of utils import
        CLASSES = dataset.classes if hasattr(dataset, 'classes') else CLASSES
        # Import utils for Evaluator only
        from ttadapters.methods.other_method import utils
        evaluator = utils.Evaluator(class_list=CLASSES, task=task_name, reference_preprocessor=self.preprocessor)

        batch_count = 0

        # Process each batch following baseline.py approach exactly
        for batch_i, labels, input in zip(tqdm(range(len(raw_data))), raw_data, valid_dataloader):
            img = input['pixel_values'].to(self.device, non_blocking=True)

            # Start timing
            start_time = time.time()

            # Method-specific adaptation
            outputs = self.adapt_single_batch(input)

            # End timing
            inference_time += time.time() - start_time

            evaluator.add(outputs, labels)
            batch_count += 1

        # Get results from evaluator
        results = evaluator.compute()

        # Add performance metrics
        results['inference_time'] = inference_time
        results['fps'] = batch_count / inference_time if inference_time > 0 else 0
        results['total_samples'] = batch_count

        print(f"Evaluation completed on {task_name}")
        return results

    def prepare_dataloaders(self, tasks=None):
        """
        Prepare dataloaders for all tasks in advance (following DirectMethod approach)
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = {}
        print("Preparing dataloaders for all tasks...")

        # Use local class definitions instead of utils import
        from functools import partial

        for task in tasks:
            print(f"Loading dataset for {task}...")

            # Create dataset using utils (same as DirectMethod)
            shift_dataset = SHIFTCorruptedTaskDatasetForObjectDetection(
                root=self.data_root, valid=True, task=task
            )

            # Create raw_data dataloader for labels (GT) - use SHIFT dataset directly
            raw_data = DataLoader(
                LabelDataset(shift_dataset),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=naive_collate_fn
            )

            # Create valid_dataloader for preprocessed inputs - use SHIFT dataset directly
            valid_dataloader = DataLoader(
                DatasetAdapterForTransformers(shift_dataset),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=partial(collate_fn, preprocessor=self.preprocessor)
            )

            # Keep shift_dataset as the main dataset reference
            dataset = shift_dataset

            dataloaders[task] = (dataset, raw_data, valid_dataloader)
            print(f"  - {task}: {len(dataset)} samples")

        print(f"All {len(dataloaders)} dataloaders prepared!")
        return dataloaders


    def evaluate_all_tasks(self, tasks=None):
        """
        Evaluate on all tasks (automatically prepares dataloaders and runs evaluation)
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = self.prepare_dataloaders(tasks)

        results = {}
        for task in tasks:
            if task not in dataloaders:
                print(f"Warning: No dataloader found for task {task}, skipping...")
                continue

            dataset, raw_data, valid_dataloader = dataloaders[task]
            results[task] = self.evaluate_task(
                task=task,
                dataset=dataset,
                raw_data=raw_data,
                valid_dataloader=valid_dataloader
            )
            print(f"Results for {task}: {results[task]}")

        return results


# Usage examples for different adaptation layer settings:
#
# RT-DETR Model Structure:
# - Backbone: RTDetrResNetBackbone with RTDetrFrozenBatchNorm2d
# - Encoder: encoder_input_proj, lateral_convs, fpn_blocks, etc. with BatchNorm2d
#            encoder.layers with LayerNorm
# - Decoder: decoder_input_proj with BatchNorm2d, decoder.layers with LayerNorm
#
# 1. Backbone layers only (RTDetrFrozenBatchNorm2d):
# actmad_backbone = ActMAD.load(model, data_root, adaptation_layers="backbone")
# dua_backbone = DUA.load(model, data_root, adaptation_layers="backbone")
# norm_backbone = NORM.load(model, data_root, adaptation_layers="backbone")
#
# 2. Encoder layers only (BatchNorm2d + LayerNorm):
# actmad_encoder = ActMAD.load(model, data_root, adaptation_layers="encoder")
# dua_encoder = DUA.load(model, data_root, adaptation_layers="encoder")
# norm_encoder = NORM.load(model, data_root, adaptation_layers="encoder")
#
# 3. Backbone + Encoder layers (default):
# actmad_both = ActMAD.load(model, data_root, adaptation_layers="backbone+encoder")
# dua_both = DUA.load(model, data_root, adaptation_layers="backbone+encoder")
# norm_both = NORM.load(model, data_root, adaptation_layers="backbone+encoder")
#
# Then run evaluation:
# results = method.evaluate_all_tasks()


class MeanTeacher:
    """
    Mean-Teacher method for RT-DETR with FixMatch-style augmentation
    """

    def __init__(
        self,
        model,
        data_root,
        device=None,
        batch_size=4,
        learning_rate=0.001,
        ema_alpha=0.99,
        conf_threshold=0.5,
        image_size=800,
        model_states=None,
        reference_model_id="PekingU/rtdetr_r50vd",
        class_num=6
    ):
        self.model = model
        self.data_root = data_root
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.ema_alpha = ema_alpha
        self.conf_threshold = conf_threshold
        self.image_size = image_size
        self.model_states = model_states
        self.reference_model_id = reference_model_id
        self.class_num = class_num

        self.teacher_model = None
        self.optimizer = None

        # Setup image processor
        self.preprocessor = RTDetrImageProcessorFast.from_pretrained(self.reference_model_id)
        self.preprocessor.format = AnnotationFormat.COCO_DETECTION
        self.preprocessor.do_resize = False
        self.preprocessor.size = {"height": self.image_size, "width": self.image_size}

    @classmethod
    def load(cls, model, data_root, **kwargs):
        instance = cls(model=model, data_root=data_root, **kwargs)
        instance._setup()
        return instance

    def _setup(self):
        self.model.to(self.device)
        self._setup_teacher_model()

    def _setup_teacher_model(self):
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Setup optimizer for student model
        for param in self.model.parameters():
            param.requires_grad = True

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

    def _update_teacher_ema(self):
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_model.parameters(), self.model.parameters()):
                teacher_param.data = self.ema_alpha * teacher_param.data + (1 - self.ema_alpha) * student_param.data

    def _apply_augmentation(self, images):
        """Apply simple augmentation for strong/weak augmentation"""
        augmented_images = []
        for image in images:
            # Simple augmentation - can be enhanced
            if random.random() > 0.5:
                # Random horizontal flip
                image = T.functional.hflip(image)
            augmented_images.append(image)
        return augmented_images

    def adapt_single_batch(self, batch):
        """
        Adapt the model on a single batch using Mean-Teacher for RT-DETR
        """
        images, labels = batch

        # Weak augmentation for teacher
        weak_images = images  # Use original images as weak augmentation

        # Strong augmentation for student
        strong_images = self._apply_augmentation(images)

        # Teacher forward (no gradient)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_inputs = self.preprocessor(images=weak_images, annotations=labels, return_tensors="pt")
            teacher_pixel_values = teacher_inputs['pixel_values'].to(self.device)
            teacher_outputs = self.teacher_model(pixel_values=teacher_pixel_values)

            # Generate pseudo labels from teacher predictions
            teacher_results = self.preprocessor.post_process_object_detection(
                teacher_outputs,
                target_sizes=torch.tensor([[self.image_size, self.image_size]] * input['pixel_values'].shape[0]),
                threshold=self.conf_threshold
            )

        # Create pseudo labels for strong augmented images
        pseudo_labels = []
        for result in teacher_results:
            if len(result['boxes']) > 0:
                pseudo_label = {
                    'boxes2d': result['boxes'].detach().cpu().numpy(),
                    'boxes2d_classes': result['labels'].detach().cpu().numpy()
                }
            else:
                pseudo_label = {'boxes2d': np.array([]), 'boxes2d_classes': np.array([])}
            pseudo_labels.append(pseudo_label)

        # Student training with pseudo labels
        if any(len(pl['boxes2d']) > 0 for pl in pseudo_labels):
            self.model.train()
            self.optimizer.zero_grad()

            try:
                # Prepare student inputs
                student_inputs = self.preprocessor(images=strong_images, annotations=pseudo_labels, return_tensors="pt")

                # Student forward pass
                student_pixel_values = student_inputs['pixel_values'].to(self.device)
                if 'labels' in student_inputs:
                    student_labels = [{k: v.to(self.device) for k, v in label.items()} for label in student_inputs['labels']]
                    student_outputs = self.model(pixel_values=student_pixel_values, labels=student_labels)

                    # Extract loss
                    if hasattr(student_outputs, 'loss'):
                        loss = student_outputs.loss
                        loss.backward()
                        self.optimizer.step()

                        # Update teacher with EMA
                        self._update_teacher_ema()

            except Exception as e:
                print(f"Training step failed: {e}")

        # Return teacher outputs for evaluation
        return teacher_outputs

    def evaluate_task(self, task=None, dataset=None, raw_data=None, valid_dataloader=None, threshold=0.0):
        """
        Evaluate on a specific task
        """
        if raw_data is None or valid_dataloader is None:
            raise ValueError("Both raw_data and valid_dataloader must be provided")

        task_name = task if task is not None else "unknown"
        print(f"Starting Mean-Teacher evaluation on {task_name}")

        if dataset is not None:
            CLASSES = dataset
        else:
            CLASSES = None

        map_metric = MeanAveragePrecision()
        predictions_list = []
        targets_list = []

        torch.cuda.empty_cache()
        gc.collect()

        inference_time = 0
        # Use the same evaluator as DirectMethod
        # Use local class definitions instead of utils import
        CLASSES = dataset.classes if hasattr(dataset, 'classes') else CLASSES
        # Import utils for Evaluator only
        from ttadapters.methods.other_method import utils
        evaluator = utils.Evaluator(class_list=CLASSES, task=task_name, reference_preprocessor=self.preprocessor)

        batch_count = 0

        # Process each batch following DirectMethod approach with TTA
        for batch_i, labels, input in zip(tqdm(range(len(raw_data))), raw_data, valid_dataloader):
            images, labels = convert_detectron_to_rtdetr_format(batch)
            start_time = time.time()
            outputs = self.adapt_single_batch(batch)
            inference_time += time.time() - start_time

            # Process outputs for evaluation
            results = self.preprocessor.post_process_object_detection(
                outputs, target_sizes=torch.tensor([[self.image_size, self.image_size]] * input['pixel_values'].shape[0]), threshold=threshold
            )

            for i, (result, target) in enumerate(zip(results, targets)):
                # Convert predictions
                if len(result['boxes']) > 0:
                    pred_detection = Detections(
                        xyxy=result['boxes'].detach().cpu().numpy(),
                        class_id=result['labels'].detach().cpu().numpy(),
                        confidence=result['scores'].detach().cpu().numpy()
                    )
                else:
                    pred_detection = Detections.empty()

                # Convert ground truth from COCO format
                if 'annotations' in target and len(target['annotations']) > 0:
                    # Convert COCO bbox format [x, y, w, h] to Pascal VOC [x1, y1, x2, y2]
                    boxes = []
                    class_ids = []
                    for ann in target['annotations']:
                        x, y, w, h = ann['bbox']
                        boxes.append([x, y, x + w, y + h])  # Convert to xyxy
                        class_ids.append(ann['category_id'])

                    target_detection = Detections(
                        xyxy=np.array(boxes),
                        class_id=np.array(class_ids)
                    )
                else:
                    target_detection = Detections.empty()

                predictions_list.append(pred_detection)
                targets_list.append(target_detection)

        map_metric.update(predictions=predictions_list, targets=targets_list)
        print(f"Computing mAP for {task_name}")
        m_ap = map_metric.compute()

        per_class_map = {}
        if CLASSES is not None and hasattr(m_ap, 'matched_classes'):
            per_class_map = {
                f"{CLASSES[idx]}_mAP@0.50:0.95": m_ap.ap_per_class[idx].mean().item()
                for idx in m_ap.matched_classes
            }

        total_samples = len(predictions_list)
        fps = total_samples / inference_time if inference_time > 0 else 0

        return {
            "mAP@0.50:0.95": m_ap.map50_95.item(),
            "mAP@0.50": m_ap.map50.item(),
            "mAP@0.75": m_ap.map75.item(),
            "inference_time": inference_time,
            "fps": fps,
            "total_samples": total_samples,
            **per_class_map,
        }

    def prepare_dataloaders(self, tasks=None):
        """
        Prepare dataloaders for all tasks in advance (following DirectMethod approach)
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = {}
        print("Preparing dataloaders for all tasks...")

        # Use local class definitions instead of utils import
        from functools import partial

        for task in tasks:
            print(f"Loading dataset for {task}...")

            # Create dataset using utils (same as DirectMethod)
            shift_dataset = SHIFTCorruptedTaskDatasetForObjectDetection(
                root=self.data_root, valid=True, task=task
            )

            # Create raw_data dataloader for labels (GT) - use SHIFT dataset directly
            raw_data = DataLoader(
                LabelDataset(shift_dataset),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=naive_collate_fn
            )

            # Create valid_dataloader for preprocessed inputs - use SHIFT dataset directly
            valid_dataloader = DataLoader(
                DatasetAdapterForTransformers(shift_dataset),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=partial(collate_fn, preprocessor=self.preprocessor)
            )

            # Keep shift_dataset as the main dataset reference
            dataset = shift_dataset

            dataloaders[task] = (dataset, raw_data, valid_dataloader)
            print(f"  - {task}: {len(dataset)} samples")

        print(f"All {len(dataloaders)} dataloaders prepared!")
        return dataloaders


    def evaluate_all_tasks(self, tasks=None):
        """
        Evaluate on all tasks (automatically prepares dataloaders and runs evaluation)
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = self.prepare_dataloaders(tasks)

        results = {}
        for task in tasks:
            if task not in dataloaders:
                print(f"Warning: No dataloader found for task {task}, skipping...")
                continue

            dataset, raw_data, valid_dataloader = dataloaders[task]
            results[task] = self.evaluate_task(
                task=task,
                dataset=dataset,
                raw_data=raw_data,
                valid_dataloader=valid_dataloader
            )
            print(f"Results for {task}: {results[task]}")

        return results
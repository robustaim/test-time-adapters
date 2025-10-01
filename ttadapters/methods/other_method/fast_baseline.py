import os
import math
import copy
from os import path, makedirs
from os.path import join
from pathlib import Path
from tqdm.auto import tqdm
import time
import gc
import random
import numpy as np
from typing import Union, Optional, Any
from torch import save, load

import os
os.chdir("/workspace/ptta") # os.chdir("/home/ubuntu/test-time-adapters")

import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
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

# ContinualTTA의 custom detectron2 import (표준 detectron2보다 우선)
from detectron2.layers import FrozenBatchNorm2d
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import EventStorage

from torchvision.tv_tensors import Image, BoundingBoxes

# PIL imports for FixMatch augmentation
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image as PILImage

from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.modeling import GeneralizedRCNN, SwinTransformer
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo

from torchvision.tv_tensors import Image, BoundingBoxes
from detectron2.structures import Boxes, Instances

from torch import hub, nn

import warnings

from torch.utils.data import Dataset

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


class SHIFTCorruptedDatasetForObjectDetection(SHIFTDiscreteSubsetForObjectDetection):
    """SHIFT corrupted dataset wrapper"""
    def __init__(
            self, root: str, force_download: bool = False,
            train: bool = True, valid: bool = False,
            transform=None, target_transform=None, transforms=None,
            task="clear"
    ):
        super().__init__(
            root=root, force_download=force_download,
            train=train, valid=valid, subset_type=task_to_subset_types(task),
            transform=transform, target_transform=target_transform, transforms=transforms
        )


def collate_fn(batch):
    """Collate function for object detection"""
    batched_inputs = []
    for image, metadata in batch:
        original_height, original_width = image.shape[-2:]
        instances = Instances(image_size=(original_height, original_width))
        instances.gt_boxes = Boxes(metadata["boxes2d"])  # xyxy
        instances.gt_classes = metadata["boxes2d_classes"]
        batched_inputs.append({
            "image": image,
            "instances": instances,
            "height": original_height,
            "width": original_width
        })
    return batched_inputs

class DIRECT:
    """
    DIRECT (No Adaptation) method - baseline without any test-time adaptation
    """

    def __init__(
        self,
        model,
        data_root,
        device=None,
        batch_size=4
    ):
        self.model = model
        self.data_root = data_root
        self.device = device
        self.batch_size = batch_size

    @classmethod
    def load(cls, model, data_root, **kwargs):
        """
        Factory method to create and initialize DIRECT

        Args:
            model: The model to adapt
            data_root: Path to the data root directory
            **kwargs: Additional parameters

        Returns:
            DIRECT: Initialized DIRECT instance
        """
        instance = cls(model=model, data_root=data_root, **kwargs)
        instance._setup()
        return instance

    def _setup(self):
        """Setup the DIRECT method"""
        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)

    def adapt_single_batch(self, batch):
        """
        Run inference on a single batch without adaptation

        Args:
            batch: Input batch

        Returns:
            outputs: Model outputs
        """
        self.model.eval()
        outputs = self.model(batch)
        return outputs

    def evaluate_task(self, task=None, dataset=None, dataloader=None, threshold=0.0):
        """
        Evaluate on a specific task

        Args:
            task: Task name (e.g., "cloudy", "overcast", etc.) - used for logging only
            dataset: Pre-loaded dataset (optional, for CLASSES info)
            dataloader: Pre-loaded dataloader
            threshold: Confidence threshold for predictions

        Returns:
            dict: Evaluation results including FPS measurement
        """
        if dataloader is None:
            raise ValueError("dataloader must be provided for accurate FPS measurement")

        task_name = task if task is not None else "unknown"
        print(f"Starting DIRECT evaluation on {task_name}")

        if dataset is not None:
            CLASSES = dataset
        else:
            CLASSES = None

        map_metric = MeanAveragePrecision()
        predictions_list = []
        targets_list = []

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            for batch in tqdm(dataloader, total=dataloader.valid_len, desc=f"DIRECT {task_name}"):
                start_time = time.time()
                outputs = self.adapt_single_batch(batch)

                for i, (output, input_data) in enumerate(zip(outputs, batch)):
                    instances = output['instances']
                    mask = instances.scores > threshold

                    pred_detection = Detections(
                        xyxy=instances.pred_boxes.tensor[mask].detach().cpu().numpy(),
                        class_id=instances.pred_classes[mask].detach().cpu().numpy(),
                        confidence=instances.scores[mask].detach().cpu().numpy()
                    )
                    gt_instances = input_data['instances']
                    target_detection = Detections(
                        xyxy=gt_instances.gt_boxes.tensor.detach().cpu().numpy(),
                        class_id=gt_instances.gt_classes.detach().cpu().numpy()
                    )

                    predictions_list.append(pred_detection)
                    targets_list.append(target_detection)

        torch.cuda.synchronize()
        end = time.perf_counter()

        map_metric.update(predictions=predictions_list, targets=targets_list)
        print(f"Computing mAP for {task_name}")
        m_ap = map_metric.compute()

        per_class_map = {}
        if CLASSES is not None:
            per_class_map = {
                f"{CLASSES[idx]}_mAP@0.50:0.95": m_ap.ap_per_class[idx].mean().item()
                for idx in m_ap.matched_classes
            }

        total_samples = len(predictions_list)
        inference_time = end - start

        return {
            "mAP@0.50:0.95": m_ap.map50_95.item(),
            "mAP@0.50": m_ap.map50.item(),
            "mAP@0.75": m_ap.map75.item(),
            "inference_time": inference_time,
            "total_samples": total_samples,
            **per_class_map,
        }

    def prepare_dataloaders(self, tasks=None):
        """
        Prepare dataloaders for all tasks in advance

        Args:
            tasks: List of task names. If None, uses default tasks.

        Returns:
            dict: Dictionary with task names as keys and (dataset, dataloader) tuples as values
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = {}
        print("Preparing dataloaders for all tasks...")

        for task in tasks:
            print(f"Loading dataset for {task}...")
            dataset = SHIFTCorruptedDatasetForObjectDetection(
                root=self.data_root, valid=True,
                transform=datasets.detectron_image_transform,
                transforms=datasets.default_valid_transforms,
                task=task
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            dataloader.valid_len = math.ceil(len(dataset) / self.batch_size)

            dataloaders[task] = (dataset, dataloader)
            print(f"  - {task}: {len(dataset)} samples, {dataloader.valid_len} batches")

        print(f"All {len(dataloaders)} dataloaders prepared!")
        return dataloaders

    def evaluate_all_tasks(self, tasks=None):
        """
        Evaluate on all tasks (automatically prepares dataloaders and runs evaluation)

        Args:
            tasks: List of task names. If None, uses default tasks.

        Returns:
            dict: Results for all tasks
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = self.prepare_dataloaders(tasks)

        results = {}
        final_inference_time = 0.0
        final_total_samples = 0

        for task in tasks:
            if task not in dataloaders:
                print(f"Warning: No dataloader found for task {task}, skipping...")
                continue

            dataset, dataloader = dataloaders[task]
            results[task] = self.evaluate_task(
                task=task,
                dataset=dataset,
                dataloader=dataloader
            )

            final_inference_time += results[task]["inference_time"]
            final_total_samples += results[task]["total_samples"]
            print(f"Results for {task}: {results[task]}")
        fps = final_total_samples / final_inference_time
        print(f"FPS : {fps:.3f}")
        return results

class ActMAD:
    """
    ActMAD (Activation Mean Alignment and Discrepancy) method for test-time adaptation
    """

    def __init__(
        self,
        model,
        data_root,
        device=None,
        batch_size=4,
        learning_rate=0.0001,
        clean_bn_extract_batch=8
    ):
        self.model = model
        self.data_root = data_root
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.clean_bn_extract_batch = clean_bn_extract_batch

        # Statistics storage
        self.clean_mean_list_final = None
        self.clean_var_list_final = None
        self.layer_names = None
        self.chosen_bn_layers = None

        # Optimizer
        self.optimizer = None
        self.l1_loss = nn.L1Loss(reduction="mean")

        # Statistics save path
        self.stats_save_path = Path("/workspace/ptta/ttadapters/methods/other_method") / "actmad_clean_statistics_faster_rcnn.pt"

    @classmethod
    def load(cls, model, data_root, **kwargs):
        """
        Factory method to create and initialize ActMAD

        Args:
            model: The model to adapt
            data_root: Path to the data root directory
            **kwargs: Additional parameters

        Returns:
            ActMAD: Initialized ActMAD instance
        """
        instance = cls(model=model, data_root=data_root, **kwargs)
        instance._setup()
        return instance

    def _setup(self):
        """Setup the ActMAD method"""
        # Move model to device
        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)

        # Unfreeze model parameters
        for k, v in self.model.named_parameters():
            v.requires_grad = True

        # Setup optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # Extract or load clean statistics
        self._extract_or_load_clean_statistics()

    def extract_activation_alignment(self, method, data_root, batch_size=16):
        """Extract activation alignment from clean data"""
        dataset = SHIFTClearDatasetForObjectDetection(
            root=data_root, train=True,
            transform=datasets.detectron_image_transform,
            transforms=datasets.default_valid_transforms
        )

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        loader.train_len = math.ceil(len(dataset)/batch_size)

        # model unfreeze
        for k, v in self.model.named_parameters():
            v.requires_grad = True

        chosen_bn_info = []
        if method == "actmad":
            for name, m in self.model.named_modules():
                if isinstance(m, (FrozenBatchNorm2d)):
                    chosen_bn_info.append((name, m))

        # chosen_bn_layers
        """
        Since high-level representations are more sensitive to domain shift,
        only the later BN layers are selected.
        The cutoff point is determined empirically.
        """
        cutoff = len(chosen_bn_info) // 2
        chosen_bn_info = chosen_bn_info[cutoff:]
        chosen_bn_layers = [module for name, module in chosen_bn_info]
        layer_names = [name for name, module in chosen_bn_info]

        n_chosen_layers = len(chosen_bn_layers)

        save_outputs = [utils.SaveOutput() for _ in range(n_chosen_layers)]

        clean_mean_act_list = [utils.AverageMeter() for _ in range(n_chosen_layers)]
        clean_var_act_list = [utils.AverageMeter() for _ in range(n_chosen_layers)]

        clean_mean_list_final = []
        clean_var_list_final = []

        # extract the activation alignment in train dataset
        print("Start extracting BN statistics from the training dataset")

        with torch.no_grad():
            for batch in tqdm(loader, total=loader.train_len, desc="Evaluation"):
                self.model.eval()
                hook_list = [chosen_bn_layers[i].register_forward_hook(save_outputs[i]) for i in range(n_chosen_layers)]
                _ = self.model(batch)

                for i in range(n_chosen_layers):
                    clean_mean_act_list[i].update(save_outputs[i].get_out_mean())  # compute mean from clean data
                    clean_var_act_list[i].update(save_outputs[i].get_out_var())  # compute variane from clean data

                    save_outputs[i].clear()
                    hook_list[i].remove()

            for i in range(n_chosen_layers):
                clean_mean_list_final.append(clean_mean_act_list[i].avg)  # [C, H, W]
                clean_var_list_final.append(clean_var_act_list[i].avg)  # [C, H, W]

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

            # Statistics만 저장 (chosen_bn_layers는 저장하지 않음)
            print(f"Saving ActMAD statistics to {self.stats_save_path}")
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
            if isinstance(module, FrozenBatchNorm2d)
        }

        self.chosen_bn_layers = []
        for layer_name in self.layer_names:
            if layer_name in current_bn_dict:
                self.chosen_bn_layers.append(current_bn_dict[layer_name])
            else:
                print(f"Warning: Layer {layer_name} not found!")

    def adapt_single_batch(self, batch):
        """
        Adapt the model on a single batch using ActMAD

        Args:
            batch: Input batch

        Returns:
            outputs: Model outputs after adaptation
        """
        # Ensure model parameters are unfrozen
        for param in self.model.parameters():
            param.requires_grad = True

        self.model.eval()
        self.optimizer.zero_grad()

        n_chosen_layers = len(self.chosen_bn_layers)
        save_outputs_tta = [utils.SaveOutput() for _ in range(n_chosen_layers)]

        hook_list_tta = [
            self.chosen_bn_layers[x].register_forward_hook(save_outputs_tta[x])
            for x in range(n_chosen_layers)
        ]

        # Forward pass
        outputs = self.model(batch)

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
        self.optimizer.step()

        # Clean up hooks
        for z in range(n_chosen_layers):
            save_outputs_tta[z].clear()
            hook_list_tta[z].remove()

        return outputs

    def evaluate_task(self, task=None, dataset=None, dataloader=None, threshold=0.0):
        """
        Evaluate on a specific task

        Args:
            task: Task name (e.g., "cloudy", "overcast", etc.) - used for logging only
            dataset: Pre-loaded dataset (optional, for CLASSES info)
            dataloader: Pre-loaded dataloader
            threshold: Confidence threshold for predictions

        Returns:
            dict: Evaluation results including FPS measurement
        """
        if dataloader is None:
            raise ValueError("dataloader must be provided for accurate FPS measurement")

        task_name = task if task is not None else "unknown"
        print(f"Starting ActMAD evaluation on {task_name}")

        # Get classes info from dataset if provided
        if dataset is not None:
            CLASSES = dataset
        else:
            CLASSES = None

        # Evaluation metrics
        map_metric = MeanAveragePrecision()
        predictions_list = []
        targets_list = []

        # Performance measurement
        import time
        import gc

        torch.cuda.empty_cache()
        gc.collect()

        inference_time = 0
        total_batches = len(dataloader) if hasattr(dataloader, '__len__') else getattr(dataloader, 'valid_len', None)

        for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches, desc=f"ActMAD {task_name}")):
            # Start timing for inference only (excluding data loading)
            start_time = time.time()

            # Adapt and get outputs
            outputs = self.adapt_single_batch(batch)

            # End timing
            inference_time += time.time() - start_time

            # Process outputs for evaluation
            for i, (output, input_data) in enumerate(zip(outputs, batch)):
                instances = output['instances']
                mask = instances.scores > threshold

                pred_detection = Detections(
                    xyxy=instances.pred_boxes.tensor[mask].detach().cpu().numpy(),
                    class_id=instances.pred_classes[mask].detach().cpu().numpy(),
                    confidence=instances.scores[mask].detach().cpu().numpy()
                )
                gt_instances = input_data['instances']
                target_detection = Detections(
                    xyxy=gt_instances.gt_boxes.tensor.detach().cpu().numpy(),
                    class_id=gt_instances.gt_classes.detach().cpu().numpy()
                )

                predictions_list.append(pred_detection)
                targets_list.append(target_detection)

        # Compute metrics
        map_metric.update(predictions=predictions_list, targets=targets_list)
        print(f"Computing mAP for {task_name}")
        m_ap = map_metric.compute()

        per_class_map = {}
        if CLASSES is not None:
            per_class_map = {
                f"{CLASSES[idx]}_mAP@0.50:0.95": m_ap.ap_per_class[idx].mean().item()
                for idx in m_ap.matched_classes
            }

        # Performance metrics
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
        Prepare dataloaders for all tasks in advance

        Args:
            tasks: List of task names. If None, uses default tasks.

        Returns:
            dict: Dictionary with task names as keys and (dataset, dataloader) tuples as values
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = {}
        print("Preparing dataloaders for all tasks...")

        for task in tasks:
            print(f"Loading dataset for {task}...")
            dataset = SHIFTCorruptedDatasetForObjectDetection(
                root=self.data_root, valid=True,
                transform=datasets.detectron_image_transform,
                transforms=datasets.default_valid_transforms,
                task=task
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            dataloader.valid_len = math.ceil(len(dataset) / self.batch_size)

            dataloaders[task] = (dataset, dataloader)
            print(f"  - {task}: {len(dataset)} samples, {dataloader.valid_len} batches")

        print(f"All {len(dataloaders)} dataloaders prepared!")
        return dataloaders

    def evaluate_all_tasks(self, tasks=None):
        """
        Evaluate on all tasks (automatically prepares dataloaders and runs evaluation)

        Args:
            tasks: List of task names. If None, uses default tasks.

        Returns:
            dict: Results for all tasks
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

            dataset, dataloader = dataloaders[task]
            results[task] = self.evaluate_task(
                task=task,
                dataset=dataset,
                dataloader=dataloader
            )
            print(f"Results for {task}: {results[task]}")

        return results


class DUA:
    """
    DUA (Dynamic Update Adaptation) method for test-time adaptation
    """

    def __init__(
        self,
        model,
        data_root,
        device=None,
        batch_size=4,
        decay_factor=0.94,
        mom_pre=0.01,
        min_momentum_constant=0.0001
    ):
        self.model = model
        self.data_root = data_root
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.decay_factor = decay_factor
        self.mom_pre = mom_pre
        self.min_momentum_constant = min_momentum_constant

    @classmethod
    def load(cls, model, data_root, **kwargs):
        """
        Factory method to create and initialize DUA

        Args:
            model: The model to adapt
            data_root: Path to the data root directory
            **kwargs: Additional parameters

        Returns:
            DUA: Initialized DUA instance
        """
        instance = cls(model=model, data_root=data_root, **kwargs)
        instance._setup()
        return instance

    def _setup(self):
        """Setup the DUA method"""
        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)
        self._apply_dua_adaptation()

    def _apply_dua_adaptation(self):
        """Apply DUA adaptation to BatchNorm layers"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d)):
                module.adapt_type = "DUA"
                module.min_momentum_constant = self.min_momentum_constant
                module.decay_factor = self.decay_factor
                module.mom_pre = self.mom_pre

                if not hasattr(module, 'original_running_mean'):
                    module.original_running_mean = module.running_mean.clone()
                    module.original_running_var = module.running_var.clone()

                def dua_forward(self, x):
                    if hasattr(self, 'adapt_type') and self.adapt_type == "DUA":
                        with torch.no_grad():
                            current_momentum = self.mom_pre + self.min_momentum_constant
                            batch_mean = x.mean(dim=[0, 2, 3])
                            batch_var = x.var(dim=[0, 2, 3], unbiased=True)

                            self.running_mean.mul_(1 - current_momentum).add_(batch_mean, alpha=current_momentum)
                            self.running_var.mul_(1 - current_momentum).add_(batch_var, alpha=current_momentum)
                            self.mom_pre *= self.decay_factor
                        scale = self.weight * (self.running_var + self.eps).rsqrt()
                        bias = self.bias - self.running_mean * scale
                    else:
                        scale = self.weight * (self.running_var + self.eps).rsqrt()
                        bias = self.bias - self.running_mean * scale

                    scale = scale.reshape(1, -1, 1, 1)
                    bias = bias.reshape(1, -1, 1, 1)
                    out_dtype = x.dtype
                    out = x * scale.to(out_dtype) + bias.to(out_dtype)

                    return out

                module.forward = dua_forward.__get__(module, module.__class__)

    def reset_dua_momentum(self, mom_pre=0.01):
        """Reset DUA momentum for new task"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d)) and hasattr(module, 'adapt_type'):
                if module.adapt_type == "DUA":
                    module.mom_pre = mom_pre
                    if hasattr(module, 'original_running_mean'):
                        module.running_mean = module.original_running_mean.clone()
                        module.running_var = module.original_running_var.clone()

    def adapt_single_batch(self, batch):
        """
        Adapt the model on a single batch using DUA

        Args:
            batch: Input batch

        Returns:
            outputs: Model outputs after adaptation
        """
        self.model.eval()
        outputs = self.model(batch)
        return outputs

    def evaluate_task(self, task=None, dataset=None, dataloader=None, threshold=0.0):
        """
        Evaluate on a specific task

        Args:
            task: Task name (e.g., "cloudy", "overcast", etc.) - used for logging only
            dataset: Pre-loaded dataset (optional, for CLASSES info)
            dataloader: Pre-loaded dataloader
            threshold: Confidence threshold for predictions

        Returns:
            dict: Evaluation results including FPS measurement
        """
        if dataloader is None:
            raise ValueError("dataloader must be provided for accurate FPS measurement")

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
        total_batches = len(dataloader) if hasattr(dataloader, '__len__') else getattr(dataloader, 'valid_len', None)

        for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches, desc=f"DUA {task_name}")):
            start_time = time.time()
            outputs = self.adapt_single_batch(batch)
            inference_time += time.time() - start_time

            for i, (output, input_data) in enumerate(zip(outputs, batch)):
                instances = output['instances']
                mask = instances.scores > threshold

                pred_detection = Detections(
                    xyxy=instances.pred_boxes.tensor[mask].detach().cpu().numpy(),
                    class_id=instances.pred_classes[mask].detach().cpu().numpy(),
                    confidence=instances.scores[mask].detach().cpu().numpy()
                )
                gt_instances = input_data['instances']
                target_detection = Detections(
                    xyxy=gt_instances.gt_boxes.tensor.detach().cpu().numpy(),
                    class_id=gt_instances.gt_classes.detach().cpu().numpy()
                )

                predictions_list.append(pred_detection)
                targets_list.append(target_detection)

        map_metric.update(predictions=predictions_list, targets=targets_list)
        print(f"Computing mAP for {task_name}")
        m_ap = map_metric.compute()

        per_class_map = {}
        if CLASSES is not None:
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
        Prepare dataloaders for all tasks in advance

        Args:
            tasks: List of task names. If None, uses default tasks.

        Returns:
            dict: Dictionary with task names as keys and (dataset, dataloader) tuples as values
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = {}
        print("Preparing dataloaders for all tasks...")

        for task in tasks:
            print(f"Loading dataset for {task}...")
            dataset = SHIFTCorruptedDatasetForObjectDetection(
                root=self.data_root, valid=True,
                transform=datasets.detectron_image_transform,
                transforms=datasets.default_valid_transforms,
                task=task
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            dataloader.valid_len = math.ceil(len(dataset) / self.batch_size)

            dataloaders[task] = (dataset, dataloader)
            print(f"  - {task}: {len(dataset)} samples, {dataloader.valid_len} batches")

        print(f"All {len(dataloaders)} dataloaders prepared!")
        return dataloaders

    def evaluate_all_tasks(self, tasks=None):
        """
        Evaluate on all tasks (automatically prepares dataloaders and runs evaluation)

        Args:
            tasks: List of task names. If None, uses default tasks.

        Returns:
            dict: Results for all tasks
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = self.prepare_dataloaders(tasks)

        results = {}
        for task in tasks:
            if task not in dataloaders:
                print(f"Warning: No dataloader found for task {task}, skipping...")
                continue

            dataset, dataloader = dataloaders[task]
            results[task] = self.evaluate_task(
                task=task,
                dataset=dataset,
                dataloader=dataloader
            )
            print(f"Results for {task}: {results[task]}")

        return results


class NORM:
    """
    NORM (Normalization) method for test-time adaptation
    """

    def __init__(
        self,
        model,
        data_root,
        device=None,
        batch_size=4,
        source_sum=128
    ):
        self.model = model
        self.data_root = data_root
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.source_sum = source_sum

    @classmethod
    def load(cls, model, data_root, **kwargs):
        """
        Factory method to create and initialize NORM

        Args:
            model: The model to adapt
            data_root: Path to the data root directory
            **kwargs: Additional parameters

        Returns:
            NORM: Initialized NORM instance
        """
        instance = cls(model=model, data_root=data_root, **kwargs)
        instance._setup()
        return instance

    def _setup(self):
        """Setup the NORM method"""
        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)
        self._apply_norm_adaptation()

    def _apply_norm_adaptation(self):
        """Apply NORM adaptation to BatchNorm layers"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d)):
                module.adapt_type = "NORM"
                module.source_sum = self.source_sum

                def norm_forward(self, x):
                    if hasattr(self, 'adapt_type') and self.adapt_type == "NORM":
                        alpha = x.shape[0] / (self.source_sum + x.shape[0])
                        running_mean = (1 - alpha) * self.running_mean + alpha * x.mean(dim=[0,2,3])
                        running_var = (1 - alpha) * self.running_var + alpha * x.var(dim=[0,2,3])
                        scale = self.weight * (running_var + self.eps).rsqrt()
                        bias = self.bias - running_mean * scale
                    else:
                        scale = self.weight * (self.running_var + self.eps).rsqrt()
                        bias = self.bias - self.running_mean * scale

                    scale = scale.reshape(1, -1, 1, 1)
                    bias = bias.reshape(1, -1, 1, 1)
                    out_dtype = x.dtype
                    out = x * scale.to(out_dtype) + bias.to(out_dtype)
                    return out

                module.forward = norm_forward.__get__(module, module.__class__)

    def adapt_single_batch(self, batch):
        """
        Adapt the model on a single batch using NORM

        Args:
            batch: Input batch

        Returns:
            outputs: Model outputs after adaptation
        """
        self.model.eval()
        outputs = self.model(batch)
        return outputs

    def evaluate_task(self, task=None, dataset=None, dataloader=None, threshold=0.0):
        """
        Evaluate on a specific task

        Args:
            task: Task name (e.g., "cloudy", "overcast", etc.) - used for logging only
            dataset: Pre-loaded dataset (optional, for CLASSES info)
            dataloader: Pre-loaded dataloader
            threshold: Confidence threshold for predictions

        Returns:
            dict: Evaluation results including FPS measurement
        """
        if dataloader is None:
            raise ValueError("dataloader must be provided for accurate FPS measurement")

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
        total_batches = len(dataloader) if hasattr(dataloader, '__len__') else getattr(dataloader, 'valid_len', None)

        for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches, desc=f"NORM {task_name}")):
            start_time = time.time()
            outputs = self.adapt_single_batch(batch)
            inference_time += time.time() - start_time

            for i, (output, input_data) in enumerate(zip(outputs, batch)):
                instances = output['instances']
                mask = instances.scores > threshold

                pred_detection = Detections(
                    xyxy=instances.pred_boxes.tensor[mask].detach().cpu().numpy(),
                    class_id=instances.pred_classes[mask].detach().cpu().numpy(),
                    confidence=instances.scores[mask].detach().cpu().numpy()
                )
                gt_instances = input_data['instances']
                target_detection = Detections(
                    xyxy=gt_instances.gt_boxes.tensor.detach().cpu().numpy(),
                    class_id=gt_instances.gt_classes.detach().cpu().numpy()
                )

                predictions_list.append(pred_detection)
                targets_list.append(target_detection)

        map_metric.update(predictions=predictions_list, targets=targets_list)
        print(f"Computing mAP for {task_name}")
        m_ap = map_metric.compute()

        per_class_map = {}
        if CLASSES is not None:
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
        Prepare dataloaders for all tasks in advance

        Args:
            tasks: List of task names. If None, uses default tasks.

        Returns:
            dict: Dictionary with task names as keys and (dataset, dataloader) tuples as values
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = {}
        print("Preparing dataloaders for all tasks...")

        for task in tasks:
            print(f"Loading dataset for {task}...")
            dataset = SHIFTCorruptedDatasetForObjectDetection(
                root=self.data_root, valid=True,
                transform=datasets.detectron_image_transform,
                transforms=datasets.default_valid_transforms,
                task=task
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            dataloader.valid_len = math.ceil(len(dataset) / self.batch_size)

            dataloaders[task] = (dataset, dataloader)
            print(f"  - {task}: {len(dataset)} samples, {dataloader.valid_len} batches")

        print(f"All {len(dataloaders)} dataloaders prepared!")
        return dataloaders

    def evaluate_all_tasks(self, tasks=None):
        """
        Evaluate on all tasks (automatically prepares dataloaders and runs evaluation)

        Args:
            tasks: List of task names. If None, uses default tasks.

        Returns:
            dict: Results for all tasks
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = self.prepare_dataloaders(tasks)

        results = {}
        for task in tasks:
            if task not in dataloaders:
                print(f"Warning: No dataloader found for task {task}, skipping...")
                continue

            dataset, dataloader = dataloaders[task]
            results[task] = self.evaluate_task(
                task=task,
                dataset=dataset,
                dataloader=dataloader
            )
            print(f"Results for {task}: {results[task]}")

        return results


class DUA:
    """
    DUA (Dynamic Update Adaptation) method for test-time adaptation
    """

    def __init__(
        self,
        model,
        data_root,
        device=None,
        batch_size=4,
        decay_factor=0.94,
        mom_pre=0.01,
        min_momentum_constant=0.0001
    ):
        self.model = model
        self.data_root = data_root
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.decay_factor = decay_factor
        self.mom_pre = mom_pre
        self.min_momentum_constant = min_momentum_constant

    @classmethod
    def load(cls, model, data_root, **kwargs):
        """
        Factory method to create and initialize DUA

        Args:
            model: The model to adapt
            data_root: Path to the data root directory
            **kwargs: Additional parameters

        Returns:
            DUA: Initialized DUA instance
        """
        instance = cls(model=model, data_root=data_root, **kwargs)
        instance._setup()
        return instance

    def _setup(self):
        """Setup the DUA method"""
        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)
        self._apply_dua_adaptation()

    def _apply_dua_adaptation(self):
        """Apply DUA adaptation to BatchNorm layers"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d)):
                module.adapt_type = "DUA"
                module.min_momentum_constant = self.min_momentum_constant
                module.decay_factor = self.decay_factor
                module.mom_pre = self.mom_pre

                if not hasattr(module, 'original_running_mean'):
                    module.original_running_mean = module.running_mean.clone()
                    module.original_running_var = module.running_var.clone()

                def dua_forward(self, x):
                    if hasattr(self, 'adapt_type') and self.adapt_type == "DUA":
                        with torch.no_grad():
                            current_momentum = self.mom_pre + self.min_momentum_constant
                            batch_mean = x.mean(dim=[0, 2, 3])
                            batch_var = x.var(dim=[0, 2, 3], unbiased=True)

                            self.running_mean.mul_(1 - current_momentum).add_(batch_mean, alpha=current_momentum)
                            self.running_var.mul_(1 - current_momentum).add_(batch_var, alpha=current_momentum)
                            self.mom_pre *= self.decay_factor
                        scale = self.weight * (self.running_var + self.eps).rsqrt()
                        bias = self.bias - self.running_mean * scale
                    else:
                        scale = self.weight * (self.running_var + self.eps).rsqrt()
                        bias = self.bias - self.running_mean * scale

                    scale = scale.reshape(1, -1, 1, 1)
                    bias = bias.reshape(1, -1, 1, 1)
                    out_dtype = x.dtype
                    out = x * scale.to(out_dtype) + bias.to(out_dtype)

                    return out

                module.forward = dua_forward.__get__(module, module.__class__)

    def reset_dua_momentum(self, mom_pre=0.01):
        """Reset DUA momentum for new task"""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d)) and hasattr(module, 'adapt_type'):
                if module.adapt_type == "DUA":
                    module.mom_pre = mom_pre
                    if hasattr(module, 'original_running_mean'):
                        module.running_mean = module.original_running_mean.clone()
                        module.running_var = module.original_running_var.clone()

    def adapt_single_batch(self, batch):
        """
        Adapt the model on a single batch using DUA

        Args:
            batch: Input batch

        Returns:
            outputs: Model outputs after adaptation
        """
        self.model.eval()
        outputs = self.model(batch)
        return outputs

    def evaluate_task(self, task=None, dataset=None, dataloader=None, threshold=0.0):
        """
        Evaluate on a specific task

        Args:
            task: Task name (e.g., "cloudy", "overcast", etc.) - used for logging only
            dataset: Pre-loaded dataset (optional, for CLASSES info)
            dataloader: Pre-loaded dataloader
            threshold: Confidence threshold for predictions

        Returns:
            dict: Evaluation results including FPS measurement
        """
        if dataloader is None:
            raise ValueError("dataloader must be provided for accurate FPS measurement")

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
        total_batches = len(dataloader) if hasattr(dataloader, '__len__') else getattr(dataloader, 'valid_len', None)

        for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches, desc=f"DUA {task_name}")):
            start_time = time.time()
            outputs = self.adapt_single_batch(batch)
            inference_time += time.time() - start_time

            for i, (output, input_data) in enumerate(zip(outputs, batch)):
                instances = output['instances']
                mask = instances.scores > threshold

                pred_detection = Detections(
                    xyxy=instances.pred_boxes.tensor[mask].detach().cpu().numpy(),
                    class_id=instances.pred_classes[mask].detach().cpu().numpy(),
                    confidence=instances.scores[mask].detach().cpu().numpy()
                )
                gt_instances = input_data['instances']
                target_detection = Detections(
                    xyxy=gt_instances.gt_boxes.tensor.detach().cpu().numpy(),
                    class_id=gt_instances.gt_classes.detach().cpu().numpy()
                )

                predictions_list.append(pred_detection)
                targets_list.append(target_detection)

        map_metric.update(predictions=predictions_list, targets=targets_list)
        print(f"Computing mAP for {task_name}")
        m_ap = map_metric.compute()

        per_class_map = {}
        if CLASSES is not None:
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
        Prepare dataloaders for all tasks in advance

        Args:
            tasks: List of task names. If None, uses default tasks.

        Returns:
            dict: Dictionary with task names as keys and (dataset, dataloader) tuples as values
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = {}
        print("Preparing dataloaders for all tasks...")

        for task in tasks:
            print(f"Loading dataset for {task}...")
            dataset = SHIFTCorruptedDatasetForObjectDetection(
                root=self.data_root, valid=True,
                transform=datasets.detectron_image_transform,
                transforms=datasets.default_valid_transforms,
                task=task
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            dataloader.valid_len = math.ceil(len(dataset) / self.batch_size)

            dataloaders[task] = (dataset, dataloader)
            print(f"  - {task}: {len(dataset)} samples, {dataloader.valid_len} batches")

        print(f"All {len(dataloaders)} dataloaders prepared!")
        return dataloaders

    def evaluate_all_tasks(self, tasks=None):
        """
        Evaluate on all tasks (automatically prepares dataloaders and runs evaluation)

        Args:
            tasks: List of task names. If None, uses default tasks.

        Returns:
            dict: Results for all tasks
        """
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = self.prepare_dataloaders(tasks)

        results = {}
        for task in tasks:
            if task not in dataloaders:
                print(f"Warning: No dataloader found for task {task}, skipping...")
                continue

            dataset, dataloader = dataloaders[task]
            results[task] = self.evaluate_task(
                task=task,
                dataset=dataset,
                dataloader=dataloader
            )
            print(f"Results for {task}: {results[task]}")

        return results
    
class MeanTeacher:
    """
    Mean-Teacher method for test-time adaptation following ContinualTTA implementation
    """

    def __init__(
        self,
        model,
        data_root,
        device=None,
        batch_size=4,
        learning_rate=0.0001,
        ema_alpha=0.99,
        conf_threshold=0.5,
        weight_decay=0.0,
        weight_reg=0.0,
        momentum=0.9,
        adaptation_where="full",  # "full", "normalization", "head"
        augment_strength_n=2,  # RandAugment operations (student only, teacher uses original)
        augment_strength_m=10,  # RandAugment magnitude (student only, teacher uses original)
        cutout_size=16  # Cutout size (student only, teacher uses original)
    ):
        self.model = model
        self.data_root = data_root
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.ema_alpha = ema_alpha
        self.conf_threshold = conf_threshold
        self.weight_decay = weight_decay
        self.weight_reg = weight_reg
        self.momentum = momentum
        self.adaptation_where = adaptation_where
        self.augment_strength_n = augment_strength_n
        self.augment_strength_m = augment_strength_m
        self.cutout_size = cutout_size

        self.teacher_model = None
        self.optimizer = None
        self.init_weights = []

    @classmethod
    def load(cls, model, data_root, **kwargs):
        """
        Factory method to create and initialize MeanTeacher with improved ContinualTTA-like settings

        Args:
            model: The model to adapt
            data_root: Path to the data root directory
            **kwargs: Additional parameters including:
                - ema_alpha (float): EMA coefficient (default: 0.99)
                - conf_threshold (float): Confidence threshold (default: 0.5)
                - adaptation_where (str): Adaptation strategy - "full", "normalization", "head" (default: "full")
                - weight_reg (float): Weight regularization coefficient (default: 0.0)
                - learning_rate (float): Learning rate (default: 0.0001)
                - momentum (float): SGD momentum (default: 0.9)
                - weight_decay (float): Weight decay (default: 0.0)
        """
        instance = cls(model=model, data_root=data_root, **kwargs)
        instance._setup()
        return instance

    def _setup(self):
        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)
        self._setup_teacher_model()
        self._setup_training_mode()
        self._setup_strong_augmentation()

    def _setup_teacher_model(self):
        # Create teacher model as deep copy
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

        # Configure teacher model attributes similar to ContinualTTA
        self.teacher_model.online_adapt = False
        self.model.online_adapt = False


        # Set up optimizer following ContinualTTA's exact parameter selection strategy
        params = []

        if self.learning_rate > 0:
            if self.adaptation_where == "full" and hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'bottom_up'):
                # Full adaptation: adapt all backbone parameters like ContinualTTA
                for m_name, m in self.model.backbone.bottom_up.named_modules():
                    # Normalization layers (skip FrozenBatchNorm2d as it should stay frozen)
                    if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                        m.weight.requires_grad = True
                        m.bias.requires_grad = True
                        params += [m.weight, m.bias]

                    # Conv2d and Linear parameters (except patch_embed attention layers)
                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        if "patch_embed" in m_name and "attn" in m_name:
                            continue
                        m.weight.requires_grad = True
                        params += [m.weight]
                        if m.bias is not None:
                            m.bias.requires_grad = True
                            params += [m.bias]

            elif self.adaptation_where == "normalization":
                # Adapt normalization and lightweight layers (skip FrozenBatchNorm2d as it should stay frozen)
                for m_name, m in self.model.named_modules():
                    # Regular BatchNorm and LayerNorm
                    if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                        m.weight.requires_grad = True
                        m.bias.requires_grad = True
                        params += [m.weight, m.bias]

                    # Add bias parameters from Conv2d and Linear layers (lightweight adaptation)
                    elif isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None:
                        m.bias.requires_grad = True
                        params += [m.bias]

                    # Add Group Normalization if present
                    elif isinstance(m, nn.GroupNorm):
                        m.weight.requires_grad = True
                        m.bias.requires_grad = True
                        params += [m.weight, m.bias]

            elif self.adaptation_where == "head" and hasattr(self.model, 'roi_heads'):
                # Only adapt detection head
                self.model.roi_heads.box_head.requires_grad_(True)
                params += list(self.model.roi_heads.box_head.parameters())

            # Fallback: adapt backbone normalization and lightweight layers (skip FrozenBatchNorm2d)
            elif hasattr(self.model, 'backbone'):
                for name, module in self.model.backbone.named_modules():
                    # Regular BatchNorm and LayerNorm
                    if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                        if hasattr(module, 'weight') and module.weight is not None:
                            module.weight.requires_grad = True
                            params.append(module.weight)
                        if hasattr(module, 'bias') and module.bias is not None:
                            module.bias.requires_grad = True
                            params.append(module.bias)

                    # Add bias parameters from Conv2d and Linear layers
                    elif isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'bias') and module.bias is not None:
                        module.bias.requires_grad = True
                        params.append(module.bias)

        print(f"Mean-Teacher: Adapting {len(params)} parameters with strategy '{self.adaptation_where}'")

        # 실제로 학습 가능한 파라미터 개수 확인
        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        print(f"Mean-Teacher: Total trainable parameters: {trainable_params:,}")

        if params:
            self.optimizer = optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
            # Store initial weights for regularization
            self.init_weights = [p.clone().detach() for p in params]
        else:
            self.optimizer = None
            print("Warning: No parameters selected for adaptation!")

    def _setup_training_mode(self):
        """Setup training mode like ContinualTTA"""
        # Set model to training mode for adaptation (like ContinualTTA)
        self.model.training = True
        if hasattr(self.model, 'proposal_generator'):
            self.model.proposal_generator.training = True
        if hasattr(self.model, 'roi_heads'):
            self.model.roi_heads.training = True

        # Set BatchNorm layers to training mode (COMPLETELY SKIP FrozenBatchNorm2d)
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):  # FrozenBatchNorm2d 완전 제외
                module.train()

    def _setup_strong_augmentation(self):
        """Setup RandAugmentMC like ContinualTTA"""
        # Implement RandAugmentMC following ContinualTTA's implementation
        self.strong_augment = self._create_randaugment_mc(n=self.augment_strength_n, m=self.augment_strength_m)

    def _create_randaugment_mc(self, n, m):
        """Create RandAugmentMC similar to ContinualTTA implementation"""
        import random
        import numpy as np
        import PIL.ImageOps
        import PIL.ImageEnhance
        import PIL.ImageDraw
        from PIL import Image

        def AutoContrast(img, **kwarg):
            return PIL.ImageOps.autocontrast(img)

        def Brightness(img, v, max_v, bias=0):
            v = float(v) * max_v / 10 + bias
            return PIL.ImageEnhance.Brightness(img).enhance(v)

        def Color(img, v, max_v, bias=0):
            v = float(v) * max_v / 10 + bias
            return PIL.ImageEnhance.Color(img).enhance(v)

        def Contrast(img, v, max_v, bias=0):
            v = float(v) * max_v / 10 + bias
            return PIL.ImageEnhance.Contrast(img).enhance(v)

        def Equalize(img, **kwarg):
            return PIL.ImageOps.equalize(img)

        def Identity(img, **kwarg):
            return img

        def Posterize(img, v, max_v, bias=0):
            v = int(v * max_v / 10) + bias
            return PIL.ImageOps.posterize(img, v)

        def Sharpness(img, v, max_v, bias=0):
            v = float(v) * max_v / 10 + bias
            return PIL.ImageEnhance.Sharpness(img).enhance(v)

        def ShearX(img, v, max_v, bias=0):
            v = float(v) * max_v / 10 + bias
            if random.random() < 0.5:
                v = -v
            return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

        def ShearY(img, v, max_v, bias=0):
            v = float(v) * max_v / 10 + bias
            if random.random() < 0.5:
                v = -v
            return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

        def Solarize(img, v, max_v, bias=0):
            v = int(v * max_v / 10) + bias
            return PIL.ImageOps.solarize(img, 256 - v)

        def CutoutAbs(img, v, **kwarg):
            w, h = img.size
            x0 = np.random.uniform(0, w)
            y0 = np.random.uniform(0, h)
            x0 = int(max(0, x0 - v / 2.))
            y0 = int(max(0, y0 - v / 2.))
            x1 = int(min(w, x0 + v))
            y1 = int(min(h, y0 + v))
            xy = (x0, y0, x1, y1)
            color = (127, 127, 127)
            img = img.copy()
            PIL.ImageDraw.Draw(img).rectangle(xy, color)
            return img

        # FixMatch augmentation pool (same as ContinualTTA)
        augment_pool = [
            (AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0)
        ]

        class RandAugmentMC:
            def __init__(self, n, m, pool):
                self.n = n
                self.m = m
                self.augment_pool = pool

            def __call__(self, img):
                if isinstance(img, torch.Tensor):
                    # Convert tensor to PIL Image
                    img = T.ToPILImage()(img)
                elif isinstance(img, np.ndarray):
                    img = Image.fromarray(img)

                ops = random.choices(self.augment_pool, k=self.n)
                for op, max_v, bias in ops:
                    v = np.random.randint(1, self.m)
                    if random.random() < 0.5:
                        img = op(img, v=v, max_v=max_v, bias=bias)

                # Apply cutout (like ContinualTTA)
                img = CutoutAbs(img, self.cutout_size)

                # Convert back to numpy array then tensor
                img_array = np.array(img)
                return torch.as_tensor(np.ascontiguousarray(img_array.transpose(2, 0, 1)))

        return RandAugmentMC(n, m, augment_pool)


    def _update_teacher_ema(self):
        """Update teacher model parameters using EMA following ContinualTTA approach"""
        with torch.no_grad():
            # Update ALL parameters that require gradients (like ContinualTTA)
            for t_p, s_p in zip(self.teacher_model.parameters(), self.model.parameters()):
                if s_p.requires_grad:
                    t_p.data = self.ema_alpha * t_p.data + (1 - self.ema_alpha) * s_p.data

    def _apply_augmentation(self, batch):
        """Apply weak and strong augmentations like ContinualTTA"""
        weak_batch = []
        strong_batch = []

        for item in batch:
            # Weak augmentation (original)
            weak_item = copy.deepcopy(item)
            weak_batch.append(weak_item)

            # Strong augmentation
            strong_item = copy.deepcopy(item)
            try:
                # Apply RandAugmentMC which handles the conversion internally
                strong_item["strong_aug_image"] = self.strong_augment(strong_item["image"])
            except Exception as e:
                # Fallback to original image if augmentation fails
                strong_item["strong_aug_image"] = strong_item["image"]
            strong_batch.append(strong_item)

        return weak_batch, strong_batch

    def _set_pseudo_labels(self, inputs, outputs):
        """Set pseudo labels following ContinualTTA implementation exactly"""
        new_inputs = []
        for inp, oup in zip(inputs, outputs):
            # Get high confidence instances (like ContinualTTA)
            inst = oup['instances'][oup['instances'].scores > self.conf_threshold]

            # Create new input (always, even if no high-conf detections)
            new_inp = {k: inp[k] for k in inp if k not in ['instances', 'image', 'strong_aug_image']}
            new_inp['image'] = inp['strong_aug_image'] if 'strong_aug_image' in inp else inp['image']

            # Get image sizes
            new_img_size = inp['instances'].image_size
            ori_img_size = inst.image_size

            # Create new instances (even if empty)
            new_inst = Instances(new_img_size)
            new_inst.gt_classes = inst.pred_classes
            new_inst.gt_boxes = inst.pred_boxes

            # Scale boxes if needed
            if new_img_size != ori_img_size:
                new_inst.gt_boxes.scale(
                    new_img_size[1] / ori_img_size[1],
                    new_img_size[0] / ori_img_size[0]
                )

            new_inp['instances'] = new_inst
            new_inputs.append(new_inp)  # Always append, like ContinualTTA

        return new_inputs

    def adapt_single_batch(self, batch):
        """Adapt single batch following ContinualTTA mean-teacher approach"""
        # Apply weak and strong augmentations
        weak_batch, strong_batch = self._apply_augmentation(batch)

        # Teacher model inference on weak augmentation
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(weak_batch)

        # Perform adaptation if optimizer is available (following ContinualTTA exactly)
        if self.optimizer is not None:
            # Generate pseudo labels
            pseudo_labeled_batch = self._set_pseudo_labels(strong_batch, teacher_outputs)

            # Set model modes exactly like ContinualTTA
            self.model.online_adapt = False
            self.model.proposal_generator.training = True
            self.model.roi_heads.training = True

            # Always do training step like ContinualTTA (even with empty pseudo labels)
            self.optimizer.zero_grad()

            # Forward pass with EventStorage (following ContinualTTA)
            with EventStorage() as storage:
                model_output = self.model(pseudo_labeled_batch)

            # Handle losses and add weight regularization
            if isinstance(model_output, dict):
                losses = model_output
                total_detection_loss = sum([losses[k] for k in losses])

                # Add weight regularization like ContinualTTA
                if self.weight_reg > 0.0:
                    reg_loss = torch.tensor(0.0, device=self.device)
                    for param, init_param in zip(self.optimizer.param_groups[0]['params'], self.init_weights):
                        reg_loss += torch.mean((param - init_param) ** 2)
                    total_detection_loss += self.weight_reg * reg_loss

                if total_detection_loss > 0:
                    total_detection_loss.backward()

                    # Add gradient clipping (similar to ContinualTTA)
                    if hasattr(self.model, 'backbone'):
                        torch.nn.utils.clip_grad_norm_(self.model.backbone.parameters(), 1.0)

                    self.optimizer.step()

            # Always do EMA update like ContinualTTA
            self._update_teacher_ema()

        # Final inference using teacher model (more stable predictions)
        self.teacher_model.eval()
        with torch.no_grad():
            final_outputs = self.teacher_model(batch)  # Original batch, no augmentation

        return final_outputs

    def evaluate_task(self, task=None, dataset=None, dataloader=None, threshold=0.0):
        if dataloader is None:
            raise ValueError("dataloader must be provided for accurate FPS measurement")

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
        total_batches = len(dataloader) if hasattr(dataloader, '__len__') else getattr(dataloader, 'valid_len', None)

        for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches, desc=f"Mean-Teacher {task_name}")):
            start_time = time.time()
            outputs = self.adapt_single_batch(batch)
            inference_time += time.time() - start_time

            for i, (output, input_data) in enumerate(zip(outputs, batch)):
                instances = output['instances']
                mask = instances.scores > threshold

                pred_detection = Detections(
                    xyxy=instances.pred_boxes.tensor[mask].detach().cpu().numpy(),
                    class_id=instances.pred_classes[mask].detach().cpu().numpy(),
                    confidence=instances.scores[mask].detach().cpu().numpy()
                )
                gt_instances = input_data['instances']
                target_detection = Detections(
                    xyxy=gt_instances.gt_boxes.tensor.detach().cpu().numpy(),
                    class_id=gt_instances.gt_classes.detach().cpu().numpy()
                )

                predictions_list.append(pred_detection)
                targets_list.append(target_detection)

        map_metric.update(predictions=predictions_list, targets=targets_list)
        print(f"Computing mAP for {task_name}")
        m_ap = map_metric.compute()

        per_class_map = {}
        if CLASSES is not None:
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
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = {}
        print("Preparing dataloaders for all tasks...")

        for task in tasks:
            print(f"Loading dataset for {task}...")
            dataset = SHIFTCorruptedDatasetForObjectDetection(
                root=self.data_root, valid=True,
                transform=datasets.detectron_image_transform,
                transforms=datasets.default_valid_transforms,
                task=task
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            dataloader.valid_len = math.ceil(len(dataset) / self.batch_size)

            dataloaders[task] = (dataset, dataloader)
            print(f"  - {task}: {len(dataset)} samples, {dataloader.valid_len} batches")

        print(f"All {len(dataloaders)} dataloaders prepared!")
        return dataloaders

    def evaluate_all_tasks(self, tasks=None):
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = self.prepare_dataloaders(tasks)

        results = {}
        for task in tasks:
            if task not in dataloaders:
                print(f"Warning: No dataloader found for task {task}, skipping...")
                continue

            dataset, dataloader = dataloaders[task]
            results[task] = self.evaluate_task(
                task=task,
                dataset=dataset,
                dataloader=dataloader
            )
            print(f"Results for {task}: {results[task]}")

        return results


class ParallelAdapter(nn.Module):
    """
    Parallel Adapter following ContinualTTA implementation
    """
    def __init__(self, in_channels, bottleneck_ratio=32):
        super(ParallelAdapter, self).__init__()
        bottleneck_channels = max(1, in_channels // bottleneck_ratio)

        self.down_proj = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=True)
        self.non_linear = nn.ReLU(inplace=True)
        self.up_proj = nn.Conv2d(bottleneck_channels, in_channels, 1, bias=True)

        # Zero initialization for identity mapping at start
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        return self.up_proj(self.non_linear(self.down_proj(x)))

class ParallelAdapterWithProjection(nn.Module):
    """
    Parallel Adapter with projection to match output dimensions
    Takes conv2 output and projects to conv3 output dimension
    Exactly matches ContinualTTA's Adapter implementation
    """
    def __init__(self, in_channels, out_channels, bottleneck_ratio=32):
        super(ParallelAdapterWithProjection, self).__init__()
        bottleneck_channels = max(1, in_channels // bottleneck_ratio)

        self.down_proj = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=True)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=True)

        # Initialize exactly like ContinualTTA
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        return self.up_proj(self.non_linear_func(self.down_proj(x)))


class ConvTaskWrapper(nn.Module):
    """
    Wraps a Conv2d layer with parallel adapter, matching ContinualTTA's conv_task implementation.
    This wrapper adds adapter output to the conv output: y = norm(conv(x)) + adapter(x) * scalar
    """
    def __init__(self, original_conv, adapter_mode='parallel', r=32, scalar=1.0):
        super(ConvTaskWrapper, self).__init__()
        self.conv = original_conv  # Original Conv2d layer (with norm)
        self.mode = adapter_mode
        self.scalar = scalar if isinstance(scalar, torch.Tensor) else torch.tensor(scalar)

        if self.mode == 'parallel':
            # Get input and output channels from original conv
            in_channels = original_conv.in_channels if hasattr(original_conv, 'in_channels') else original_conv.conv.in_channels
            out_channels = original_conv.out_channels if hasattr(original_conv, 'out_channels') else original_conv.conv.out_channels

            # Create adapter: input -> bottleneck -> output (matching conv_task lines 102-119)
            self.down_proj = nn.Conv2d(in_channels, in_channels // r, kernel_size=1, stride=1, bias=True)
            self.non_linear_func = nn.ReLU()
            self.up_proj = nn.Conv2d(in_channels // r, out_channels, kernel_size=1, stride=1, bias=True)

            # Initialize like ContinualTTA (conv_task lines 107-110)
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        # Base output: norm(conv(x))
        y = self.conv(x)

        if self.mode == 'parallel':
            # Adapter output: adapter(x) * scalar (conv_task line 149-156)
            adapt_x = self.up_proj(self.non_linear_func(self.down_proj(x)))
            y = y + adapt_x * self.scalar.to(y.device)

        return y


class WHW:
    """
    WHW method for continual test-time adaptation using ContinualTTA's parallel adapter approach
    This implements ONLY the ContinualTTA parallel adapter mechanism without Mean-Teacher components.
    """

    def __init__(
        self,
        model,
        data_root,
        device=None,
        batch_size=4,
        learning_rate=0.0001,
        weight_decay=1e-4,
        momentum=0.9,
        adaptation_where="adapter",
        adapter_bottleneck_ratio=32,
        **kwargs
    ):
        self.model = model
        self.data_root = data_root
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.adaptation_where = adaptation_where
        self.adapter_bottleneck_ratio = adapter_bottleneck_ratio

        self.optimizer = None
        self.adapters = {}

        # ContinualTTA-style loss skipping parameters
        self.skip_redundant = kwargs.get('skip_redundant', None)  # None, 'stat', 'period', 'ema'
        self.skip_period = kwargs.get('skip_period', 10)
        self.skip_beta = kwargs.get('skip_beta', 1.2)  # EMA threshold ratio
        self.skip_tau = kwargs.get('skip_tau', 1.0)

        # EMA tracking for loss-based skipping
        self.loss_ema99 = 0.0
        self.loss_ema95 = 0.0
        self.loss_ema90 = 0.0
        self.adaptation_steps = 0
        self.used_steps = 0

        # ContinualTTA feature alignment parameters
        self.fg_align = kwargs.get('fg_align', 'KL')  # None, 'KL'
        self.gl_align = kwargs.get('gl_align', 'KL')  # None, 'KL', 'bn_stats'
        self.alpha_fg = kwargs.get('alpha_fg', 1.0)
        self.alpha_gl = kwargs.get('alpha_gl', 1.0)
        self.ema_gamma = kwargs.get('ema_gamma', 128)
        self.freq_weight = kwargs.get('freq_weight', False)

        # Source and target feature statistics (to be loaded)
        self.source_feat_stats = kwargs.get('source_feat_stats', None)
        self.s_stats = None  # Source statistics
        self.t_stats = {}    # Target statistics
        self.template_cov = {}
        self.ema_n = {}      # EMA counters for each class
        self.s_div = {}      # Source divergence stats for skipping

        # Get number of classes from model
        if hasattr(model, 'roi_heads') and hasattr(model.roi_heads, 'box_predictor'):
            self.num_classes = model.roi_heads.box_predictor.num_classes
        else:
            self.num_classes = 6  # Default for SHIFT dataset

    @classmethod
    def load(cls, model, data_root, **kwargs):
        """
        Factory method to create and initialize WHW with ContinualTTA parallel adapter approach

        Args:
            model: The model to adapt
            data_root: Path to the data root directory
            **kwargs: Additional parameters

        Returns:
            WHW: Initialized WHW instance
        """
        instance = cls(model=model, data_root=data_root, **kwargs)
        instance._setup()
        return instance

    def _setup(self):
        """Setup the WHW method with parallel adapters"""
        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)
        self._freeze_backbone()
        self._load_source_statistics()
        self._initialize_feature_stats()
        self._setup_parallel_adapters()
        self._patch_forward_box()
        self._patch_model_forward()  # Override model forward to match ContinualTTA
        self._setup_optimizer()
        self._setup_training_mode()

    def _freeze_backbone(self):
        """Freeze backbone parameters like ContinualTTA"""
        # Freeze all model parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        print("WHW: Frozen all backbone parameters")

    def _load_source_statistics(self):
        """Load or create source domain feature statistics"""
        # Try provided path first
        if self.source_feat_stats is not None and os.path.exists(self.source_feat_stats):
            print(f"WHW: Loading source statistics from {self.source_feat_stats}")
            self.s_stats = torch.load(self.source_feat_stats, map_location=self.device)
            return

        # Check same folder
        default_path = "whw_source_statistics_clear.pt"
        if os.path.exists(default_path):
            print(f"WHW: Found source statistics at {default_path}")
            self.s_stats = torch.load(default_path, map_location=self.device)
            return

        # Create new statistics
        print(f"WHW: Source statistics not found, creating new one at {default_path}")
        self.s_stats = self.collect_source_statistics(task="clear", output_path=default_path)

    def _create_dummy_source_stats(self):
        """Create dummy source statistics for testing when no real stats available"""
        dummy_stats = {
            "gl": {},
            "fg": {},
            "kl_div": {}
        }

        # Dummy global stats for backbone features (p2, p3, p4, p5, p6)
        for layer in ["p2", "p3", "p4", "p5", "p6"]:
            feat_dim = {"p2": 256, "p3": 256, "p4": 256, "p5": 256, "p6": 256}[layer]
            mean = torch.zeros(feat_dim)
            cov = torch.eye(feat_dim) * 0.1
            dummy_stats["gl"][layer] = (mean, cov)
            dummy_stats["kl_div"][layer] = 1.0

        # Dummy foreground stats for each class
        for k in range(self.num_classes):
            feat_dim = 1024  # ROI feature dimension
            mean = torch.zeros(feat_dim)
            cov = torch.eye(feat_dim) * 0.1
            dummy_stats["fg"][k] = (mean, cov)
            dummy_stats["kl_div"][k] = 1.0

        return dummy_stats

    def _initialize_feature_stats(self):
        """Initialize target feature statistics and template covariances"""
        if self.s_stats is None:
            return

        # Initialize global stats
        if self.gl_align is not None and self.gl_align == "KL":
            for k in self.s_stats["gl"]:
                mean, cov = self.s_stats["gl"][k]
                self.template_cov["gl"] = self.template_cov.get("gl", {})
                self.template_cov["gl"][k] = torch.eye(mean.shape[0]) * cov.max().item() / 30
                self.t_stats["gl"] = self.t_stats.get("gl", {})
                self.t_stats["gl"][k] = (mean.clone(), cov.clone())

        # Initialize foreground stats
        if self.fg_align is not None and self.fg_align == "KL":
            for k in self.s_stats["fg"]:
                mean, cov = self.s_stats["fg"][k]
                self.template_cov["fg"] = self.template_cov.get("fg", {})
                self.template_cov["fg"][k] = torch.eye(mean.shape[0]) * cov.max().item() / 30
                self.t_stats["fg"] = self.t_stats.get("fg", {})
                self.t_stats["fg"][k] = (mean.clone(), cov.clone())
                self.ema_n[k] = 0

        # Initialize divergence stats for skipping
        if "kl_div" in self.s_stats:
            self.s_div = self.s_stats["kl_div"]
        else:
            self.s_div = {k: 1.0 for k in range(self.num_classes)}

        print(f"WHW: Initialized feature statistics for {len(self.s_stats.get('gl', {}))} global and {len(self.s_stats.get('fg', {}))} foreground features")

    def _setup_parallel_adapters(self):
        """Add parallel adapters to ResNet backbone using ContinualTTA's add_adapter method"""
        if not (hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'bottom_up')):
            print("Warning: Model doesn't have expected ResNet backbone structure")
            return

        # Check if backbone has add_adapter method (ContinualTTA style)
        if hasattr(self.model.backbone.bottom_up, 'add_adapter'):
            # Use ContinualTTA's add_adapter method
            self.model.backbone.bottom_up.add_adapter(
                adapter="parallel",
                th=0.9,
                scalar=None,
                r=self.adapter_bottleneck_ratio
            )
            print("WHW: Added adapters using ContinualTTA's add_adapter method")

            # Collect adapter parameters for optimizer
            for stage in self.model.backbone.bottom_up.stages:
                for block_idx, block in enumerate(stage):
                    if hasattr(block, 'adapter'):
                        adapter_name = f"stage_{len(self.adapters)}_block{block_idx}_adapter"
                        self.adapters[adapter_name] = block.adapter
        else:
            # Fallback: manual monkey patching (current implementation)
            print("WHW: Falling back to manual adapter patching")
            self._setup_manual_adapters()

        print(f"WHW: Total adapters added: {len(self.adapters)}")

    def _setup_manual_adapters(self):
        """
        Wrap conv1 layers with ConvTaskWrapper to match ContinualTTA's add_adapter implementation.
        This follows the exact pattern from resnet.py lines 760-767 where conv1 is wrapped with conv_task.
        """
        # Add adapters to ResNet bottleneck blocks (following ContinualTTA pattern)
        for stage_name in ['res2', 'res3', 'res4', 'res5']:
            if hasattr(self.model.backbone.bottom_up, stage_name):
                stage = getattr(self.model.backbone.bottom_up, stage_name)
                for block_idx, block in enumerate(stage):
                    if hasattr(block, 'conv1') and hasattr(block, 'conv2') and hasattr(block, 'conv3'):
                        # Only wrap conv1 in blocks with stride=1 (matching line 762: if idx >= 0 and block.conv1.stride[0] == 1)
                        # For BottleneckBlock, conv1 can have stride in stride_in_1x1 mode, but conv2 usually has stride
                        # We check conv1 stride to match the original logic
                        if hasattr(block.conv1, 'stride') and block.conv1.stride == (1, 1):
                            adapter_name = f"{stage_name}_block{block_idx}_conv1_adapter"

                            # Wrap conv1 with ConvTaskWrapper (matching conv_task wrapping)
                            wrapped_conv1 = ConvTaskWrapper(
                                original_conv=block.conv1,
                                adapter_mode='parallel',
                                r=self.adapter_bottleneck_ratio,
                                scalar=1.0
                            )
                            wrapped_conv1.to(self.device)

                            # Replace conv1 with wrapped version
                            block.conv1 = wrapped_conv1

                            # Store the wrapper (which contains the adapter) for optimizer
                            self.adapters[adapter_name] = wrapped_conv1

                            print(f"WHW: Wrapped {stage_name} block{block_idx} conv1 with adapter")

        # If no adapters were added (all blocks have stride != 1), fall back to the old approach
        if len(self.adapters) == 0:
            print("WHW: No stride-1 conv1 blocks found, using alternative adapter approach")
            self._setup_manual_adapters_fallback()

    def _setup_manual_adapters_fallback(self):
        """
        Fallback method: add adapters using the old block forward patching approach.
        This is used when no stride-1 conv1 blocks are found.
        """
        for stage_name in ['res2', 'res3', 'res4', 'res5']:
            if hasattr(self.model.backbone.bottom_up, stage_name):
                stage = getattr(self.model.backbone.bottom_up, stage_name)
                for block_idx, block in enumerate(stage):
                    if hasattr(block, 'conv2'):  # BottleneckBlock
                        # Get input and output channels for adapter
                        conv2_channels = block.conv2.out_channels  # bottleneck channels
                        conv3_channels = block.conv3.out_channels  # final output channels
                        adapter_name = f"{stage_name}_block{block_idx}_adapter"

                        # Create and add adapter that converts conv2 output to conv3 output size
                        adapter = ParallelAdapterWithProjection(conv2_channels, conv3_channels, self.adapter_bottleneck_ratio)
                        adapter.to(self.device)

                        # Store adapter
                        self.adapters[adapter_name] = adapter

                        # Monkey patch the block's forward method
                        self._patch_block_forward(block, adapter)

    def _patch_block_forward(self, block, adapter):
        """Patch block's forward method to include parallel adapter"""
        original_forward = block.forward

        def forward_with_adapter(x):
            # Detectron2 BottleneckBlock structure: conv layers include norm, activation is F.relu_

            # Conv1 (includes norm)
            out = block.conv1(x)
            out = F.relu_(out)

            # Conv2 (includes norm)
            out_tmp = block.conv2(out)  # Store conv2 output like ContinualTTA
            out = F.relu_(out_tmp)

            # Conv3 (includes norm, no activation yet)
            out = block.conv3(out)

            # Shortcut connection
            if block.shortcut is not None:
                shortcut = block.shortcut(x)
            else:
                shortcut = x

            # Apply parallel adapter exactly like ContinualTTA: out + shortcut + adapter(out_tmp)
            if hasattr(block, 'has_adapter') and block.has_adapter:
                out = out + shortcut + block.adapter(out_tmp)
            else:
                out = out + shortcut

            # Final activation
            out = F.relu_(out)
            return out

        # Store adapter reference in block and replace forward
        block.adapter = adapter  # Use 'adapter' like ContinualTTA, not '_adapter'
        block.has_adapter = True  # Add has_adapter flag like ContinualTTA
        block.forward = forward_with_adapter

    def _patch_forward_box(self):
        """Patch roi_heads._forward_box to support ContinualTTA-style outs parameter"""
        roi_heads = self.model.roi_heads
        original_forward_box = roi_heads._forward_box

        def _forward_box_with_outs(features, proposals, targets=None, outs=False):
            """Modified _forward_box that returns box_features and predictions when outs=True"""
            features_list = [features[f] for f in roi_heads.box_in_features]
            box_features = roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
            box_features = roi_heads.box_head(box_features)
            predictions = roi_heads.box_predictor(box_features)

            if roi_heads.training:
                losses = roi_heads.box_predictor.losses(predictions, proposals)
                if roi_heads.train_on_pred_boxes:
                    with torch.no_grad():
                        pred_boxes = roi_heads.box_predictor.predict_boxes_for_gt_classes(
                            predictions, proposals
                        )
                        for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                            proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
                return losses
            else:
                pred_instances, _ = roi_heads.box_predictor.inference(predictions, proposals)
                if outs:
                    return pred_instances, predictions, box_features
                else:
                    return pred_instances

        # Replace the method
        roi_heads._forward_box = _forward_box_with_outs

    def _patch_model_forward(self):
        """
        Patch model's forward to return (results, adapt_loss, feature_sim) like ContinualTTA
        This makes the model behave exactly like ContinualTTA's GeneralizedRCNN.adapt()
        """
        original_forward = self.model.forward

        def continual_tta_forward(batched_inputs):
            """ContinualTTA-style forward that returns adapt_loss"""
            # This is essentially ContinualTTA's adapt() method
            images = self.model.preprocess_image(batched_inputs)
            features = self.model.backbone(images.tensor)

            if isinstance(features, tuple):
                features = features[0]

            adapt_loss = {}
            feature_sim = {}

            # Set to eval mode for proposal/roi_heads (ContinualTTA style)
            self.model.roi_heads.training = False
            self.model.proposal_generator.training = False

            # Get proposals
            proposals, _ = self.model.proposal_generator(images, features, None)

            # Get box features and predictions
            pred_instances, predictions, box_features = self.model.roi_heads._forward_box(features, proposals, outs=True)
            results = self.model.roi_heads.forward_with_given_boxes(features, pred_instances)

            # Compute foreground alignment (exactly like ContinualTTA line 304-349)
            if self.fg_align is not None:
                _scores = nn.Softmax(dim=1)(predictions[0])
                bg_scores = _scores[:, -1]
                fg_scores, fg_preds = _scores[:, :-1].max(dim=1)

                valid = fg_scores >= 0.5
                fg_preds[~valid] = torch.ones((~valid).sum()).long().to(valid.device) * self.num_classes
                fg_scores[~valid] = bg_scores[~valid]

                loss_fg_align = 0
                loss_n = 0

                for _k in fg_preds[fg_preds != self.num_classes].unique():
                    k = _k.item()
                    if (fg_preds == k).sum() > 0:  # ContinualTTA line 318 - only check sum
                        cur_feats = box_features[fg_preds == k]
                        self.ema_n[k] += cur_feats.shape[0]

                        diff = cur_feats - self.t_stats["fg"][k][0][None, :].to(self.device)
                        delta = 1 / self.ema_gamma * diff.sum(dim=0)
                        cur_target_mean = self.t_stats["fg"][k][0].to(self.device) + delta

                        try:
                            t_dist = torch.distributions.MultivariateNormal(
                                cur_target_mean,
                                self.s_stats["fg"][k][1].to(self.device) + self.template_cov["fg"][k].to(self.device)
                            )
                            s_dist = torch.distributions.MultivariateNormal(
                                self.s_stats["fg"][k][0].to(self.device),
                                self.s_stats["fg"][k][1].to(self.device) + self.template_cov["fg"][k].to(self.device)
                            )

                            # ContinualTTA line 334-341: freq_weight option
                            if self.freq_weight:
                                class_weight = np.log((max([self.ema_n[_k] for _k in range(self.num_classes)]) / self.ema_n[k]))
                                cur_loss_fg_align = min(class_weight + 0.01, 10) ** 2 * \
                                                    ((torch.distributions.kl.kl_divergence(s_dist, t_dist) +
                                                      torch.distributions.kl.kl_divergence(t_dist, s_dist)) / 2)
                            else:
                                cur_loss_fg_align = (torch.distributions.kl.kl_divergence(s_dist, t_dist) +
                                                   torch.distributions.kl.kl_divergence(t_dist, s_dist)) / 2

                            if cur_loss_fg_align < 10**5:
                                loss_fg_align += cur_loss_fg_align
                                self.t_stats["fg"][k] = (cur_target_mean.detach(), None)
                                loss_n += 1
                        except:
                            pass

                if loss_n > 0:
                    adapt_loss["fg_align"] = self.alpha_fg * loss_fg_align

            # Compute global alignment (exactly like ContinualTTA line 351-379)
            if self.gl_align is not None:
                loss_gl_align = 0
                if self.gl_align == "KL":
                    for k in features.keys():
                        if k in self.t_stats.get("gl", {}):
                            cur_feats = features[k].mean(dim=[2, 3])
                            diff = cur_feats - self.t_stats["gl"][k][0][None, :].to(self.device)
                            delta = 1 / self.ema_gamma * diff.sum(dim=0)
                            cur_target_mean = self.t_stats["gl"][k][0].to(self.device) + delta

                            try:
                                t_dist = torch.distributions.MultivariateNormal(
                                    cur_target_mean,
                                    self.s_stats["gl"][k][1].to(self.device) + self.template_cov["gl"][k].to(self.device)
                                )
                                s_dist = torch.distributions.MultivariateNormal(
                                    self.s_stats["gl"][k][0].to(self.device),
                                    self.s_stats["gl"][k][1].to(self.device) + self.template_cov["gl"][k].to(self.device)
                                )

                                cur_loss_gl_align = (torch.distributions.kl.kl_divergence(s_dist, t_dist) +
                                                   torch.distributions.kl.kl_divergence(t_dist, s_dist)) / 2

                                if cur_loss_gl_align < 10**5:
                                    loss_gl_align += cur_loss_gl_align
                                    self.t_stats["gl"][k] = (cur_target_mean.detach(), None)
                            except:
                                pass

                adapt_loss["global_align"] = self.alpha_gl * loss_gl_align

            # Postprocess (ContinualTTA line 381-384)
            processed_results = self.model._postprocess(results, batched_inputs, images.image_sizes)

            # Return in ContinualTTA format: (results, adapt_loss, feature_sim)
            return processed_results, adapt_loss, feature_sim

        # Replace model's forward
        self.model.forward = continual_tta_forward

    def _setup_optimizer(self):
        """Setup optimizer for ContinualTTA adaptation"""
        params = []
        if self.learning_rate > 0:
            if self.adaptation_where == "adapter":
                # Only adapt adapter parameters (ContinualTTA default)
                for name, adapter_module in self.adapters.items():
                    # For ConvTaskWrapper, only train the adapter parts (down_proj, up_proj)
                    if isinstance(adapter_module, ConvTaskWrapper):
                        # Freeze original conv
                        for param in adapter_module.conv.parameters():
                            param.requires_grad = False
                        # Enable adapter parameters
                        if hasattr(adapter_module, 'down_proj'):
                            for param in adapter_module.down_proj.parameters():
                                param.requires_grad = True
                                params.append(param)
                        if hasattr(adapter_module, 'up_proj'):
                            for param in adapter_module.up_proj.parameters():
                                param.requires_grad = True
                                params.append(param)
                    else:
                        # For other adapter types, train all parameters
                        adapter_module.requires_grad_(True)
                        params += list(adapter_module.parameters())

            elif self.adaptation_where == "full":
                # Adapt all parameters (backbone + adapters)
                for param in self.model.parameters():
                    param.requires_grad = True
                    params.append(param)
                # Add adapter parameters (already included above, but ensure they're trainable)
                for name, adapter_module in self.adapters.items():
                    if isinstance(adapter_module, ConvTaskWrapper):
                        if hasattr(adapter_module, 'down_proj'):
                            for param in adapter_module.down_proj.parameters():
                                param.requires_grad = True
                        if hasattr(adapter_module, 'up_proj'):
                            for param in adapter_module.up_proj.parameters():
                                param.requires_grad = True
                    else:
                        adapter_module.requires_grad_(True)

            elif self.adaptation_where == "normalization":
                # Adapt normalization layers + adapters
                for name, module in self.model.named_modules():
                    if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d, nn.LayerNorm)):
                        if hasattr(module, 'weight') and module.weight is not None:
                            module.weight.requires_grad = True
                            params.append(module.weight)
                        if hasattr(module, 'bias') and module.bias is not None:
                            module.bias.requires_grad = True
                            params.append(module.bias)

                # Add adapter parameters
                for name, adapter_module in self.adapters.items():
                    if isinstance(adapter_module, ConvTaskWrapper):
                        # Freeze original conv
                        for param in adapter_module.conv.parameters():
                            param.requires_grad = False
                        # Enable adapter parameters
                        if hasattr(adapter_module, 'down_proj'):
                            for param in adapter_module.down_proj.parameters():
                                param.requires_grad = True
                                params.append(param)
                        if hasattr(adapter_module, 'up_proj'):
                            for param in adapter_module.up_proj.parameters():
                                param.requires_grad = True
                                params.append(param)
                    else:
                        adapter_module.requires_grad_(True)
                        params += list(adapter_module.parameters())

        print(f"WHW: Adapting {len(params)} parameters with strategy '{self.adaptation_where}'")
        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        print(f"WHW: Total trainable parameters: {trainable_params:,}")

        if params:
            self.optimizer = optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay
            )
        else:
            self.optimizer = None
            print("Warning: No parameters selected for adaptation!")

    def _setup_training_mode(self):
        """Setup training mode for ContinualTTA"""
        # Set model components to training mode
        self.model.training = True
        if hasattr(self.model, 'proposal_generator'):
            self.model.proposal_generator.training = True
        if hasattr(self.model, 'roi_heads'):
            self.model.roi_heads.training = True

        # Set adapters to training mode
        for adapter in self.adapters.values():
            adapter.train()

        # Set online_adapt flag
        self.model.online_adapt = True

    def adapt_single_batch(self, batch):
        """
        Adapt single batch using ContinualTTA's approach
        Now using the patched forward that returns (results, adapt_loss, feature_sim)
        """
        self.model.train()  # Keep in training mode for adaptation

        if self.optimizer is not None:
            self.optimizer.zero_grad()

            # Forward pass - returns (results, adapt_loss, feature_sim) like ContinualTTA
            _, adapt_loss, feature_sim = self.model(batch)

            if adapt_loss:
                total_loss = sum(adapt_loss.values())
                self.adaptation_steps += 1

                # ContinualTTA-style loss-based skipping logic
                cur_used = self._should_adapt_continual_tta(total_loss, adapt_loss)

                if total_loss > 0 and cur_used:
                    total_loss.backward()

                    # Clip gradients like ContinualTTA
                    if hasattr(self.model, 'backbone'):
                        torch.nn.utils.clip_grad_norm_(self.model.backbone.parameters(), 1.0)

                    # Clip adapter gradients
                    for adapter in self.adapters.values():
                        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)

                    self.optimizer.step()
                    self.used_steps += 1

                    # Only print occasionally to avoid slowing down
                    if self.used_steps % 10 == 0:
                        loss_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in adapt_loss.items()])
                        print(f"WHW: Used {self.used_steps}/{self.adaptation_steps} steps - {loss_str}")

                # Update EMA for loss tracking
                if "global_align" in adapt_loss:
                    self._update_loss_ema(adapt_loss["global_align"].item())

        # Get final outputs for evaluation (call forward again)
        with torch.no_grad():
            final_outputs, _, _ = self.model(batch)

        return final_outputs

    def _compute_adaptation_loss(self, batch):
        """
        Compute ContinualTTA adaptation losses (foreground + global alignment)
        This replicates the adapt() method from ContinualTTA exactly
        """
        adapt_loss = {}
        feature_sim = {}

        # Set training mode for ROI heads but eval for proposal generator
        self.model.roi_heads.training = False
        self.model.proposal_generator.training = False

        # Preprocess images
        images = self.model.preprocess_image(batch)

        # Get backbone features
        features = self.model.backbone(images.tensor)
        if isinstance(features, tuple):
            features = features[0]

        # Get proposals
        proposals, _ = self.model.proposal_generator(images, features, None)

        # Get box features and predictions for foreground alignment
        pred_instances, predictions, box_features = self.model.roi_heads._forward_box(features, proposals, outs=True)

        # Compute foreground alignment loss
        if self.fg_align is not None:
            fg_loss = self._compute_foreground_alignment_loss(predictions, box_features)
            if fg_loss is not None:
                adapt_loss["fg_align"] = fg_loss

        # Compute global alignment loss
        if self.gl_align is not None:
            gl_loss = self._compute_global_alignment_loss(features)
            if gl_loss is not None:
                adapt_loss["global_align"] = gl_loss

        return adapt_loss, feature_sim

    def _compute_foreground_alignment_loss(self, predictions, box_features):
        """Compute class-wise foreground feature alignment loss"""
        if self.fg_align != "KL" or not self.t_stats.get("fg"):
            return None

        _scores = nn.Softmax(dim=1)(predictions[0])
        bg_scores = _scores[:, -1]
        fg_scores, fg_preds = _scores[:, :-1].max(dim=1)

        # Filter by confidence threshold (like ContinualTTA)
        valid = fg_scores >= 0.5
        fg_preds[~valid] = torch.ones((~valid).sum()).long().to(valid.device) * self.num_classes
        fg_scores[~valid] = bg_scores[~valid]

        loss_fg_align = 0
        loss_n = 0

        for _k in fg_preds[fg_preds != self.num_classes].unique():
            k = _k.item()
            if k >= len(self.t_stats["fg"]) or (fg_preds == k).sum() == 0:
                continue

            cur_feats = box_features[fg_preds == k]
            self.ema_n[k] += cur_feats.shape[0]

            # Update target statistics with EMA
            diff = cur_feats - self.t_stats["fg"][k][0][None, :].to(self.device)
            delta = 1 / self.ema_gamma * diff.sum(dim=0)
            cur_target_mean = self.t_stats["fg"][k][0].to(self.device) + delta

            # Compute KL divergence between source and target distributions
            try:
                t_dist = torch.distributions.MultivariateNormal(
                    cur_target_mean,
                    self.s_stats["fg"][k][1].to(self.device) + self.template_cov["fg"][k].to(self.device)
                )
                s_dist = torch.distributions.MultivariateNormal(
                    self.s_stats["fg"][k][0].to(self.device),
                    self.s_stats["fg"][k][1].to(self.device) + self.template_cov["fg"][k].to(self.device)
                )

                # Symmetric KL divergence
                cur_loss_fg_align = (torch.distributions.kl.kl_divergence(s_dist, t_dist) +
                                   torch.distributions.kl.kl_divergence(t_dist, s_dist)) / 2

                if cur_loss_fg_align < 10**5:
                    loss_fg_align += cur_loss_fg_align
                    self.t_stats["fg"][k] = (cur_target_mean.detach(), None)
                    loss_n += 1

            except Exception as e:
                # Skip this class if covariance is not positive definite
                continue

        if loss_n > 0:
            return self.alpha_fg * loss_fg_align  # Don't divide by loss_n (ContinualTTA style)
        return None

    def _compute_global_alignment_loss(self, features):
        """Compute global backbone feature alignment loss"""
        if self.gl_align != "KL" or not self.t_stats.get("gl"):
            return None

        loss_gl_align = 0
        loss_n = 0

        for k in features.keys():
            if k not in self.t_stats["gl"]:
                continue

            # Get global features by spatial averaging
            cur_feats = features[k].mean(dim=[2, 3])

            # Update target statistics with EMA
            diff = cur_feats - self.t_stats["gl"][k][0][None, :].to(self.device)
            delta = 1 / self.ema_gamma * diff.sum(dim=0)
            cur_target_mean = self.t_stats["gl"][k][0].to(self.device) + delta

            # Compute KL divergence
            try:
                t_dist = torch.distributions.MultivariateNormal(
                    cur_target_mean,
                    self.s_stats["gl"][k][1].to(self.device) + self.template_cov["gl"][k].to(self.device)
                )
                s_dist = torch.distributions.MultivariateNormal(
                    self.s_stats["gl"][k][0].to(self.device),
                    self.s_stats["gl"][k][1].to(self.device) + self.template_cov["gl"][k].to(self.device)
                )

                cur_loss_gl_align = (torch.distributions.kl.kl_divergence(s_dist, t_dist) +
                                   torch.distributions.kl.kl_divergence(t_dist, s_dist)) / 2

                if cur_loss_gl_align < 10**5:
                    loss_gl_align += cur_loss_gl_align
                    self.t_stats["gl"][k] = (cur_target_mean.detach(), None)
                    loss_n += 1

            except Exception as e:
                # Skip this layer if covariance is not positive definite
                continue

        return self.alpha_gl * loss_gl_align  # ContinualTTA doesn't check loss_n for global

    def _should_adapt_continual_tta(self, total_loss, adapt_loss):
        """
        ContinualTTA's exact skipping logic using global_align loss
        """
        if self.skip_redundant is None:
            return True

        # Use global_align loss for skipping decisions like ContinualTTA
        if "global_align" not in adapt_loss:
            return True

        loss_value = adapt_loss["global_align"].item()

        # Calculate divergence threshold
        div_thr = 2 * sum(self.s_div.values()) * self.skip_tau if self.skip_redundant is not None else 2 * sum(self.s_div.values())

        cur_used = False

        # Period-based skipping
        if 'period' in self.skip_redundant and self.adaptation_steps % self.skip_period == 0:
            cur_used = True

        # Statistical threshold-based skipping
        elif 'stat' in self.skip_redundant and loss_value > div_thr:
            cur_used = True

        # EMA-based skipping
        elif 'ema' in self.skip_redundant and self.loss_ema99 > 0:
            ema_ratio = loss_value / (self.loss_ema99 + 1e-7)
            if ema_ratio > self.skip_beta:
                cur_used = True

        return cur_used

    def _should_adapt(self, total_loss, loss_dict):
        """
        ContinualTTA-style loss-based skipping logic
        Decides whether to perform adaptation step based on loss characteristics
        """
        if self.skip_redundant is None:
            return True

        # Convert total_loss to float for comparison
        loss_value = total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)

        # Period-based skipping
        if 'period' in self.skip_redundant and self.adaptation_steps % self.skip_period == 0:
            return True

        # EMA-based skipping (similar to ContinualTTA's logic)
        if 'ema' in self.skip_redundant and self.loss_ema99 > 0:
            ema_ratio = loss_value / (self.loss_ema99 + 1e-7)
            if ema_ratio > self.skip_beta:
                return True

        # Statistical threshold-based skipping (simplified version)
        if 'stat' in self.skip_redundant:
            # Use a simple threshold based on recent loss history
            threshold = self.loss_ema95 * self.skip_tau
            if loss_value > threshold:
                return True

        return False

    def _update_loss_ema(self, loss_value):
        """Update exponential moving averages for loss tracking"""
        self.loss_ema99 = 0.99 * self.loss_ema99 + 0.01 * loss_value
        self.loss_ema95 = 0.95 * self.loss_ema95 + 0.05 * loss_value
        self.loss_ema90 = 0.9 * self.loss_ema90 + 0.1 * loss_value

    def evaluate_task(self, task=None, dataset=None, dataloader=None, threshold=0.0):
        """Evaluate on a specific task using ContinualTTA parallel adapters"""
        if dataloader is None:
            raise ValueError("dataloader must be provided for accurate FPS measurement")

        task_name = task if task is not None else "unknown"
        print(f"Starting WHW evaluation on {task_name}")

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
        total_batches = len(dataloader) if hasattr(dataloader, '__len__') else getattr(dataloader, 'valid_len', None)

        for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches, desc=f"WHW {task_name}")):
            start_time = time.time()
            outputs = self.adapt_single_batch(batch)
            inference_time += time.time() - start_time

            for i, (output, input_data) in enumerate(zip(outputs, batch)):
                instances = output['instances']
                mask = instances.scores > threshold

                pred_detection = Detections(
                    xyxy=instances.pred_boxes.tensor[mask].detach().cpu().numpy(),
                    class_id=instances.pred_classes[mask].detach().cpu().numpy(),
                    confidence=instances.scores[mask].detach().cpu().numpy()
                )
                gt_instances = input_data['instances']
                target_detection = Detections(
                    xyxy=gt_instances.gt_boxes.tensor.detach().cpu().numpy(),
                    class_id=gt_instances.gt_classes.detach().cpu().numpy()
                )

                predictions_list.append(pred_detection)
                targets_list.append(target_detection)

        map_metric.update(predictions=predictions_list, targets=targets_list)
        print(f"Computing mAP for {task_name}")
        m_ap = map_metric.compute()

        per_class_map = {}
        if CLASSES is not None:
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
        """Prepare dataloaders for all tasks in advance"""
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = {}
        print("Preparing dataloaders for all tasks...")

        for task in tasks:
            print(f"Loading dataset for {task}...")
            dataset = SHIFTCorruptedDatasetForObjectDetection(
                root=self.data_root, valid=True,
                transform=datasets.detectron_image_transform,
                transforms=datasets.default_valid_transforms,
                task=task
            )

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )
            dataloader.valid_len = math.ceil(len(dataset) / self.batch_size)

            dataloaders[task] = (dataset, dataloader)
            print(f"  - {task}: {len(dataset)} samples, {dataloader.valid_len} batches")

        print(f"All {len(dataloaders)} dataloaders prepared!")
        return dataloaders

    def collect_source_statistics(self, task="clear", num_samples=1000, iou_threshold=0.5, output_path=None):
        """
        Collect source domain feature statistics (ContinualTTA style)

        Args:
            task: Source domain task (default: "clear" for SHIFT)
            num_samples: Number of samples to collect
            iou_threshold: IoU threshold for matching proposals with GT
            output_path: Path to save statistics
        """
        print(f"Collecting source statistics from {task} domain...")

        # Prepare dataloader
        dataset = SHIFTClearDatasetForObjectDetection(
            root=self.data_root, train=True,
            transform=datasets.detectron_image_transform,
            transforms=datasets.default_valid_transforms
        )

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        loader.train_len = math.ceil(len(dataset)/self.batch_size)

        # Collections
        gl_features = {}  # Global features from backbone
        fg_features = {}  # Foreground features from ROI heads
        iou_with_gt = {}  # IoU with ground truth

        print(f"Collecting features from {len(dataset)} samples...")
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc="Collecting features")):
                # Get images and GT
                images = self.model.preprocess_image(batch)

                # Get backbone features
                features = self.model.backbone(images.tensor)
                if isinstance(features, tuple):
                    features = features[0]

                # Collect global features (spatial average)
                for k in features.keys():
                    cur_feats = features[k].mean(dim=[2, 3]).detach()
                    if k not in gl_features:
                        gl_features[k] = cur_feats
                    else:
                        gl_features[k] = torch.cat([gl_features[k], cur_feats], dim=0)

                # Get GT instances (already created by collate_fn)
                gt_instances = [input_data["instances"].to(self.device) for input_data in batch]

                # Get proposals
                proposals, _ = self.model.proposal_generator(images, features, None)

                # Get box features
                roi_heads = self.model.roi_heads
                features_list = [features[f] for f in roi_heads.box_in_features]
                box_features = roi_heads.box_pooler(features_list, [x.proposal_boxes for x in proposals])
                box_features = roi_heads.box_head(box_features).detach()

                # Match proposals with GT
                cur_iou_with_gt = torch.Tensor([]).to(self.device)
                gt_label = torch.Tensor([]).to(self.device)

                for i, p in enumerate(proposals):
                    if len(gt_instances[i]) > 0:
                        _iou_with_gt, _gt_label_idx = pairwise_iou(p.proposal_boxes, gt_instances[i].gt_boxes).max(dim=1)
                        cur_iou_with_gt = torch.cat([cur_iou_with_gt, _iou_with_gt])
                        _gt_label = torch.where(_iou_with_gt > iou_threshold,
                                               gt_instances[i].gt_classes[_gt_label_idx],
                                               self.num_classes)
                        gt_label = torch.cat([gt_label, _gt_label])
                    else:
                        cur_iou_with_gt = torch.cat([cur_iou_with_gt, torch.zeros(len(p)).to(self.device)])
                        gt_label = torch.cat([gt_label, torch.ones(len(p), dtype=torch.long).to(self.device) * self.num_classes])

                gt_label = gt_label.type(torch.long)

                # Collect foreground features per class
                for _gt in torch.unique(gt_label):
                    gt = _gt.item()
                    if gt >= self.num_classes:  # Skip background
                        continue

                    pick_idx = torch.randperm((gt_label == gt).sum())[:100]
                    cur_feats = box_features[gt_label == gt][pick_idx]
                    cur_iou = cur_iou_with_gt[gt_label == gt][pick_idx]

                    if gt not in fg_features:
                        fg_features[gt] = cur_feats
                        iou_with_gt[gt] = cur_iou
                    else:
                        fg_features[gt] = torch.cat([fg_features[gt], cur_feats])
                        iou_with_gt[gt] = torch.cat([iou_with_gt[gt], cur_iou])

        # Compute statistics
        print("Computing statistics...")
        stats = {
            "gl": {},
            "fg": {},
            "kl_div": {}
        }

        # Global statistics
        for k in gl_features.keys():
            feats = gl_features[k].cpu()
            mean = feats.mean(dim=0)
            cov = torch.from_numpy(np.cov(feats.T.numpy())).float()
            # Regularize covariance
            cov = cov + torch.eye(cov.shape[0]) * 1e-4
            stats["gl"][k] = (mean, cov)
            print(f"  GL[{k}]: mean shape {mean.shape}, cov shape {cov.shape}")

        # Foreground statistics
        for k in range(self.num_classes):
            if k in fg_features:
                feats = fg_features[k].cpu()
                mean = feats.mean(dim=0)
                cov = torch.from_numpy(np.cov(feats.T.numpy())).float()
                # Regularize covariance
                cov = cov + torch.eye(cov.shape[0]) * 1e-4
                stats["fg"][k] = (mean, cov)
                stats["kl_div"][k] = 1.0
                print(f"  FG[{k}]: {len(feats)} samples, mean shape {mean.shape}, cov shape {cov.shape}")
            else:
                print(f"  FG[{k}]: No samples collected")

        # Save statistics
        if output_path is None:
            output_path = f"whw_source_statistics_{task}.pt"

        torch.save(stats, output_path)
        print(f"Saved source statistics to {output_path}")

        return stats

    def evaluate_all_tasks(self, tasks=None):
        """Evaluate on all tasks using ContinualTTA parallel adapters"""
        if tasks is None:
            tasks = ["cloudy", "overcast", "foggy", "rainy", "dawn", "night", "clear"]

        dataloaders = self.prepare_dataloaders(tasks)

        results = {}
        for task in tasks:
            if task not in dataloaders:
                print(f"Warning: No dataloader found for task {task}, skipping...")
                continue

            dataset, dataloader = dataloaders[task]
            results[task] = self.evaluate_task(
                task=task,
                dataset=dataset,
                dataloader=dataloader
            )
            print(f"Results for {task}: {results[task]}")

        return results


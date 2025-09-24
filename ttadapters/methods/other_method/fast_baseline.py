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

import os
os.chdir("/workspace/ptta") # os.chdir("/home/ubuntu/test-time-adapters")

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
from ttadapters.models.rcnn import FasterRCNNForObjectDetection, SwinRCNNForObjectDetection
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
        learning_rate=0.001,
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

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
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
                shuffle=True,
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
                shuffle=True,
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
                shuffle=True,
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
                shuffle=True,
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
    Mean-Teacher method for test-time adaptation with FixMatch-style augmentation
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
        fixmatch_n=2,
        fixmatch_m=5
    ):
        self.model = model
        self.data_root = data_root
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.ema_alpha = ema_alpha
        self.conf_threshold = conf_threshold
        self.fixmatch_n = fixmatch_n
        self.fixmatch_m = fixmatch_m

        self.teacher_model = None
        self.optimizer = None
        self.fixmatch_augment = None

    @classmethod
    def load(cls, model, data_root, **kwargs):
        instance = cls(model=model, data_root=data_root, **kwargs)
        instance._setup()
        return instance

    def _setup(self):
        self.model.to(self.device)
        self._setup_teacher_model()
        self._setup_fixmatch_augment()

    def _setup_teacher_model(self):
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

        params = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d)):
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.requires_grad = True
                    params.append(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.requires_grad = True
                    params.append(module.bias)

        self.optimizer = optim.SGD(params, lr=self.learning_rate, momentum=0.9) if params else None

    def _setup_fixmatch_augment(self):
        self.fixmatch_augment = self._create_fixmatch_augment()

    def _create_fixmatch_augment(self):
        class FixMatchAugment:
            def __init__(self, n=2, m=5):
                self.n = n
                self.m = m

            def __call__(self, img):
                if isinstance(img, torch.Tensor):
                    img = T.ToPILImage()(img)

                for _ in range(self.n):
                    if random.random() < 0.5:
                        if random.random() < 0.5:
                            img = PIL.ImageEnhance.Brightness(img).enhance(0.5 + random.random() * 0.5)
                        if random.random() < 0.5:
                            img = PIL.ImageEnhance.Contrast(img).enhance(0.5 + random.random() * 0.5)

                if not isinstance(img, torch.Tensor):
                    img = T.ToTensor()(img)
                return img

        return FixMatchAugment(n=self.fixmatch_n, m=self.fixmatch_m)

    def _update_teacher_ema(self):
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_model.parameters(), self.model.parameters()):
                if teacher_param.requires_grad or student_param.requires_grad:
                    teacher_param.data = self.ema_alpha * teacher_param.data + (1 - self.ema_alpha) * student_param.data

    def _apply_fixmatch_augmentation(self, batch):
        weak_batch = []
        strong_batch = []

        for item in batch:
            weak_item = copy.deepcopy(item)
            weak_batch.append(weak_item)

            strong_item = copy.deepcopy(item)
            try:
                strong_item["image"] = self.fixmatch_augment(strong_item["image"])
            except Exception:
                strong_item = copy.deepcopy(weak_item)
            strong_batch.append(strong_item)

        return weak_batch, strong_batch

    def _set_pseudo_labels(self, inputs, outputs):
        new_inputs = []
        for inp, oup in zip(inputs, outputs):
            instances = oup['instances']
            high_conf_mask = instances.scores > self.conf_threshold
            high_conf_instances = instances[high_conf_mask]

            if len(high_conf_instances) == 0:
                continue

            new_inp = {k: inp[k] for k in inp if k not in ['instances']}
            new_instances = Instances(inp['instances'].image_size)
            new_instances.gt_classes = high_conf_instances.pred_classes
            new_instances.gt_boxes = high_conf_instances.pred_boxes
            new_inp['instances'] = new_instances
            new_inputs.append(new_inp)

        return new_inputs

    def adapt_single_batch(self, batch):
        weak_batch, strong_batch = self._apply_fixmatch_augmentation(batch)

        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(weak_batch)

        if self.optimizer is not None:
            pseudo_labeled_batch = self._set_pseudo_labels(strong_batch, teacher_outputs)

            if len(pseudo_labeled_batch) > 0:
                self.model.train()

                if hasattr(self.model, 'backbone'):
                    self.model.backbone.train()
                if hasattr(self.model, 'proposal_generator'):
                    self.model.proposal_generator.train()
                    self.model.proposal_generator.training = True
                if hasattr(self.model, 'roi_heads'):
                    self.model.roi_heads.train()
                    self.model.roi_heads.training = True

                for name, module in self.model.named_modules():
                    if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d)):
                        module.train()

                try:
                    self.optimizer.zero_grad()

                    with EventStorage() as storage:
                        model_output = self.model(pseudo_labeled_batch)

                    if isinstance(model_output, dict):
                        losses = model_output
                        total_detection_loss = sum([losses[k] for k in losses])

                        if total_detection_loss > 0:
                            total_detection_loss.backward()
                            self.optimizer.step()
                            self._update_teacher_ema()

                except Exception:
                    pass

        self.teacher_model.eval()
        with torch.no_grad():
            final_outputs = self.teacher_model(weak_batch)

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
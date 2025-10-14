
import sys
import os
from os import path
from argparse import ArgumentParser
from pathlib import Path
import copy
import warnings
from types import ModuleType
import math
from tqdm.auto import tqdm
from dataclasses import dataclass

import random
import numpy as np
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from detectron2.layers import FrozenBatchNorm2d
from detectron2.structures import Boxes, Instances, pairwise_iou
from transformers.models.rt_detr.modeling_rt_detr import RTDetrFrozenBatchNorm2d, RTDetrObjectDetectionOutput

from .utils import AverageMeter, SaveOutput, SaveOutputRTDETR


@dataclass
class ActMADConfig:
    model_type: str = "rcnn" # "swinrcnn", "yolo11", "rtdetr"
    adaptation_layers: str = "backbone+encoder" # "backbone", "encoder"
    
    data_root: str = './datasets'
    device: torch.device = torch.device("cuda")
    batch_size: int = 4
    clean_bn_extract_batch: int = 32

    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4 # 0.01
    optimizer_option: str = "SGD" # AdamW
    loss: nn.Module = nn.L1Loss(reduction="mean")

    clear_dataset: Dataset | None = None
    collate_fn: ModuleType | None = None
    discrete_scenario: ModuleType | None = None
    continuous_scenario: ModuleType | None = None

    statistic_save_path: Path = Path("./save_actmad_statistics.pt")


class ActMAD(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.cfg = config
        
        self.data_root = config.data_root
        self.device = config.device
        self.batch_size = config.batch_size
        self.clean_bn_extract_batch = config.clean_bn_extract_batch

        self.clean_mean_list_final: list[torch.Tensor] | None = None
        self.clean_var_list_final: list[torch.Tensor] | None = None
        self.layer_names: list[str] | None = None
        self.chosen_bn_layers: list[str] | None = None

        self.loss = config.loss

        if config.model_type in ("rtdetr", "yolo"):
            self.adaptation_layers = config.adaptation_layers
        
        self._setup()

    def _setup(self):
        # Basic setup
        self.model = copy.deepcopy(self.model)
        self.model.to(self.cfg.device)

        for k, v in self.model.named_parameters():
            v.requires_grad = True

        if self.cfg.optimizer_option == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.cfg.lr,
                # momentum=self.cfg.momentum,
                # weight_decay=self.cfg.weight_decay
            )

        elif self.cfg.optimizer_option == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                # lr=self.cfg.lr,
                # weight_decay=self.cfg.weight_decay
            )

        else :
            warnings.warn("Unknown optimizer_option.")
        
        # Extract statistics from the trained data. 
        # If a precomputed statistics file exists, load it instead.
        self._extract_or_load_clean_statistics()
    
    def _extract_or_load_clean_statistics(self):
        # Load existing statistics if available; otherwise, create new ones

        if self.cfg.statistic_save_path.exists():
            saved_stats = torch.load(self.cfg.statistic_save_path)
            self.clean_mean_list_final = saved_stats["clean_mean_list_final"]
            self.clean_var_list_final = saved_stats["clean_var_list_final"]
            self.layer_names = saved_stats["layer_names"]

        else:
            (
                self.clean_mean_list_final,
                self.clean_var_list_final,
                self.layer_names
            ) = self.extract_activation_alignment(
                batch_size=self.clean_bn_extract_batch
            )

            torch.save({
                "clean_mean_list_final": self.clean_mean_list_final,
                "clean_var_list_final": self.clean_var_list_final,
                "layer_names": self.layer_names
            }, self.cfg.statistic_save_path)

        # Select batch normalization layers to be used for loss generation
        self._setup_chosen_bn_layers()

    def extract_activation_alignment(self, batch_size):
        # Extract statistics from the training set (clear condition) used during training
        dataset = self.cfg.clear_dataset

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        loader_len = math.ceil(len(dataset)/batch_size)

        chosen_bn_info = []

        if self.cfg.model_type == "rtdetr":
            for name, m in self.model.named_modules():
                if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, RTDetrFrozenBatchNorm2d)):
                    if self.adaptation_layers == "backbone":
                        if ('model.backbone' in name and 'RTDetrFrozenBatchNorm2d' in str(type(m))) or \
                           ('backbone' in name.lower() and isinstance(m, RTDetrFrozenBatchNorm2d)):
                            chosen_bn_info.append((name, m))
                    elif self.adaptation_layers == "encoder":
                        if ('encoder' in name.lower() and not 'decoder' in name.lower()) and \
                           (isinstance(m, (nn.BatchNorm2d, nn.LayerNorm))):
                            chosen_bn_info.append((name, m))
                    elif self.adaptation_layers == "backbone+encoder":
                        if 'decoder' not in name.lower():
                            chosen_bn_info.append((name, m))
                    else:
                        if 'decoder' not in name.lower():
                            chosen_bn_info.append((name, m))

        elif self.cfg.model_type == "rcnn": 
            for name, m in self.model.named_modules():
                if isinstance(m, (FrozenBatchNorm2d)):
                    chosen_bn_info.append((name, m))

        # Select only the first half of all layers  
        cutoff = len(chosen_bn_info) // 2
        chosen_bn_info = chosen_bn_info[cutoff:]
        chosen_bn_layers = [module for name, module in chosen_bn_info]
        layer_names = [name for name, module in chosen_bn_info]

        n_chosen_layers = len(chosen_bn_layers)

        if self.cfg.model_type == "rtdetr":
            save_outputs = [SaveOutputRTDETR() for _ in range(n_chosen_layers)]
        
        elif self.cfg.model_type in ("rcnn", "swinrcnn"):
            save_outputs = [SaveOutput() for _ in range(n_chosen_layers)]

        clean_mean_act_list = [AverageMeter() for _ in range(n_chosen_layers)]
        clean_var_act_list = [AverageMeter() for _ in range(n_chosen_layers)]

        clean_mean_list_final = []
        clean_var_list_final = []

        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(loader, total=loader_len, desc="Extract statistic"):
                hook_list = [chosen_bn_layers[i].register_forward_hook(save_outputs[i]) for i in range(n_chosen_layers)]
                if self.cfg.model_type == "rtdetr":
                    # Move batch data to device
                    pixel_values = batch["pixel_values"].to(self.device)
                    labels = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                              for k, v in label.items()} for label in batch["labels"]]
                    _ = self.model(pixel_values=pixel_values, labels=labels)
                elif self.cfg.model_type == "rcnn":
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
    
    def _setup_chosen_bn_layers(self):
        if self.cfg.model_type == "rtdetr":
            current_bn_dict = {
            name: module for name, module in self.model.named_modules()
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, RTDetrFrozenBatchNorm2d))
        }
        elif self.cfg.model_type == "rcnn":
            current_bn_dict = {
                name: module for name, module in self.model.named_modules()
                if isinstance(module, FrozenBatchNorm2d)
            }

        self.chosen_bn_layers = []
        for layer_name in self.layer_names:
            if layer_name in current_bn_dict:
                self.chosen_bn_layers.append(current_bn_dict[layer_name])
            else:
                warnings.warn(f"Layer {layer_name} not found!")

    def forward(self, x=None, **kwargs):
        if x is None:
            x = kwargs

        for param in self.model.parameters():
            param.requires_grad = True

        self.model.eval()
        self.optimizer.zero_grad()

        if self.cfg.model_type == "rtdetr":
            n_chosen_layers = len(self.chosen_bn_layers)
            save_outputs_tta = [SaveOutputRTDETR() for _ in range(n_chosen_layers)]

            hook_list_tta = [
                self.chosen_bn_layers[i].register_forward_hook(save_outputs_tta[i])
                for i in range(n_chosen_layers)
            ]

            # Move batch data to device
            pixel_values = x['pixel_values'].to(self.device)
            labels = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in label.items()} for label in x['labels']]
            outputs = self.model(pixel_values=pixel_values, labels=labels)

        elif self.cfg.model_type == "rcnn":
            n_chosen_layers = len(self.chosen_bn_layers)
            save_outputs_tta = [SaveOutput() for _ in range(n_chosen_layers)]

            hook_list_tta = [
                self.chosen_bn_layers[i].register_forward_hook(save_outputs_tta[i])
                for i in range(n_chosen_layers)
            ]

            outputs = self.model(x)

        # Extract current batch statistics
        batch_mean_tta = [save_outputs_tta[x].get_out_mean() for x in range(n_chosen_layers)]
        batch_var_tta = [save_outputs_tta[x].get_out_var() for x in range(n_chosen_layers)]

        # Compute ActMAD loss
        loss_mean = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(self.device)
        loss_var = torch.tensor(0, requires_grad=True, dtype=torch.float).float().to(self.device)

        for i in range(n_chosen_layers):
            loss_mean += self.loss(
                batch_mean_tta[i].to(self.device),
                self.clean_mean_list_final[i].to(self.device)
            )
            loss_var += self.loss(
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
    
@dataclass
class NORMConfig:
    model_type: str = "rcnn" # "swinrcnn", "yolo11", "rtdetr"
    adaptation_layers: str = "backbone+encoder" # "backbone", "encoder"
    
    data_root: str = './datasets'
    device: torch.device = torch.device("cuda")
    batch_size: int = 4
    source_sum=128

    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4 # 0.01
    optimizer_option: str = "SGD" # AdamW

class NORM(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.cfg = config
        
        self.data_root = config.data_root
        self.device = config.device
        self.batch_size = config.batch_size

        if config.model_type in ("rtdetr", "yolo"):
            self.adaptation_layers = config.adaptation_layers
        
        self._setup()
    
    def _setup(self):
        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)
        self._apply_norm_adaptation()
    
    def _apply_norm_adaptation(self):
        # This code is required because some models use frozen batch normalization layers.
        for name, module in self.model.named_modules():
            if self.cfg.model_type == "rtdetr":
                if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, RTDetrFrozenBatchNorm2d)):
                    should_adapt = False
                    if self.adaptation_layers == "backbone":
                        if ('model.backbone' in name and isinstance(module, RTDetrFrozenBatchNorm2d)) or \
                        ('backbone' in name.lower() and isinstance(module, RTDetrFrozenBatchNorm2d)):
                            should_adapt = True
                    elif self.adaptation_layers == "encoder":
                        if ('encoder' in name.lower() and not 'decoder' in name.lower()) and \
                        isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                            should_adapt = True
                    elif self.adaptation_layers == "backbone+encoder":
                        if 'decoder' not in name.lower():
                            should_adapt = True
                    else:
                        if 'decoder' not in name.lower():
                            should_adapt = True

                    if not should_adapt:
                        continue

                    module.adapt_type = "NORM"
                    module.source_sum = self.cfg.source_sum

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
            
            elif self.cfg.model_type == "rcnn":
                for name, module in self.model.named_modules():
                    if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d)):
                        module.adapt_type = "NORM"
                        module.source_sum = self.cfg.source_sum

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

    def forward(self, x=None, **kwargs):
        if x is None:
            x = kwargs
        self.model.eval()
        if self.cfg.model_type == "rtdetr":
            pixel_values = x['pixel_values'].to(self.device)
            labels = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in label.items()} for label in x['labels']]
            outputs = self.model(pixel_values=pixel_values, labels=labels)
        elif self.cfg.model_type == "rcnn":
            outputs = self.model(x)
        return outputs

@dataclass
class DUAConfig:
    model_type: str = "rcnn" # "swinrcnn", "yolo11", "rtdetr"
    adaptation_layers: str = "backbone+encoder" # "backbone", "encoder"
    
    data_root: str = './datasets'
    device: torch.device = torch.device("cuda")
    batch_size: int = 4

    min_momentum_constant: int = 0.0001
    decay_factor: int = 0.94
    mom_pre: float = 0.01

    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4 # 0.01
    optimizer_option: str = "SGD" # AdamW

class DUA(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.cfg = config
        
        self.data_root = config.data_root
        self.device = config.device
        self.batch_size = config.batch_size

        self.min_momentum_constant = config.min_momentum_constant
        self.decay_factor = config.decay_factor
        self.mom_pre = config.mom_pre

        if config.model_type in ("rtdetr", "yolo"):
            self.adaptation_layers = config.adaptation_layers
        
        self._setup()
    
    def _setup(self):
        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)
        self._apply_dua_adaptation()

    def _apply_dua_adaptation(self):
        # This code is required because some models use frozen batch normalization layers.
        if self.cfg.model_type == "rtdetr":
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, RTDetrFrozenBatchNorm2d)):
                    should_adapt = False
                    if self.adaptation_layers == "backbone":
                        if ('model.backbone' in name and isinstance(module, RTDetrFrozenBatchNorm2d)) or \
                        ('backbone' in name.lower() and isinstance(module, RTDetrFrozenBatchNorm2d)):
                            should_adapt = True
                    elif self.adaptation_layers == "encoder":
                        if ('encoder' in name.lower() and not 'decoder' in name.lower()) and \
                        isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                            should_adapt = True
                    elif self.adaptation_layers == "backbone+encoder":
                        if 'decoder' not in name.lower():
                            should_adapt = True
                    else:
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
                                    dims = tuple(range(-len(self.normalized_shape), 0))
                                    batch_mean = x.mean(dim=dims, keepdim=True).squeeze()
                                    batch_var = x.var(dim=dims, keepdim=True, unbiased=True).squeeze()

                                if hasattr(self, 'running_mean'):
                                    self.running_mean.mul_(1 - current_momentum).add_(batch_mean, alpha=current_momentum)
                                    self.running_var.mul_(1 - current_momentum).add_(batch_var, alpha=current_momentum)

                                self.mom_pre *= self.decay_factor

                        # Standard normalization
                        if isinstance(self, nn.BatchNorm2d):
                            scale = self.weight * (self.running_var + self.eps).rsqrt()
                            bias = self.bias - self.running_mean * scale
                            scale = scale.reshape(1, -1, 1, 1)
                            bias = bias.reshape(1, -1, 1, 1)
                            out_dtype = x.dtype
                            out = x * scale.to(out_dtype) + bias.to(out_dtype)
                        elif isinstance(self, RTDetrFrozenBatchNorm2d):
                            eps = getattr(self, 'eps', 1e-5)
                            scale = self.weight * (self.running_var + eps).rsqrt()
                            bias = self.bias - self.running_mean * scale
                            scale = scale.reshape(1, -1, 1, 1)
                            bias = bias.reshape(1, -1, 1, 1)
                            out_dtype = x.dtype
                            out = x * scale.to(out_dtype) + bias.to(out_dtype)
                        elif isinstance(self, nn.LayerNorm):
                            out = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
                        else:
                            out = x

                        return out

                    module.forward = dua_forward.__get__(module, module.__class__)

        if self.cfg.model_type == "rcnn":
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
                        scale = scale.reshape(1, -1, 1, 1)
                        bias = bias.reshape(1, -1, 1, 1)
                        out_dtype = x.dtype
                        out = x * scale.to(out_dtype) + bias.to(out_dtype)

                        return out

                    module.forward = dua_forward.__get__(module, module.__class__)

    def forward(self, x=None, **kwargs):
        if x is None:
            x = kwargs
        self.model.eval()

        if self.cfg.model_type == "rtdetr":
            pixel_values = x['pixel_values'].to(self.device)
            labels = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in label.items()} for label in x['labels']]
            outputs = self.model(pixel_values=pixel_values, labels=labels)
        elif self.cfg.model_type == "rcnn":
            outputs = self.model(x)

        return outputs
    
@dataclass
class MeanTeacherConfig:
    model_type: str = "rcnn" # "swinrcnn", "yolo11", "rtdetr"
    adaptation_layers: str = "backbone+encoder" # "backbone", "encoder"

    data_root: str = './datasets'
    device: torch.device = torch.device("cuda")
    batch_size: int = 4

    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4 # 0.01
    optimizer_option: str = "SGD" # AdamW

    conf_threshold: float = 0.3
    ema_alpha: float = 0.99
    # Augmentation strength (reduce these values for RT-DETR if performance is poor)
    # For RT-DETR: consider augment_strength_n=1, augment_strength_m=5
    # For RCNN: keep augment_strength_n=2, augment_strength_m=10
    augment_strength_n: int = 2
    augment_strength_m: int = 10
    cutout_size: int = 16

    # RT-DETR specific
    image_size: int = 800
    reference_model_id: str = "PekingU/rtdetr_r50vd"

class MeanTeacher(nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.cfg = config
        
        self.data_root = config.data_root
        self.device = config.device
        self.batch_size = config.batch_size

        if config.model_type in ("rtdetr", "yolo"):
            self.adaptation_layers = config.adaptation_layers
        
        self._setup()
    
    def _setup(self):
        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)
        self._setup_teacher_model()
        self._setup_strong_augmentation()

    def _setup_teacher_model(self):
        # Make a teacher model
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

        # Set EMA alpha from config
        self.ema_alpha = self.cfg.ema_alpha

        # Initialize weight_reg (default to 0.0 if not set)
        if not hasattr(self, 'weight_reg'):
            self.weight_reg = 0.0

        params = []

        if self.cfg.model_type == "rtdetr":
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, RTDetrFrozenBatchNorm2d)):
                    should_adapt = False
                    if self.adaptation_layers == "backbone":
                        if ('model.backbone' in name and isinstance(module, RTDetrFrozenBatchNorm2d)) or \
                           ('backbone' in name.lower() and isinstance(module, RTDetrFrozenBatchNorm2d)):
                            should_adapt = True
                    elif self.adaptation_layers == "encoder":
                        if ('encoder' in name.lower() and not 'decoder' in name.lower()) and \
                           isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                            should_adapt = True
                    elif self.adaptation_layers == "backbone+encoder":
                        if 'decoder' not in name.lower():
                            should_adapt = True
                    else:
                        if 'decoder' not in name.lower():
                            should_adapt = True

                    if should_adapt:
                        if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                            if hasattr(module, 'weight') and module.weight is not None:
                                module.weight.requires_grad = True
                                params.append(module.weight)
                            if hasattr(module, 'bias') and module.bias is not None:
                                module.bias.requires_grad = True
                                params.append(module.bias)
                        elif isinstance(module, RTDetrFrozenBatchNorm2d):
                            if hasattr(module, 'weight') and module.weight is not None and not isinstance(module.weight, (int, float)):
                                module.weight.requires_grad = True
                                params.append(module.weight)
                            if hasattr(module, 'bias') and module.bias is not None and not isinstance(module.bias, (int, float)):
                                module.bias.requires_grad = True
                                params.append(module.bias)

                elif isinstance(module, (nn.Conv2d, nn.Linear)):
                    should_adapt = False
                    if self.adaptation_layers == "backbone":
                        if 'model.backbone' in name or 'backbone' in name.lower():
                            should_adapt = True
                    elif self.adaptation_layers == "encoder":
                        if 'encoder' in name.lower() and 'decoder' not in name.lower():
                            should_adapt = True
                    elif self.adaptation_layers == "backbone+encoder":
                        if 'decoder' not in name.lower():
                            should_adapt = True
                    else:
                        if 'decoder' not in name.lower():
                            should_adapt = True

                    if should_adapt and hasattr(module, 'bias') and module.bias is not None:
                        module.bias.requires_grad = True
                        params.append(module.bias)

        elif self.cfg.model_type == "rcnn":
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'bottom_up'):
                for m_name, m in self.model.backbone.bottom_up.named_modules():
                    if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                        m.weight.requires_grad = True
                        m.bias.requires_grad = True
                        params += [m.weight, m.bias]

                    if isinstance(m, (nn.Conv2d, nn.Linear)):
                        if "patch_embed" in m_name and "attn" in m_name:
                            continue
                        m.weight.requires_grad = True
                        params += [m.weight]
                        if m.bias is not None:
                            m.bias.requires_grad = True
                            params += [m.bias]
        
        trainable_params = params 
        
        if self.cfg.optimizer_option == "SGD":
            self.optimizer = optim.SGD(
                trainable_params,
                lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.weight_decay
            )

        elif self.cfg.optimizer_option == "AdamW":
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay
            )

        self.init_weights = [p.clone().detach() for p in params]

    def _setup_strong_augmentation(self):
        self.strong_augment = self._create_randaugment_mc(n=self.cfg.augment_strength_n, m=self.cfg.augment_strength_m)
    
    def _create_randaugment_mc(self, n: int, m: int):
        # https://github.com/natureyoo/ContinualTTA_ObjectDetection
        # I referenced this augmentation code from the link below.
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
    
    def forward(self, x=None, **kwargs):
        if x is None:
            x = kwargs

        # Apply augmentation
        weak_batch, strong_batch = self._apply_augmentation(x)

        # Get teacher predictions on weak (original) batch
        self.teacher_model.eval()
        with torch.no_grad():
            if self.cfg.model_type == "rtdetr":
                teacher_outputs = self.teacher_model(
                    pixel_values=weak_batch['pixel_values'].to(self.device),
                    labels=weak_batch['labels']
                )
            elif self.cfg.model_type == "rcnn":
                teacher_outputs = self.teacher_model(weak_batch)

        # Create pseudo labels for strong batch
        pseudo_labeled_batch = self._set_pseudo_labels(strong_batch, teacher_outputs)

        # Train student model with pseudo labels
        self.model.train()
        self.optimizer.zero_grad()

        if self.cfg.model_type == "rtdetr":
            model_output = self.model(
                pixel_values=pseudo_labeled_batch['pixel_values'].to(self.device),
                labels=pseudo_labeled_batch['labels']
            )

            if model_output.loss is not None and model_output.loss > 0:
                total_loss = model_output.loss

                # Add weight regularization if needed
                if hasattr(self, 'weight_reg') and self.weight_reg > 0.0:
                    reg_loss = torch.tensor(0.0, device=self.device)
                    for param, init_param in zip(self.optimizer.param_groups[0]['params'], self.init_weights):
                        reg_loss += torch.mean((param - init_param) ** 2)
                    total_loss += self.weight_reg * reg_loss

                total_loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

        elif self.cfg.model_type == "rcnn":
            # For Detectron2, we need EventStorage context for training
            from detectron2.utils.events import EventStorage

            with EventStorage() as storage:
                model_output = self.model(pseudo_labeled_batch)
                if isinstance(model_output, dict):
                    losses = model_output
                    total_loss = sum([losses[k] for k in losses])

                    if hasattr(self, 'weight_reg') and self.weight_reg > 0.0:
                        reg_loss = torch.tensor(0.0, device=self.device)
                        for param, init_param in zip(self.optimizer.param_groups[0]['params'], self.init_weights):
                            reg_loss += torch.mean((param - init_param) ** 2)
                        total_loss += self.weight_reg * reg_loss

                    if total_loss > 0:
                        total_loss.backward()

                        if hasattr(self.model, 'backbone'):
                            torch.nn.utils.clip_grad_norm_(self.model.backbone.parameters(), 1.0)

                        self.optimizer.step()

        # Update teacher with EMA
        self._update_teacher_ema()

        # Return teacher's output for evaluation (use eval mode)
        self.teacher_model.eval()
        with torch.no_grad():
            if self.cfg.model_type == "rtdetr":
                final_outputs = self.teacher_model(
                    pixel_values=x['pixel_values'].to(self.device),
                    labels=x['labels']
                )
            elif self.cfg.model_type == "rcnn":
                final_outputs = self.teacher_model(x)

        return final_outputs


    def _apply_augmentation(self, batch):
        if self.cfg.model_type == "rtdetr":
            # For RT-DETR: batch is {'pixel_values': tensor, 'labels': list}
            # Return weak (original) and strong (augmented) versions
            weak_batch = batch

            # Apply strong augmentation to pixel_values
            pixel_values = batch['pixel_values']
            batch_size = pixel_values.shape[0]
            strong_pixel_values = []

            for i in range(batch_size):
                img = pixel_values[i]  # [C, H, W]
                try:
                    # Apply RandAugmentMC
                    strong_img = self.strong_augment(img)
                    strong_pixel_values.append(strong_img)
                except Exception as e:
                    # Fallback to original image if augmentation fails
                    strong_pixel_values.append(img)

            strong_pixel_values = torch.stack(strong_pixel_values)
            strong_batch = {
                'pixel_values': strong_pixel_values,
                'labels': batch['labels']
            }

            return weak_batch, strong_batch

        elif self.cfg.model_type == "rcnn":
            # For RCNN: batch is a list of dict items
            weak_batch = []
            strong_batch = []

            for item in batch:
                weak_item = copy.deepcopy(item)
                weak_batch.append(weak_item)

                strong_item = copy.deepcopy(item)
                try:
                    # Apply RandAugmentMC which handles the conversion internally
                    strong_item["strong_aug_image"] = self.strong_augment(strong_item["image"])
                except Exception as e:
                    # Fallback to original image if augmentation fails
                    strong_item["strong_aug_image"] = strong_item["image"]
                strong_batch.append(strong_item)

            return weak_batch, strong_batch
    
    def _set_pseudo_labels(self, strong_batch, outputs):
        if self.cfg.model_type == "rtdetr":
            # strong_batch is {'pixel_values': tensor, 'labels': list}
            # outputs is RTDetrObjectDetectionOutput from teacher model
            annotation = []
            for bbox, logit in zip(outputs.pred_boxes, outputs.logits):
                probs = F.softmax(logit, dim=-1)
                scores, labels = probs.max(dim=-1)

                mask = scores > self.cfg.conf_threshold
                bbox = bbox[mask]
                scores = scores[mask]
                labels = labels[mask]
                annotation.append({
                    'class_labels' : labels,
                    'boxes' : bbox
                })
            pseudo_labels = {
                'pixel_values': strong_batch['pixel_values'],
                'labels' : annotation
            }
            return pseudo_labels

        elif self.cfg.model_type == "rcnn":
            # strong_batch is a list of dict items
            # outputs is a list of predictions from teacher model
            pseudo_labels = []
            for img, label in zip(strong_batch, outputs):
                inst = label['instances'][label['instances'].scores > self.cfg.conf_threshold]

                new_inp = {k: img[k] for k in img if k not in ['instances', 'image', 'strong_aug_image']}
                new_inp['image'] = img['strong_aug_image'] if 'strong_aug_image' in img else img['image']

                new_img_size = img['instances'].image_size
                ori_img_size = inst.image_size

                new_inst = Instances(new_img_size)
                new_inst.gt_classes = inst.pred_classes
                new_inst.gt_boxes = inst.pred_boxes

                if new_img_size != ori_img_size:
                    new_inst.gt_boxes.scale(
                        new_img_size[1] / ori_img_size[1],
                        new_img_size[0] / ori_img_size[0]
                    )

                new_inp['instances'] = new_inst
                pseudo_labels.append(new_inp)

            return pseudo_labels

    def _update_teacher_ema(self):
        with torch.no_grad():
            for t_p, s_p in zip(self.teacher_model.parameters(), self.model.parameters()):
                if s_p.requires_grad:
                    t_p.data = self.ema_alpha * t_p.data + (1 - self.ema_alpha) * s_p.data

class conv1x1(nn.Module):
    """
    1x1 convolution wrapper - matches ContinualTTA implementation exactly
    This wraps nn.Conv2d in a .conv attribute for compatibility with ContinualTTA's Adapter class
    """
    def __init__(self, in_planes, out_planes=None, stride=1, mode="parallel", bias=False):
        super(conv1x1, self).__init__()
        self.mode = mode
        if out_planes is None:
            self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=stride, padding=0, bias=bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)

    @property
    def weight(self):
        """Provide direct access to weight for initialization"""
        return self.conv.weight

    @property
    def bias(self):
        """Provide direct access to bias for initialization"""
        return self.conv.bias

    def forward(self, x):
        return self.conv(x)


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
    Exactly matches ContinualTTA's Adapter class (lines 76-87 in resnet.py)
    """
    def __init__(self, in_planes, planes, r=32, mode='parallel'):
        super(ParallelAdapterWithProjection, self).__init__()
        # Use conv1x1 wrapper like ContinualTTA for exact match
        bottleneck_channels = max(1, in_planes // r)  # Ensure at least 1 channel
        self.down_proj = conv1x1(in_planes, bottleneck_channels, 1, mode=mode, bias=True)
        self.non_linear_func = nn.ReLU()
        self.up_proj = conv1x1(bottleneck_channels, planes, 1, mode=mode, bias=True)

        # Initialize exactly like ContinualTTA Adapter (lines 82-85)
        nn.init.kaiming_uniform_(self.down_proj.conv.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.conv.weight)
        nn.init.zeros_(self.down_proj.conv.bias)
        nn.init.zeros_(self.up_proj.conv.bias)

    def forward(self, x):
        return self.up_proj(self.non_linear_func(self.down_proj(x)))


class ConvTaskWrapper(nn.Module):
    """
    Wraps a Conv2d layer with parallel adapter, matching ContinualTTA's conv_task implementation.
    This wrapper adds adapter output to the conv output: y = norm(conv(x)) + adapter(x) * scalar
    Exactly matches conv_task class from ContinualTTA (lines 90-165 in resnet.py)
    """
    def __init__(self, original_conv, adapter_mode='parallel', r=32, scalar=1.0):
        super(ConvTaskWrapper, self).__init__()
        self.conv = original_conv  # Original Conv2d layer (with norm)
        self.mode = adapter_mode
        self.scalar_type = scalar if isinstance(scalar, str) else None
        self.scalar = nn.Parameter(torch.ones(1)) if scalar == 'learnable_scalar' else torch.ones(1)

        if self.mode == 'parallel':
            # Get input and output channels from original conv
            in_channels = original_conv.in_channels if hasattr(original_conv, 'in_channels') else original_conv.conv.in_channels
            out_channels = original_conv.out_channels if hasattr(original_conv, 'out_channels') else original_conv.conv.out_channels

            # Create adapter using conv1x1 wrapper like ContinualTTA (lines 103-105)
            bottleneck_channels = max(1, in_channels // r)  # Ensure at least 1 channel
            self.down_proj = conv1x1(in_channels, bottleneck_channels, 1, mode=self.mode, bias=True)
            self.non_linear_func = nn.ReLU()
            self.up_proj = conv1x1(bottleneck_channels, out_channels, 1, mode=self.mode, bias=True)

            # CRITICAL: Add adapter_norm like ContinualTTA (line 106)
            # Even though it's not used in forward, it's still trained
            self.adapter_norm = nn.BatchNorm2d(out_channels)

            # Initialize like ContinualTTA (lines 107-110)
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.down_proj.bias)
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        # Base output: norm(conv(x))
        y = self.conv(x)

        if self.mode == 'parallel':
            # Adapter output: adapter(x) * scalar (conv_task lines 148-156)
            # Note: adapter_norm is NOT used in forward (line 148 is commented out in ContinualTTA)
            adapt_x = self.up_proj(self.non_linear_func(self.down_proj(x)))
            y = y + adapt_x * self.scalar.to(y.device)

        return y

@dataclass
class WHWConfig:
    model_type: str = "rcnn" # "swinrcnn", "yolo11", "rtdetr"
    adaptation_layers: str = "backbone+encoder" # "backbone", "encoder"
    
    data_root: str = './datasets'
    device: torch.device = torch.device("cuda")
    batch_size: int = 4

    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4 # 0.01
    optimizer_option: str = "SGD" # AdamW

    adaptation_where: str = "adapter"
    adapter_bottleneck_ratio: int = 32

    skip_redundant: str | None = None # "ema+"
    skip_period: int = 10
    skip_beta: float = 1.2
    skip_tau: float = 1.0

    fg_align: str | None = 'KL'
    gl_align: str | None = 'KL'
    alpha_fg: float = 1.0
    alpha_gl: float = 1.0
    ema_gamma: int = 128
    freq_weight: bool = False

    source_feat_stats: str | None = "./whw_source_statistics_clear.pt" # path 

    num_classes: int = 6

    iou_threshold: float = 0.5
    output_path: str = "./whw_source_statistics_clear.pt"

    clear_dataset: Dataset | None = None
    clear_statistics_batch: int = 16

class WHW(nn.Module):
    # https://github.com/robustaim/ContinualTTA_ObjectDetection/blob/main/detectron2/modeling/meta_arch/rcnn.py
    # Implemented by referencing the code from the link above.
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.cfg = config

        self.data_root = config.data_root
        self.device = config.device
        self.batch_size = config.batch_size
        
        self.adaptation_where = config.adaptation_where
        self.adapter_bottleneck_ratio = config.adapter_bottleneck_ratio

        self.skip_redundant = config.skip_redundant
        self.skip_period = config.skip_period
        self.skip_beta = config.skip_beta
        self.skip_tau = config.skip_tau

        self.loss_ema99 = 0.0
        self.loss_ema95 = 0.0
        self.loss_ema90 = 0.0
        self.adaptation_steps = 0
        self.used_steps = 0

        self.fg_align = config.fg_align
        self.gl_align = config.gl_align
        self.alpha_fg = config.alpha_fg
        self.alpha_gl = config.alpha_gl
        self.ema_gamma = config.ema_gamma
        self.freq_weight = config.freq_weight

        self.source_feat_stats = config.source_feat_stats
        self.adapters = nn.ModuleDict()
        self.conv1_wrappers = nn.ModuleDict()
        self.s_stats = None  # Source statistics
        self.t_stats = {}    # Target statistics
        self.template_cov = {}
        self.ema_n = {}      # EMA counters for each class
        self.s_div = {}      # Source divergence stats for skipping

        self.num_classes = config.num_classes

        self._setup()

    def _setup(self):
        self.model = copy.deepcopy(self.model)
        self.model.to(self.device)
        self._load_source_statistics()
        self._initialize_feature_stats()
        self._setup_parallel_adapters()
        self._patch_forward_box()
        self._patch_model_forward()  # Add this to patch model's forward method
        self._setup_adaptation()
        self._setup_training_mode()

    def _load_source_statistics(self):
        # Load saved statistics if available, otherwise create new ones
        if self.source_feat_stats is not None and os.path.exists(self.cfg.output_path):
            self.s_stats = torch.load(self.cfg.output_path, map_location=self.device)
            return

        self.s_stats = self.collect_source_statistics(batch_size= self.cfg.clear_statistics_batch, iou_threshold=self.cfg.iou_threshold, output_path=self.cfg.output_path)

    def collect_source_statistics(self, batch_size=16, iou_threshold=0.5, output_path=None):
        # Collect source-domain features and compute summary statistics
        dataset = self.cfg.clear_dataset

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        loader_len = math.ceil(len(dataset)/batch_size)

        # Collections
        gl_features = {} # Global features from backbone
        fg_features = {} # Foreground features from ROI head
        iou_with_gt = {} # IoU with ground truth

        self.model.eval()

        with torch.inference_mode():
            for batch in tqdm(loader, total=loader_len, desc="Collecting features"):
                images = self.model.preprocess_image(batch)

                features = self.model.backbone(images.tensor)
                if isinstance(features, tuple):
                    features = features[0]

                # Collect global features
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

                # Calculate baseline KL divergence for this class
                # Split features into two halves and compute KL divergence between them
                # This measures the internal variability of the source distribution
                kl_div_value = 1.0  # Default fallback
                if len(feats) >= 20:  # Need enough samples to split
                    try:
                        mid_point = len(feats) // 2
                        feats1 = feats[:mid_point]
                        feats2 = feats[mid_point:]

                        mean1 = feats1.mean(dim=0)
                        mean2 = feats2.mean(dim=0)
                        cov1 = torch.from_numpy(np.cov(feats1.T.numpy())).float() + torch.eye(cov.shape[0]) * 1e-4
                        cov2 = torch.from_numpy(np.cov(feats2.T.numpy())).float() + torch.eye(cov.shape[0]) * 1e-4

                        # Compute symmetric KL divergence
                        dist1 = torch.distributions.MultivariateNormal(mean1, cov1)
                        dist2 = torch.distributions.MultivariateNormal(mean2, cov2)
                        kl_div_value = ((torch.distributions.kl.kl_divergence(dist1, dist2) +
                                        torch.distributions.kl.kl_divergence(dist2, dist1)) / 2).item()

                        # Clip to reasonable range
                        kl_div_value = max(0.1, min(kl_div_value, 100.0))
                    except Exception as e:
                        print(f"    Warning: Could not compute KL div for class {k}: {e}")
                        kl_div_value = 1.0

                stats["kl_div"][k] = kl_div_value
                print(f"  FG[{k}]: {len(feats)} samples, mean shape {mean.shape}, cov shape {cov.shape}, KL div: {kl_div_value:.4f}")
            else:
                print(f"  FG[{k}]: No samples collected")

        # Save statistics
        if output_path is None:
            output_path = f"whw_source_statistics.pt"

        torch.save(stats, output_path)

        return stats
    
    def _initialize_feature_stats(self):
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
    
    def _setup_parallel_adapters(self):
        """
        Setup parallel adapters for test-time adaptation

        two types of adapters to ResNet blocks:
        1. Block-level adapter: Applied to conv2 output(ParallelAdapterWithProjection)
        2. Conv1 wrapper: Wraps conv1 with ConvTaskWrapper
        """
        if self.cfg.model_type == "rcnn":
            # Add first adapter type: block-level adapters on conv2 output
            self._add_block_adapters()

            # Add second adapter type: conv1 wrapper adapters
            self._wrap_conv1_with_adapters()

    def _add_block_adapters(self):
        if self.cfg.model_type == "rcnn":
            for stage_name in ['res2', 'res3', 'res4', 'res5']:
                if hasattr(self.model.backbone.bottom_up, stage_name):
                    stage = getattr(self.model.backbone.bottom_up, stage_name)
                    for block_idx, block in enumerate(stage):
                        if hasattr(block, 'conv2') and hasattr(block, 'conv3'):
                            # Only add to stride=1 blocks
                            if hasattr(block.conv1, 'stride') and block.conv1.stride == (1, 1):
                                conv2_channels = block.conv2.out_channels  # bottleneck channels
                                conv3_channels = block.conv3.out_channels  # final output channels

                                # Create adapter using ParallelAdapterWithProjection
                                adapter = ParallelAdapterWithProjection(
                                    in_planes=conv2_channels,
                                    planes=conv3_channels,
                                    r=self.adapter_bottleneck_ratio,
                                    mode='parallel'
                                )
                                adapter.to(self.device)

                                # Store adapter in block
                                block.adapter = adapter
                                block.has_adapter = True

                                # Register in adapters dict
                                adapter_name = f"{stage_name}_block{block_idx}_block_adapter"
                                self.adapters[adapter_name] = adapter

                                # Patch block forward to use adapter
                                self._patch_block_forward_with_adapter(block)

    def _patch_block_forward_with_adapter(self, block):
        # Patch block forward to apply adapter to conv2 output
        original_forward = block.forward

        def forward_with_adapter(x):
            # Conv1
            out = block.conv1(x)
            out = F.relu_(out)

            # Conv2
            out_tmp = block.conv2(out)  # Store conv2 output for adapter
            out = F.relu_(out_tmp)

            # Conv3
            out = block.conv3(out)

            # Shortcut connection
            if block.shortcut is not None:
                shortcut = block.shortcut(x)
            else:
                shortcut = x

            # Apply parallel adapter: out + shortcut + adapter(conv2_output)
            if hasattr(block, 'has_adapter') and block.has_adapter:
                out = out + shortcut + block.adapter(out_tmp)
            else:
                out = out + shortcut

            # Final activation
            out = F.relu_(out)
            return out

        block.forward = forward_with_adapter

    def _wrap_conv1_with_adapters(self):
        for stage_name in ['res2', 'res3', 'res4', 'res5']:
            if hasattr(self.model.backbone.bottom_up, stage_name):
                stage = getattr(self.model.backbone.bottom_up, stage_name)
                for block_idx, block in enumerate(stage):
                    if hasattr(block, 'conv1'):
                        # Only wrap conv1 in blocks with stride=1
                        if hasattr(block.conv1, 'stride') and block.conv1.stride == (1, 1):
                            # Skip if already wrapped
                            if isinstance(block.conv1, ConvTaskWrapper):
                                continue

                            adapter_name = f"{stage_name}_block{block_idx}_conv1_adapter"

                            # Wrap conv1 with ConvTaskWrapper
                            wrapped_conv1 = ConvTaskWrapper(
                                original_conv=block.conv1,
                                adapter_mode='parallel',
                                r=self.adapter_bottleneck_ratio,
                                scalar=1.0
                            )
                            wrapped_conv1.to(self.device)

                            # Replace conv1 with wrapped version
                            block.conv1 = wrapped_conv1

                            # Store the wrapper for optimizer
                            self.adapters[adapter_name] = wrapped_conv1

    def _patch_forward_box(self):
        # Patch roi_heads._forward_box
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
        Patch model's forward to return (results, adapt_loss, feature_sim)
        This makes the model behave exactly
        """
        original_forward = self.model.forward

        def continual_tta_forward(batched_inputs):
            # forward that returns adapt_loss
            images = self.model.preprocess_image(batched_inputs)
            features = self.model.backbone(images.tensor)

            if isinstance(features, tuple):
                features = features[0]

            adapt_loss = {}
            feature_sim = {}

            # Set to eval mode for proposal/roi_heads
            self.model.roi_heads.training = False
            self.model.proposal_generator.training = False

            # Get proposals
            proposals, _ = self.model.proposal_generator(images, features, None)

            # Get box features and predictions
            pred_instances, predictions, box_features = self.model.roi_heads._forward_box(features, proposals, outs=True)
            results = self.model.roi_heads.forward_with_given_boxes(features, pred_instances)

            # Compute foreground alignment
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
                    if (fg_preds == k).sum() > 0:  # only check sum
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

            # Compute global alignment
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

            # Postprocess
            processed_results = self.model._postprocess(results, batched_inputs, images.image_sizes)

            return processed_results, adapt_loss, feature_sim

        # Replace model's forward
        self.model.forward = continual_tta_forward

    def _setup_adaptation(self):
        """
        Setup optimizer for ContinualTTA adaptation
        Matches configure_adaptation_model.py lines 26-42 from ContinualTTA
        """
        params = []
        if self.cfg.lr > 0:
            if self.adaptation_where == "adapter":
                # Collect parameters from BOTH adapter types like ContinualTTA
                for stage_name in ['res2', 'res3', 'res4', 'res5']:
                    if hasattr(self.model.backbone.bottom_up, stage_name):
                        stage = getattr(self.model.backbone.bottom_up, stage_name)
                        for block_idx, block in enumerate(stage):
                            # FIRST ADAPTER: block.adapter (applied to conv2 output)
                            if hasattr(block, 'adapter'):
                                block.adapter.requires_grad_(True)
                                params += list(block.adapter.parameters())

                            # SECOND ADAPTER: block.conv1 wrapper (ConvTaskWrapper)
                            if isinstance(block.conv1, ConvTaskWrapper):
                                # Freeze original conv (ContinualTTA behavior)
                                for param in block.conv1.conv.parameters():
                                    param.requires_grad = False

                                # Enable adapter components (lines 36-38 in configure_adaptation_model.py)
                                if hasattr(block.conv1, 'down_proj'):
                                    block.conv1.down_proj.requires_grad_(True)
                                    params += list(block.conv1.down_proj.parameters())
                                if hasattr(block.conv1, 'up_proj'):
                                    block.conv1.up_proj.requires_grad_(True)
                                    params += list(block.conv1.up_proj.parameters())

                                # CRITICAL: Enable adapter_norm (lines 51-54 in configure_adaptation_model.py)
                                if hasattr(block.conv1, 'adapter_norm'):
                                    block.conv1.adapter_norm.track_running_stats = False
                                    block.conv1.adapter_norm.requires_grad_(True)
                                    params += list(block.conv1.adapter_norm.parameters())

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

        if self.cfg.optimizer_option == "SGD":
            self.optimizer = optim.SGD(
                params,
                lr=self.cfg.lr,
                momentum=self.cfg.momentum,
                weight_decay=self.cfg.weight_decay
            )

        elif self.cfg.optimizer_option == "AdamW":
            self.optimizer = optim.AdamW(
                params,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay
            )

        else :
            warnings.warn("Unknown optimizer_option.")

    def _setup_training_mode(self):
        # Set model components to training mode
        self.model.training = True
        if hasattr(self.model, 'proposal_generator'):
            self.model.proposal_generator.training = True
        if hasattr(self.model, 'roi_heads'):
            self.model.roi_heads.training = True

        for adapter in self.adapters.values():
            adapter.train()

        self.model.online_adapt = True

    def forward(self, x):
        _, adapt_loss, feature_sim = self.model(x)

        if adapt_loss:
            total_loss = sum(adapt_loss.values())
            self.adaptation_steps += 1

            # loss-based skipping logic
            cur_used = self._should_adapt(adapt_loss)

            if total_loss > 0 and cur_used:
                total_loss.backward()

                if hasattr(self.model, 'backbone'):
                    torch.nn.utils.clip_grad_norm_(self.model.backbone.parameters(), 5.0)

                # Clip adapter gradients
                for adapter in self.adapters.values():
                    torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
                
                self.optimizer.step()
                self.used_steps += 1

                if self.used_steps % 10 == 0:
                    loss_str = ", ".join([f"{k}: {v.item():.4f}" for k, v in adapt_loss.items()])
                    print(f"WHW: Used {self.used_steps}/{self.adaptation_steps} steps - {loss_str}")

            if "global_align" in adapt_loss:
                self._update_loss_ema(adapt_loss["global_align"].item())
        
        self.optimizer.zero_grad()

        # with torch.no_grad():
        final_outputs, _, _ = self.model(x)
        
        return final_outputs

    def _should_adapt(self, adapt_loss: dict) -> bool:
        if self.skip_redundant is None:
            return True

        if "global_align" not in adapt_loss:
            return True
        
        loss_value = adapt_loss["globel_align"].item()

        # Calculate divergence threshold
        div_thr = 2 * sum(self.s_div.values()) * self.skip_tau if self.skip_redundant is not None else 2 * sum(self.s_div.values())

        cur_used = False

        # Period-based skipping
        if 'period' in self.skip_redundant and self.adaptation_steps % self.skip_period == 0:
            cur_used = True

        # Statistical threshold-based skipping
        if 'stat' in self.skip_redundant and loss_value > div_thr:
            cur_used = True

        # EMA-based skipping
        if 'ema' in self.skip_redundant and self.loss_ema99 > 0:
            ema_ratio = loss_value / (self.loss_ema99 + 1e-7)
            if ema_ratio > self.skip_beta:
                cur_used = True
        
        return cur_used

    def _update_loss_ema(self, loss_value):
        self.loss_ema99 = 0.99 * self.loss_ema99 + 0.01 * loss_value
        self.loss_ema95 = 0.95 * self.loss_ema95 + 0.05 * loss_value
        self.loss_ema90 = 0.9 * self.loss_ema90 + 0.1 * loss_value
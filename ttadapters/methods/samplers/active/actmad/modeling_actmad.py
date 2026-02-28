from typing import Literal, Iterator, List, Optional
from dataclasses import dataclass
from pathlib import Path
import warnings
import re

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from detectron2.layers import FrozenBatchNorm2d
from transformers.models.rt_detr.modeling_rt_detr import RTDetrFrozenBatchNorm2d

from ....base import AdaptationEngine, BaseModel
from .configuration_actmad import ActMADConfig


# Collect stats(ResNet/YOLO)
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.clone())

    def clear(self):
        self.outputs = []

    def get_out_mean(self):
        out = torch.vstack(self.outputs)
        out = torch.mean(out, dim=0)
        return out

    def get_out_var(self):
        out = torch.vstack(self.outputs)
        out = torch.var(out, dim=0, correction=0)
        return out


# # Collect stats(RT-DETR)
class SaveOutputRTDETR:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.clone())

    def clear(self):
        self.outputs = []

    def get_out_mean(self):
        out = torch.vstack(self.outputs)
        out = torch.mean(out, dim=0)
        return out

    def get_out_var(self):
        out = torch.vstack(self.outputs)
        out = torch.var(out, dim=0, correction=0)
        return out


# Collect stats(SwinT)
class SaveOutputSwinT:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out.clone())

    def clear(self):
        self.outputs = []

    def get_out_mean(self):
        out = torch.vstack(self.outputs)
        out = torch.mean(out, dim=[0, 1])
        return out

    def get_out_var(self):
        out = torch.vstack(self.outputs)
        out = torch.var(out, dim=[0, 1], correction=0)
        return out


# ActMAD method
class ActMADEngine(AdaptationEngine):
    model_name = "ActMADEngine"
    config_class = ActMADConfig

    def __init__(self, config: ActMADConfig, base_model: BaseModel):
        super().__init__(config, base_model)
        self.config = config

    def _pre_init(self):
        self.clean_mean_list_final: Optional[List[torch.Tensor]] = None
        self.clean_var_list_final: Optional[List[torch.Tensor]] = None
        self.layer_names: Optional[List[str]] = None
        self.chosen_bn_layers: Optional[List[nn.Module]] = None

        if self.config.loss_type == "L1":
            self._loss_fn = nn.L1Loss(reduction="mean")
        else:
            self._loss_fn = nn.MSELoss(reduction="mean")

    def _post_init(self):
        for param in self.base_model.parameters():
            param.requires_grad = True
        self._load_or_init_statistics()

        self.to(self.device)

    def _load_or_init_statistics(self):
        # Load saved statistics if a stats path is provided
        if self.config.statistic_save_path and Path(self.config.statistic_save_path).exists():
            saved_stats = torch.load(self.config.statistic_save_path, weights_only=False)
            self.clean_mean_list_final = saved_stats["clean_mean_list_final"]
            self.clean_var_list_final = saved_stats["clean_var_list_final"]
            self.layer_names = saved_stats["layer_names"]

            self._setup_chosen_bn_layers()
            print(f"[ActMADEngine] Loaded statistics from {self.config.statistic_save_path}")

        elif self.config.statistic_save_path:
            print(f"[ActMADEngine] Stats file not found: {self.config.statistic_save_path}. Will collect during fit().")

        else:
            print("[ActMADEngine] No statistics path provided.")

    def _get_bn_layers_info(self) -> List[tuple]:
        # Select normalization layers (BN/LN/FrozenBN) to adapt based on model type and configured target layers
        chosen_bn_info = []

        # RT-DETR
        if self.config.base_type == "rtdetr":
            for name, m in self.base_model.named_modules():
                if isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, RTDetrFrozenBatchNorm2d)):
                    should_add = False
                    if self.config.adaptation_layers == "backbone":
                        if ('model.backbone' in name and isinstance(m, RTDetrFrozenBatchNorm2d)) or \
                           ('backbone' in name.lower() and isinstance(m, RTDetrFrozenBatchNorm2d)):
                            should_add = True
                    elif self.config.adaptation_layers == "encoder":
                        if ('encoder' in name.lower() and 'decoder' not in name.lower()) and \
                           isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                            should_add = True
                    elif self.config.adaptation_layers == "backbone+encoder":
                        if 'decoder' not in name.lower():
                            should_add = True
                    else:
                        if 'decoder' not in name.lower():
                            should_add = True

                    if should_add:
                        chosen_bn_info.append((name, m))

        # Fast R-CNN with a ResNet backbone
        elif self.config.base_type == "rcnn":
            for name, m in self.base_model.named_modules():
                if isinstance(m, FrozenBatchNorm2d):
                    chosen_bn_info.append((name, m))

        # Fast R-CNN with a SwinT backbone
        elif self.config.base_type == "swinrcnn":
            for name, m in self.base_model.named_modules():
                if isinstance(m, nn.LayerNorm):
                    should_add = False
                    if self.config.adaptation_layers == "backbone":
                        if 'backbone' in name.lower() or 'bottom_up' in name.lower():
                            should_add = True
                    elif self.config.adaptation_layers == "encoder":
                        if 'fpn' in name.lower() or 'neck' in name.lower():
                            should_add = True
                    elif self.config.adaptation_layers == "backbone+encoder":
                        if 'decoder' not in name.lower() and 'head' not in name.lower():
                            should_add = True

                    if should_add:
                        chosen_bn_info.append((name, m))
        
        # YOLO
        elif self.config.base_type == "yolo11":
            # YOLO11 uses flat Sequential indexing: model.{idx}.xxx
            # Backbone: layers 0-10, Neck: layers 11-22, Head: layer 23
            for name, m in self.base_model.named_modules():
                if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    match = re.match(r'model\.(\d+)', name)
                    if not match:
                        continue
                    layer_idx = int(match.group(1))

                    should_add = False
                    if self.config.adaptation_layers == "backbone":
                        should_add = layer_idx <= 10
                    elif self.config.adaptation_layers == "encoder":
                        should_add = 11 <= layer_idx <= 22
                    elif self.config.adaptation_layers == "backbone+encoder":
                        should_add = layer_idx <= 22

                    if should_add:
                        chosen_bn_info.append((name, m))

        return chosen_bn_info

    def extract_clean_statistics(self, dataloader: DataLoader, max_batches: Optional[int] = None):
        # Extract statistics from the source data
        
        chosen_bn_info = self._get_bn_layers_info()

        # Use second half of layers
        cutoff = len(chosen_bn_info) // 2
        chosen_bn_info = chosen_bn_info[cutoff:]
        chosen_bn_layers = [module for name, module in chosen_bn_info]
        self.layer_names = [name for name, module in chosen_bn_info]

        n_chosen_layers = len(chosen_bn_layers)

        if n_chosen_layers == 0:
            warnings.warn("[ActMADEngine] No normalization layers found!")
            return

        if self.config.base_type == "rtdetr":
            save_outputs = [SaveOutputRTDETR() for _ in range(n_chosen_layers)]

        elif self.config.base_type == "swinrcnn":
            save_outputs = [SaveOutputSwinT() for _ in range(n_chosen_layers)]
        
        else:
            save_outputs = [SaveOutput() for _ in range(n_chosen_layers)]

        clean_mean_act_list = [AverageMeter() for _ in range(n_chosen_layers)]
        clean_var_act_list = [AverageMeter() for _ in range(n_chosen_layers)]

        print(f"[ActMADEngine] Extracting statistics from {n_chosen_layers} layers...")

        with torch.no_grad():
            self.base_model.eval()
            for idx, batch in enumerate(tqdm(dataloader, desc="Extracting statistics")):
                if max_batches is not None and idx >= max_batches:
                    break

                hook_list = [
                    chosen_bn_layers[i].register_forward_hook(save_outputs[i])
                    for i in range(n_chosen_layers)
                ]

                if self.config.base_type == "rtdetr":
                    pixel_values = batch['pixel_values'].to(self.device)
                    _ = self.base_model(pixel_values=pixel_values)
        
                elif self.config.base_type == "yolo11":
                    img = batch['img'].to(self.device)
                    _ = self.base_model(img)
        
                else:
                    _ = self.base_model(batch)

                for i in range(n_chosen_layers):
                    clean_mean_act_list[i].update(save_outputs[i].get_out_mean())
                    clean_var_act_list[i].update(save_outputs[i].get_out_var())
                    save_outputs[i].clear()
                    hook_list[i].remove()

        self.clean_mean_list_final = [clean_mean_act_list[i].avg for i in range(n_chosen_layers)]
        self.clean_var_list_final = [clean_var_act_list[i].avg for i in range(n_chosen_layers)]

        self._setup_chosen_bn_layers()
        print(f"[ActMADEngine] Extracted statistics for {n_chosen_layers} layers")

    def save_statistics(self, save_path: str):
        # Save statistics to the specified path

        if self.clean_mean_list_final is None:
            raise ValueError("No statistics to save. Call extract_clean_statistics() first.")

        torch.save({
            "clean_mean_list_final": self.clean_mean_list_final,
            "clean_var_list_final": self.clean_var_list_final,
            "layer_names": self.layer_names
        }, save_path)

        print(f"[ActMADEngine] Saved statistics to {save_path}")

    def _setup_chosen_bn_layers(self):
        # Resolve configured layer names to actual normalization layers and store them for adaptation

        if self.layer_names is None:
            return

        if self.config.base_type == "rtdetr":
            current_bn_dict = {
                name: module for name, module in self.base_model.named_modules()
                if isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, RTDetrFrozenBatchNorm2d))
            }
        
        elif self.config.base_type == "rcnn":
            current_bn_dict = {
                name: module for name, module in self.base_model.named_modules()
                if isinstance(module, FrozenBatchNorm2d)
            }
        
        elif self.config.base_type == "swinrcnn":
            current_bn_dict = {
                name: module for name, module in self.base_model.named_modules()
                if isinstance(module, nn.LayerNorm)
            }
        
        elif self.config.base_type == "yolo11":
            current_bn_dict = {
                name: module for name, module in self.base_model.named_modules()
                if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm))
            }
        
        else:
            current_bn_dict = {}

        self.chosen_bn_layers = []
        for layer_name in self.layer_names:
            if layer_name in current_bn_dict:
                self.chosen_bn_layers.append(current_bn_dict[layer_name])
        
            else:
                warnings.warn(f"[ActMADEngine] Layer {layer_name} not found!")

    def fit(self, source_preparation=None, batch_size=None, **kwargs):
        if self.clean_mean_list_final is None and self.config.statistic_save_path and source_preparation is not None:
            batch_size = batch_size or self.config.clean_bn_extract_batch
            collate_fn = getattr(source_preparation, 'collate_fn', None)
            dataloader = DataLoader(source_preparation, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            self.extract_clean_statistics(dataloader)
            self.save_statistics(self.config.statistic_save_path)

    def online_parameters(self) -> Iterator[nn.Parameter]:
        return self.base_model.parameters()

    @property
    def optimizer(self):
        if self._optimizer is None:
            if self.config.optim == "SGD":
                self._optimizer = optim.SGD(
                    self.base_model.parameters(),
                    lr=self.config.adapt_lr,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay
                )
        
            elif self.config.optim == "AdamW":
                self._optimizer = optim.AdamW(
                    self.base_model.parameters(),
                    lr=self.config.adapt_lr,
                    weight_decay=self.config.weight_decay
                )
        
        return self._optimizer

    def _reset_stats(self):
        self._stats = {
            'losses': [],
            'num_batches': 0,
        }

    def reset(self, reset_stats=False):
        self.base_model.load_state_dict(self.base_state)

        for param in self.base_model.parameters():
            param.requires_grad = True
        
        self._optimizer = None
        self._setup_chosen_bn_layers()

        self.online(self.adapting)
        self.to(self.device)

        if reset_stats:
            current_stats = self._stats
            self._reset_stats()
            return current_stats
        
        return None

    def forward(self, batched_inputs=None, **kwargs):
        if batched_inputs is None and kwargs:
            batched_inputs = kwargs

        if self.clean_mean_list_final is None or self.chosen_bn_layers is None:
            warnings.warn("[ActMADEngine] Statistics not loaded. Running without adaptation.")
        
            if self.config.base_type == "rtdetr":
                return self.base_model(**batched_inputs)
            
            elif self.config.base_type == "yolo11":
                img = batched_inputs if torch.is_tensor(batched_inputs) else batched_inputs['img']
                return self.base_model(img.to(self.device))
        
            return self.base_model(batched_inputs)

        if not self.adapting:
            if self.config.base_type == "rtdetr":
                return self.base_model(**batched_inputs)
            
            elif self.config.base_type == "yolo11":
                img = batched_inputs if torch.is_tensor(batched_inputs) else batched_inputs['img']
                return self.base_model(img.to(self.device))
        
            return self.base_model(batched_inputs)

        self.base_model.eval()
        self.optimizer.zero_grad()

        n_chosen_layers = len(self.chosen_bn_layers)

        if self.config.base_type == "rtdetr":
            save_outputs = [SaveOutputRTDETR() for _ in range(n_chosen_layers)]

        elif self.config.base_type == "swinrcnn":
            save_outputs = [SaveOutputSwinT() for _ in range(n_chosen_layers)]
        
        else:
            save_outputs = [SaveOutput() for _ in range(n_chosen_layers)]

        hook_list = [
            self.chosen_bn_layers[i].register_forward_hook(save_outputs[i])
            for i in range(n_chosen_layers)
        ]

        if self.config.base_type == "rtdetr":
            pixel_values = batched_inputs['pixel_values'].to(self.device)
            outputs = self.base_model(pixel_values=pixel_values)
        
        elif self.config.base_type == "yolo11":
            img = batched_inputs if torch.is_tensor(batched_inputs) else batched_inputs['img']
            outputs = self.base_model(img.to(self.device))

        else:
            outputs = self.base_model(batched_inputs)

        batch_mean = [save_outputs[i].get_out_mean() for i in range(n_chosen_layers)]
        batch_var = [save_outputs[i].get_out_var() for i in range(n_chosen_layers)]
        loss_terms = []
        
        for i in range(n_chosen_layers):
            loss_terms.append(self._loss_fn(
                batch_mean[i].to(self.device),
                self.clean_mean_list_final[i].to(self.device)
            ))
            loss_terms.append(self._loss_fn(
                batch_var[i].to(self.device),
                self.clean_var_list_final[i].to(self.device)
            ))

        loss = sum(loss_terms) if loss_terms else torch.tensor(0.0, device=self.device)

        loss.backward()
        self.optimizer.step()

        for i in range(n_chosen_layers):
            save_outputs[i].clear()
            hook_list[i].remove()

        self._stats['losses'].append(loss.item())
        self._stats['num_batches'] = self._stats.get('num_batches', 0) + 1

        return outputs

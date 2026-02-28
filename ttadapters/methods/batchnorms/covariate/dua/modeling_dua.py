from typing import Iterator
import re

import torch
from torch import nn

from detectron2.layers import FrozenBatchNorm2d
from transformers.models.rt_detr.modeling_rt_detr import RTDetrFrozenBatchNorm2d

from ....base import AdaptationEngine, BaseModel
from .configuration_dua import DUAConfig


class DUAEngine(AdaptationEngine):
    model_name: str = "DUAEngine"

    def __init__(self, config: DUAConfig, base_model: BaseModel):
        super().__init__(config, base_model)
        self.config = config

    def _pre_init(self):
        self._dua_modules = []

    def _post_init(self):
        self._apply_dua_adaptation()
        self.to(self.device)

    def _apply_dua_adaptation(self):
        if self.config.base_type == "rcnn":
            self._apply_rcnn_dua()
        elif self.config.base_type == "rtdetr":
            self._apply_rtdetr_dua()
        elif self.config.base_type == "yolo11":
            self._apply_yolo_dua()

    def _apply_rcnn_dua(self):
        for name, module in self.base_model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d)):
                self._attach_dua(module)

    def _apply_rtdetr_dua(self):
        adaptation_layers = self.config.adaptation_layers

        for name, module in self.base_model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, RTDetrFrozenBatchNorm2d)):
                should_adapt = False

                if adaptation_layers == "backbone":
                    if ('model.backbone' in name and isinstance(module, RTDetrFrozenBatchNorm2d)) or \
                       ('backbone' in name.lower() and isinstance(module, RTDetrFrozenBatchNorm2d)):
                        should_adapt = True
                elif adaptation_layers == "encoder":
                    if ('encoder' in name.lower() and 'decoder' not in name.lower()) and \
                       isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                        should_adapt = True
                elif adaptation_layers == "backbone+encoder":
                    if 'decoder' not in name.lower():
                        should_adapt = True
                else:
                    if 'decoder' not in name.lower():
                        should_adapt = True

                if not should_adapt:
                    continue

                self._attach_dua(module)

    def _apply_yolo_dua(self):
        adaptation_layers = self.config.adaptation_layers

        for name, module in self.base_model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                # YOLO11 uses flat Sequential indexing: model.{idx}.xxx
                # Backbone: layers 0-10, Neck: layers 11-22, Head: layer 23
                match = re.match(r'model\.(\d+)', name)
                if not match:
                    continue
                layer_idx = int(match.group(1))

                should_adapt = False
                if adaptation_layers == "backbone":
                    should_adapt = layer_idx <= 10
                elif adaptation_layers == "encoder":
                    should_adapt = 11 <= layer_idx <= 22
                elif adaptation_layers == "backbone+encoder":
                    should_adapt = layer_idx <= 22

                if not should_adapt:
                    continue

                self._attach_dua(module)

    def _attach_dua(self, module):
        # Replace the forward method of the target module with the DUA forward
        module.adapt_type = "DUA"
        module.min_momentum_constant = self.config.min_momentum_constant
        module.decay_factor = self.config.decay_factor
        module.mom_pre = self.config.mom_pre

        if not hasattr(module, 'original_running_mean') and hasattr(module, 'running_mean'):
            module.original_running_mean = module.running_mean.clone()
            module.original_running_var = module.running_var.clone()

        self._dua_modules.append(module)

        def dua_forward(self, x):
            if hasattr(self, 'adapt_type') and self.adapt_type == "DUA":
                with torch.no_grad():
                    current_momentum = self.mom_pre + self.min_momentum_constant

                    if isinstance(self, (nn.BatchNorm2d, FrozenBatchNorm2d, RTDetrFrozenBatchNorm2d)):
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

            # Normalization
            if isinstance(self, nn.BatchNorm2d):
                scale = self.weight * (self.running_var + self.eps).rsqrt()
                bias = self.bias - self.running_mean * scale
                scale = scale.reshape(1, -1, 1, 1)
                bias = bias.reshape(1, -1, 1, 1)
                out_dtype = x.dtype
                out = x * scale.to(out_dtype) + bias.to(out_dtype)

            elif isinstance(self, (FrozenBatchNorm2d, RTDetrFrozenBatchNorm2d)):
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

    def online_parameters(self) -> Iterator[nn.Parameter]:
        return iter([])

    @property
    def optimizer(self):
        return None

    def _reset_stats(self):
        self._stats = {
            'num_batches': 0,
        }

    def reset(self, reset_stats=False):
        self.base_model.load_state_dict(self.base_state)
        self._dua_modules = []
        self._apply_dua_adaptation()

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
        self._stats['num_batches'] = self._stats.get('num_batches', 0) + 1

        if isinstance(batched_inputs, dict) and 'pixel_values' in batched_inputs:
            return self.base_model(**batched_inputs)
        return self.base_model(batched_inputs)

from typing import Iterator
import re

from torch import nn

from detectron2.layers import FrozenBatchNorm2d
from transformers.models.rt_detr.modeling_rt_detr import RTDetrFrozenBatchNorm2d

from ....base import AdaptationEngine, BaseModel
from .configuration_norm import NORMConfig


class NORMEngine(AdaptationEngine):
    model_name = "NORMEngine"
    config_class = NORMConfig

    def __init__(self, config: NORMConfig, base_model: BaseModel):
        super().__init__(config, base_model)
        self.config = config

    def _pre_init(self):
        self._norm_modules = []

    def _post_init(self):
        self._apply_norm_adaptation()
        self.to(self.device)

    def _apply_norm_adaptation(self):
        if self.config.model_type == "rcnn":
            self._apply_rcnn_norm()
        elif self.config.model_type == "rtdetr":
            self._apply_rtdetr_norm()
        elif self.config.model_type == "yolo11":
            self._apply_yolo_norm()

    def _apply_rcnn_norm(self):
        # Replace the forward method of the target module with the NORM forward
        for name, module in self.base_model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, FrozenBatchNorm2d)):
                module.adapt_type = "NORM"
                module.source_sum = self.config.source_sum
                self._norm_modules.append(module)

                def norm_forward(self, x):
                    if hasattr(self, 'adapt_type') and self.adapt_type == "NORM":
                        alpha = x.shape[0] / (self.source_sum + x.shape[0])
                        
                        running_mean = (1 - alpha) * self.running_mean + alpha * x.mean(dim=[0, 2, 3])
                        running_var = (1 - alpha) * self.running_var + alpha * x.var(dim=[0, 2, 3])
                        
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

    def _apply_rtdetr_norm(self):
        # Replace the forward method of the target module with the NORM forward
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

                module.adapt_type = "NORM"
                module.source_sum = self.config.source_sum
                self._norm_modules.append(module)

                def norm_forward(self, x):
                    if hasattr(self, 'adapt_type') and self.adapt_type == "NORM":
                        alpha = x.shape[0] / (self.source_sum + x.shape[0])

                        if isinstance(self, nn.BatchNorm2d):
                            running_mean = (1 - alpha) * self.running_mean + alpha * x.mean(dim=[0, 2, 3])
                            running_var = (1 - alpha) * self.running_var + alpha * x.var(dim=[0, 2, 3])
                        
                            scale = self.weight * (running_var + self.eps).rsqrt()
                            bias = self.bias - running_mean * scale
                        
                            scale = scale.reshape(1, -1, 1, 1)
                            bias = bias.reshape(1, -1, 1, 1)
                        
                        elif isinstance(self, RTDetrFrozenBatchNorm2d):
                            eps = getattr(self, 'eps', 1e-5)
                        
                            running_mean = (1 - alpha) * self.running_mean + alpha * x.mean(dim=[0, 2, 3])
                            running_var = (1 - alpha) * self.running_var + alpha * x.var(dim=[0, 2, 3])
                        
                            scale = self.weight * (running_var + eps).rsqrt()
                            bias = self.bias - running_mean * scale
                        
                            scale = scale.reshape(1, -1, 1, 1)
                            bias = bias.reshape(1, -1, 1, 1)
                        
                        elif isinstance(self, nn.LayerNorm):
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

    def _apply_yolo_norm(self):
        # Replace the forward method of the target module with the NORM forward
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

                module.adapt_type = "NORM"
                module.source_sum = self.config.source_sum
                self._norm_modules.append(module)

                def norm_forward(self, x):
                    if hasattr(self, 'adapt_type') and self.adapt_type == "NORM":
                        alpha = x.shape[0] / (self.source_sum + x.shape[0])

                        if isinstance(self, nn.BatchNorm2d):
                            running_mean = (1 - alpha) * self.running_mean + alpha * x.mean(dim=[0, 2, 3])
                            running_var = (1 - alpha) * self.running_var + alpha * x.var(dim=[0, 2, 3])

                            scale = self.weight * (running_var + self.eps).rsqrt()
                            bias = self.bias - running_mean * scale
                            
                            scale = scale.reshape(1, -1, 1, 1)
                            bias = bias.reshape(1, -1, 1, 1)
                            
                            out_dtype = x.dtype
                            out = x * scale.to(out_dtype) + bias.to(out_dtype)
                            return out
                        
                        elif isinstance(self, nn.LayerNorm):
                            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
                        
                    else:
                        if isinstance(self, nn.BatchNorm2d):
                            scale = self.weight * (self.running_var + self.eps).rsqrt()
                            bias = self.bias - self.running_mean * scale
                        
                            scale = scale.reshape(1, -1, 1, 1)
                            bias = bias.reshape(1, -1, 1, 1)
                        
                            out_dtype = x.dtype
                            out = x * scale.to(out_dtype) + bias.to(out_dtype)
                            return out
                        
                        elif isinstance(self, nn.LayerNorm):
                            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

                module.forward = norm_forward.__get__(module, module.__class__)

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

        self._norm_modules = []
        self._apply_norm_adaptation()

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

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn, optim

from ..models.base import BaseModel, ModelProvider, DataPreparation


@dataclass
class AdaptationConfig:
    adaptation_name: str = "AdaptationEngine"
    dataset_name: str = ""
    optim: Literal["SGD", "Adam"] = "SGD"
    adapt_lr: float = 1e-4


class AdaptationEngine(BaseModel):
    model_name: str = "AdaptationEngine"
    model_provider: ModelProvider = ModelProvider.HuggingFace
    loss_class = nn.MSELoss
    DataPreparation = DataPreparation
    class Trainer:
        pass
    class TrainingArguments:
        pass

    def __init__(self, base_model: BaseModel, config: AdaptationConfig):
        super(BaseModel, self).__init__()
        self.config = config
        self.dataset_name = config.dataset_name

        if config.adaptation_name != self.model_name:
            raise ValueError("AdaptationEngine name does not match the adaptation_name on config")
        self.model_name = self.model_name + "(" + base_model.model_name + ")"
        self.model_provider = base_model.model_provider
        self.DataPreparation = base_model.DataPreparation
        self.Trainer = base_model.Trainer

        self.base_model = base_model
        self.base_state = {key: value.cpu() for key, value in base_model.state_dict().items()}
        self.base_grad_state = {key: value.requires_grad for key, value in self.base_state.items()}

        self.adapting = False
        first_param = next(base_model.parameters())
        self._device = first_param.device
        self._dtype = first_param.dtype
        self._loss_function = None
        self._optimizer = None

    @property
    def device(self):
        return self._device

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self._device = torch.device(*args, **kwargs)
        except TypeError:
            pass
        try:
            self._dtype = torch.dtype(*args, **kwargs)
        except TypeError:
            pass
        return result

    @property
    def loss_function(self):
        if self._loss_function is None:
            self._loss_function = self.loss_class().to(self.device)
        return self._loss_function

    def online_parameters(self):
        return self.base_model.parameters()

    @property
    def optimizer(self):
        if self._optimizer is None:
            self._optimizer = optim.SGD(self.online_parameters(), lr=self.config.adapt_lr)
        return self._optimizer

    def online(self, mode=True):
        """Online learning mode (test-time adaptation)"""
        self.adapting = mode

        if mode:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
            for param in self.online_parameters():
                param.requires_grad = True
        else:
            self.train()
            for param in self.parameters():
                param.requires_grad = True
            base_params = dict(self.base_model.named_parameters())
            for key, grad in self.base_grad_state.items():
                if key in base_params:
                    base_params[key].requires_grad = grad

        for module in self.children():
            if hasattr(module, "online"):
                module.online(mode)
        return self

    def offline(self):
        return self.online(False)

    def fit(self, *args, **kwargs):
        """Fitting adaptation engine to base model"""
        pass

    def reset(self):
        """Reset model state"""
        self.base_model.load_state_dict(self.base_state)
        self.online(self.adapting)
        self.to(self.device)
        self.to(self.dtype)
        self.optimizer.zero_grad()

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

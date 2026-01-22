from typing import Literal, Self, Iterator
from dataclasses import dataclass

import torch
from torch import nn, optim
from muon import Muon, MuonWithAuxAdam

from ..models.base import BaseModel, ModelProvider, DataPreparation


class MethodContainer(dict):
    def __init__(self, *args, suffix: str = "Round", **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0
        self.suffix = suffix

    def names(self, round: int = None):
        if round is None:
            self.current_round += 1
            round = self.current_round
        return tuple(f"{name} {self.suffix} {round}" for name in self.keys())

    def methods(self):
        return tuple(self.values())


class OnlineMixin(nn.Module):
    pass



@dataclass
class AdaptationConfig:
    adaptation_name: str = "AdaptationEngine"
    dataset_name: str = ""
    optim: Literal["SGD", "Adam", "AdamW", "Muon", "MuonWithAuxAdam"] = "SGD"
    adapt_lr: float = 1e-4
    momentum: float = 0.9
    verbose: bool = True


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

        self._pre_init()

        self.base_model = base_model
        self.base_state = {key: value.cpu() for key, value in base_model.state_dict().items()}
        self.base_grad_state = {key: value.requires_grad for key, value in self.base_state.items()}

        self.adapting = False
        first_param = next(base_model.parameters())
        self._device = first_param.device
        self._dtype = first_param.dtype
        self._loss_function = None
        self._optimizer = None
        self._stats = {}
        self._reset_stats()

        self._post_init()

    def _pre_init(self):
        """Required init tasks before base model registration"""
        pass

    def _post_init(self):
        """Required init tasks after base model registration"""
        pass

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, *args, **kwargs) -> Self:
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
    def loss_function(self) -> nn.Module:
        if self._loss_function is None:
            self._loss_function = self.loss_class().to(self.device)
        return self._loss_function

    def online_parameters(self) -> Iterator[nn.Parameter]:
        return self.base_model.parameters()

    @property
    def optimizer(self) -> optim.Optimizer:
        if self._optimizer is None:
            if self.config.optim == "SDG":
                self._optimizer = optim.SGD(
                    self.online_parameters(), lr=self.config.adapt_lr, momentum=self.config.momentum
                )
            if self.config.optim == "Adam":
                self._optimizer = optim.Adam(self.online_parameters(), lr=self.config.adapt_lr)
            elif self.config.optim == "AdamW":
                self._optimizer = optim.AdamW(self.online_parameters(), lr=self.config.adapt_lr)
            elif self.config.optim == "Muon":
                self._optimizer = Muon(
                    list(self.online_parameters()), lr=self.config.adapt_lr, momentum=self.config.momentum
                )
            elif self.config.optim == "MuonWithAuxAdam":
                self._optimizer = MuonWithAuxAdam(
                    list(self.online_parameters()),
                    lr=self.config.adapt_lr, adam_lr=self.config.adapt_lr / 10, momentum=self.config.momentum
                )
            else:
                raise NotImplementedError(f"Optimizer {self.config.optim} cannot be recognized or is not implemented yet.")
        return self._optimizer

    def online(self, mode=True) -> Self:
        """Online learning mode (test-time adaptation)"""
        self.adapting = mode

        if mode:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
            for param_group in self.online_parameters():
                if isinstance(param_group, dict):
                    for param in param_group['params']:
                        param.requires_grad = True
                else:
                    param_group.requires_grad = True
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

    def offline(self) -> Self:
        return self.online(False)

    @property
    def stats(self) -> dict:
        return self._stats

    def _reset_stats(self):
        self._stats = {}

    def fit(self, *args, **kwargs):
        """
        Fitting adaptation engine to base model
        Implement this method if the adaptation method requires few-shot pretraining before adaptation
        """
        pass

    def reset(self, reset_stats=False) -> dict | None:
        """
        Reset model state (Return to basemodel state to get new experiment setup)
        Required for StandardTTA, GradualTTA Scenarios
        """
        self.base_model.load_state_dict(self.base_state)
        self.online(self.adapting)
        self.to(self.device)
        self.to(self.dtype)
        try:
            self.optimizer.zero_grad()
        except:
            pass
        if reset_stats:
            current_stats = self._stats
            self._reset_stats()
            return current_stats
        else:
            return None

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

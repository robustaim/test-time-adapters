from typing import Literal, Self, Iterator, ClassVar
from dataclasses import dataclass
import random
import re

import numpy as np

import torch
from torch import nn, optim
from muon import Muon, MuonWithAuxAdam
from transformers import PretrainedConfig, PreTrainedModel, AutoModel
from transformers.models.auto.auto_factory import _BaseAutoModelClass, auto_class_update

from ..models.base import BaseModel, ModelProvider, DataPreparation


class MethodContainer(dict):
    seed_base = 42

    def __init__(self, *args, total_round: int = 1, suffix: str = "Round", **kwargs):
        super().__init__(*args, **kwargs)
        self.current_round = 0
        self.total_round = total_round
        self.suffix = suffix

    def names(self, round: int = None):
        if round is None:
            self.current_round += 1
            round = self.current_round
            self.set_seed(self.seed_base + round)  # fixed seed per rounds
        return tuple(f"{name} {self.suffix} {round}" for name in self.keys())

    def methods(self):
        return tuple(self.values())

    def go_rounds(self, start_round: int | None = None, end_round: int | None = None):
        originals = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if start_round and start_round != self.current_round + 1:
            self.current_round = start_round - 1
        if end_round and end_round != self.total_round:
            self.total_round = end_round

        for i in range(self.current_round + 1, self.total_round + 1):
            yield self.names()

        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = originals

    def set_seed(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


class OnlineMixin(nn.Module):
    pass



@dataclass
class AdaptationConfig(PretrainedConfig):
    adaptation_name: ClassVar[str] = "AdaptationEngine"
    model_type: ClassVar[str] = "adaptation_engine"

    dataset_name: str = ""
    optim: Literal["SGD", "Adam", "AdamW", "Muon", "MuonWithAuxAdam"] = "SGD"
    adapt_lr: float = 1e-4
    momentum: float = 0.9
    verbose: bool = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "adaptation_name" in cls.__dict__:  # auto configuration
            cls.model_type = re.sub(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', '_', cls.adaptation_name).lower()

    @classmethod
    def from_preset(cls, base_model: BaseModel, **kwargs):
        pass


class AdaptationEngine(BaseModel, OnlineMixin, PreTrainedModel):
    model_name: str = "AdaptationEngine"
    model_provider: ModelProvider = ModelProvider.HuggingFace
    loss_class = nn.MSELoss
    DataPreparation = DataPreparation
    class Trainer:
        pass
    class TrainingArguments:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "model_name" in cls.__dict__:  # auto configuration
            cls.model_type = re.sub(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', '_', cls.adaptation_name).lower()

    def __init__(self, config: AdaptationConfig, base_model: BaseModel, **kwargs):
        super(BaseModel, self).__init__(config, **kwargs)
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

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

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
            match self.config.optim:
                case "SGD":
                    self._optimizer = optim.SGD(
                        self.online_parameters(), lr=self.config.adapt_lr, momentum=self.config.momentum
                    )
                case "Adam":
                    self._optimizer = optim.Adam(self.online_parameters(), lr=self.config.adapt_lr)
                case "AdamW":
                    self._optimizer = optim.AdamW(self.online_parameters(), lr=self.config.adapt_lr)
                case "Muon":
                    self._optimizer = Muon(
                        list(self.online_parameters()), lr=self.config.adapt_lr, momentum=self.config.momentum
                    )
                case "MuonWithAuxAdam":
                    params = list(self.online_parameters())
                    hidden_matrix_params = [p for p in params if p.ndim >= 2]  # matrix (apply muon)
                    scalar_params = [p for p in params if p.ndim < 2]  # vector (apply adam)
                    self._optimizer = MuonWithAuxAdam([
                        dict(params=hidden_matrix_params, lr=self.config.adapt_lr, momentum=self.config.momentum, use_muon=True),  # Muon group (Main)
                        dict(params=scalar_params, lr=self.config.adapt_lr / 10, betas=(0.9, 0.95), use_muon=False)  # Adam group (Auxiliary)
                    ])
                case _:
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

    def fit(self, source_dataset: DataPreparation, batch_size=1, max_samples=2000, dtype=torch.float32, **kwargs):
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
        with torch.no_grad():
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

    def __getattr__(self, name):
        """Delegate attribute access to base_model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)


class AutoAdaptationEngine(AutoModel):
    pass


class AutoAdaptationEngineForObjectDetection(AutoModel):
    pass


#class AutoAdaptationEngine(_BaseAutoModelClass):
#    """Auto class for Test-time Adaptation"""
#    _model_mapping = {}


#class AutoAdaptationEngineForObjectDetection(_BaseAutoModelClass):
#    """Auto class for TTA-OD"""
#    _model_mapping = {}


# TODO: Fix auto_class_update
#AutoAdaptationEngine = auto_class_update(AutoAdaptationEngine, head_doc="test time adaptation")
#AutoAdaptationEngineForObjectDetection = auto_class_update(AutoAdaptationEngine, head_doc="test time adaptation - object detection")

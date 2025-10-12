from typing import Optional, Union
from dataclasses import dataclass
import os

from transformers import PreTrainedModel, PretrainedConfig

from ..models.base import BaseModel, ModelProvider, DataPreparation


@dataclass
class AdaptationConfig(PretrainedConfig):
    adaptation_name: str = "AdaptationEngine"
    dataset_name: str = ""


class AdaptationEngine(BaseModel, PreTrainedModel):
    model_name: str = "AdaptationEngine"
    model_provider: ModelProvider = ModelProvider.HuggingFace
    DataPreparation = DataPreparation
    class Trainer:
        pass

    def __init__(self, basemodel: BaseModel, config: AdaptationConfig):
        super(PreTrainedModel, self).__init__()
        self.config = config
        self.dataset_name = config.dataset_name

        if config.adaptation_name != self.model_name:
            raise ValueError("AdaptationEngine name does not match the adaptation_name on config")
        self.model_name = self.model_name + "(" + basemodel.model_name + ")"
        self.model_provider = basemodel.model_provider
        self.DataPreparation = basemodel.DataPreparation
        self.Trainer = basemodel.Trainer

        self.basemodel = basemodel
        self.adapting = False

    def forward(self, *args, **kwargs):
        return self.basemodel(*args, **kwargs)

    def online(self, mode=True):
        """Online learning mode (test-time adaptation)"""
        self.adapting = mode
        for module in self.children():
            if hasattr(module, "online"):
                module.online(mode)
        return self

    def offline(self):
        """Offline mode (static mode)"""
        return self.online(False)

    def fit(self, *args, **kwargs):
        """Fitting adaptation engine to basemodel"""
        pass

    @classmethod
    def from_pretrained(
        cls,
        model_id: Optional[Union[str, os.PathLike]],
        basemodel: BaseModel,
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ) -> "AdaptationEngine":
        model_args = basemodel, *model_args
        return super(PreTrainedModel, cls).from_pretrained(
            pretrained_model_name_or_path=model_id,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            weights_only=weights_only,
            **kwargs
        )

from transformers import PreTrainedModel, PretrainedConfig
from torch import nn


class AdaptationPlugin(nn.Module):
    def __init__(self, basemodel: PreTrainedModel, config: PretrainedConfig):
        super().__init__()
        self.module = PreTrainedModel(config)
        self.module.model = basemodel

        # Combine the configurations
        self.config = config
        if hasattr(basemodel.config, 'to_dict'):
            config_dict = basemodel.config.to_dict()
            for key, value in config_dict.items():
                self.config.__dict__[key] = value
        elif hasattr(basemodel.config, '__dict__'):
            for key, value in basemodel.config.__dict__.items():
                if not key.startswith('_'):  # exclude private attributes
                    self.config.__dict__[key] = value

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self, *args, **kwargs):
        pass

    @classmethod
    def from_pretrained(
            cls,
            model_id: str,
            basemodel: PreTrainedModel
    ) -> "AdaptationPlugin":
        """
        Load a AdaptationPlugin from a pretrained model and configuration.
        """
        config = PretrainedConfig.from_pretrained(model_id)
        module = PreTrainedModel.from_pretrained(model_id)
        module.model = basemodel
        engine = cls(basemodel, config)
        engine.module = module
        return engine

    def save_pretrained(self, save_directory: str):
        """
        Save the AdaptationPlugin to a directory.
        """
        self.module.save_pretrained(save_directory)
        self.config.save_pretrained(save_directory)

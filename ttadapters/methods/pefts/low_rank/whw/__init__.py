from transformers import AutoConfig
from ....base import AutoAdaptationEngine, AutoAdaptationEngineForObjectDetection

from .configuration_whw import WHWConfig, WHWSkipConfig
from .modeling_whw import WHWEngine


class WHWSkipEngine(WHWEngine):
    model_name = "WHWSkipEngine"
    config_class = WHWSkipConfig

    def __init__(self, config: WHWSkipConfig, base_model: BaseModel):
        super().__init__(config, base_model)


AutoConfig.register(WHWConfig.model_type, WHWConfig)
AutoAdaptationEngine.register(WHWConfig, WHWEngine)
AutoAdaptationEngineForObjectDetection.register(WHWConfig, WHWEngine)

AutoConfig.register(WHWSkipConfig.model_type, WHWSkipConfig)
AutoAdaptationEngine.register(WHWSkipConfig, WHWSkipEngine)
AutoAdaptationEngineForObjectDetection.register(WHWSkipConfig, WHWSkipEngine)

from transformers import AutoConfig
from ....base import AutoAdaptationEngine, AutoAdaptationEngineForObjectDetection

from .configuration_test import TeSTConfig
from .modeling_test import TeSTEngine


AutoConfig.register(TeSTConfig.model_type, TeSTConfig)
AutoAdaptationEngine.register(TeSTConfig, TeSTEngine)
AutoAdaptationEngineForObjectDetection.register(TeSTConfig, TeSTEngine)

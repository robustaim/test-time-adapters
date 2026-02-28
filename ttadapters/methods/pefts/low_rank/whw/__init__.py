from transformers import AutoConfig
from ....base import AutoAdaptationEngine, AutoAdaptationEngineForObjectDetection

from .configuration_whw import WHWConfig
from .modeling_whw import WHWEngine


AutoConfig.register(WHWConfig.model_type, WHWConfig)
AutoAdaptationEngine.register(WHWConfig, WHWEngine)
AutoAdaptationEngineForObjectDetection.register(WHWConfig, WHWEngine)

from transformers import AutoConfig
from ....base import AutoAdaptationEngine, AutoAdaptationEngineForObjectDetection

from .configuration_actmad import ActMADConfig
from .modeling_actmad import ActMADEngine


AutoConfig.register(ActMADConfig.model_type, ActMADConfig)
AutoAdaptationEngine.register(ActMADConfig, ActMADEngine)
AutoAdaptationEngineForObjectDetection.register(ActMADConfig, ActMADEngine)

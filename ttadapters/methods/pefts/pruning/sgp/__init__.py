from transformers import AutoConfig
from ....base import AutoAdaptationEngine, AutoAdaptationEngineForObjectDetection

from .configuration_sgp import SGPConfig
from .modeling_sgp import SGPEngine


AutoConfig.register(SGPConfig.model_type, SGPConfig)
AutoAdaptationEngine.register(SGPConfig, SGPEngine)
AutoAdaptationEngineForObjectDetection.register(SGPConfig, SGPEngine)

from transformers import AutoConfig
from ....base import AutoAdaptationEngine, AutoAdaptationEngineForObjectDetection

from .configuration_dua import DUAConfig
from .modeling_dua import DUAEngine


AutoConfig.register(DUAConfig.model_type, DUAConfig)
AutoAdaptationEngine.register(DUAConfig, DUAEngine)
AutoAdaptationEngineForObjectDetection.register(DUAConfig, DUAEngine)

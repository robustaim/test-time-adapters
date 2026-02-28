from transformers import AutoConfig
from ....base import AutoAdaptationEngine, AutoAdaptationEngineForObjectDetection

from .configuration_norm import NORMConfig
from .modeling_norm import NORMEngine


AutoConfig.register(NORMConfig.model_type, NORMConfig)
AutoAdaptationEngine.register(NORMConfig, NORMEngine)
AutoAdaptationEngineForObjectDetection.register(NORMConfig, NORMEngine)

from transformers import AutoConfig
from ...base import AutoAdaptationEngine, AutoAdaptationEngineForObjectDetection

from .configuration_gita import GITAConfig, TargetKeyPreset, TARGET_KEY_PRESET
from .modeling_gita import GITAEngine


AutoConfig.register(GITAConfig.model_type, GITAConfig)
AutoAdaptationEngine.register(GITAConfig, GITAEngine)
AutoAdaptationEngineForObjectDetection.register(GITAConfig, GITAEngine)

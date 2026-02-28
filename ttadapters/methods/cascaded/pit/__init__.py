from transformers import AutoConfig
from ...base import AutoAdaptationEngine, AutoAdaptationEngineForObjectDetection

from .configuration_pit import PITConfig, TargetKeyPreset, TARGET_KEY_PRESET
from .modeling_pit import PITEngine


AutoConfig.register(PITConfig.model_type, PITConfig)
AutoAdaptationEngine.register(PITConfig, PITEngine)
AutoAdaptationEngineForObjectDetection.register(PITConfig, PITEngine)

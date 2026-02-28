from transformers import AutoConfig
from ....base import AutoAdaptationEngine, AutoAdaptationEngineForObjectDetection

from .configuration_mean_teacher import MeanTeacherConfig
from .modeling_mean_teacher import MeanTeacherEngine


AutoConfig.register(MeanTeacherConfig.model_type, MeanTeacherConfig)
AutoAdaptationEngine.register(MeanTeacherConfig, MeanTeacherEngine)
AutoAdaptationEngineForObjectDetection.register(MeanTeacherConfig, MeanTeacherEngine)

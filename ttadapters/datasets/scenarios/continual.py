from .base import BaseContinualTTAScenario, ScenarioType
from .. import (
    SHIFTDiscreteSubsetForObjectDetection,
    ACDCDatasetForObjectDetection
)


DiscreteSubsetType = SHIFTDiscreteSubsetForObjectDetection.SubsetType


class SHIFTDiscreteScenarioForContinualTTA(BaseContinualTTAScenario):
    DEFAULT = [
        DiscreteSubsetType.CLEAR_DAYTIME,  # same as NORMAL
        DiscreteSubsetType.CLEAR_NIGHT,
        DiscreteSubsetType.CLEAR_DAWN,
        DiscreteSubsetType.CLOUDY_DAYTIME,
        DiscreteSubsetType.OVERCAST_DAYTIME,
        DiscreteSubsetType.FOGGY_DAYTIME,
        DiscreteSubsetType.RAINY_DAYTIME,
        DiscreteSubsetType.CLEAR_DAYTIME  # same as NORMAL
    ]
    WHWPAPER = [
        DiscreteSubsetType.CLOUDY_DAYTIME,
        DiscreteSubsetType.OVERCAST_DAYTIME,
        DiscreteSubsetType.FOGGY_DAYTIME,
        DiscreteSubsetType.RAINY_DAYTIME,
        DiscreteSubsetType.CLEAR_DAWN,
        DiscreteSubsetType.CLEAR_NIGHT,
        DiscreteSubsetType.CLEAR_DAYTIME  # same as NORMAL
    ]
    description = "SHIFT-Discrete Scenario For ContinualTTA"
    dataset_class = SHIFTDiscreteSubsetForObjectDetection


class ACDCScenarioForContinualTTA(BaseContinualTTAScenario):
    DEFAULT = [
        ACDCDatasetForObjectDetection.CorruptionType.FOG,
        ACDCDatasetForObjectDetection.CorruptionType.NIGHT,
        ACDCDatasetForObjectDetection.CorruptionType.RAIN,
        ACDCDatasetForObjectDetection.CorruptionType.SNOW,
        #ACDCDatasetForObjectDetection.CorruptionType.NORMAL
    ]
    description = "ACDC Scenario For ContinualTTA"
    dataset_class = ACDCDatasetForObjectDetection

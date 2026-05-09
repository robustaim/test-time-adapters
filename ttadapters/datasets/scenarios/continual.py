from .base import BaseContinualTTAScenario
from .. import (
    SHIFTDiscreteSubsetForObjectDetection,
    CityScapesDiscreteDatasetForObjectDetection,
    ACDCDatasetForObjectDetection
)


SHIFTDiscreteCorruptionType = SHIFTDiscreteSubsetForObjectDetection.SubsetType
CityScapesDiscreteCorruptionType = CityScapesDiscreteDatasetForObjectDetection.CorruptionType
ACDCCorruptionType = ACDCDatasetForObjectDetection.CorruptionType


class SHIFTDiscreteScenarioForContinualTTA(BaseContinualTTAScenario):
    DEFAULT = [
        SHIFTDiscreteCorruptionType.CLOUDY_DAYTIME,
        SHIFTDiscreteCorruptionType.OVERCAST_DAYTIME,
        SHIFTDiscreteCorruptionType.FOGGY_DAYTIME,
        SHIFTDiscreteCorruptionType.RAINY_DAYTIME,
        SHIFTDiscreteCorruptionType.CLEAR_DAWN,
        SHIFTDiscreteCorruptionType.CLEAR_NIGHT,
        SHIFTDiscreteCorruptionType.CLEAR_DAYTIME  # same as NORMAL
    ]
    WHWPAPER = DEFAULT  # What-How-When Paper Setting
    description = "SHIFT-Discrete Scenario For ContinualTTA"
    dataset_class = SHIFTDiscreteSubsetForObjectDetection


class CityScapesDiscreteScenarioForContinualTTA(BaseContinualTTAScenario):
    DEFAULT = [
        CityScapesDiscreteCorruptionType.FOGGY,
        CityScapesDiscreteCorruptionType.RAIN,
        CityScapesDiscreteCorruptionType.SNOW,
        CityScapesDiscreteCorruptionType.FROST,
        CityScapesDiscreteCorruptionType.BRIGHTNESS,
        CityScapesDiscreteCorruptionType.BASE
    ]
    description = "CityScapes-Discrete Scenario For ContinualTTA"
    dataset_class = CityScapesDiscreteDatasetForObjectDetection


class ACDCScenarioForContinualTTA(BaseContinualTTAScenario):
    DEFAULT = [
        ACDCCorruptionType.FOG,
        ACDCCorruptionType.NIGHT,
        ACDCCorruptionType.RAIN,
        ACDCCorruptionType.SNOW,
        ACDCCorruptionType.NORMAL
    ]
    description = "ACDC Scenario For ContinualTTA"
    dataset_class = ACDCDatasetForObjectDetection

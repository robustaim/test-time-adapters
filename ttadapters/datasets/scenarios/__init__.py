from .base import (
    BaseDataset, BaseScenario,
    BaseStandardTTAScenario, BaseGradualTTAScenario,
    BaseContinualTTAScenario, BaseUniversalTTAScenario
)
from .standard import *
from .gradual import (
    SHIFTContinuousScenarioForGradualTTA,
    CityScapesContinuousScenarioForGradualTTA
)
from .continual import (
    SHIFTDiscreteScenarioForContinualTTA,
    ACDCScenarioForContinualTTA,
    CityScapesDiscreteScenarioForContinualTTA,
    CrossCountryScenarioForContinualTTA
)
from .universal import *

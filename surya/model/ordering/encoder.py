from surya.model.recognition.encoder import DonutSwinModel

from surya.model.ordering.config import VariableDonutSwinConfig


class VariableDonutSwinModel(DonutSwinModel):
    config_class = VariableDonutSwinConfig

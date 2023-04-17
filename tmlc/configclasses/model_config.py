from pydantic import BaseModel

from tmlc.configclasses.pretrained_config import PreTrainedConfig
from tmlc.configclasses.partial_function_config import PartialFunctionConfig

class ModelConfig(BaseModel):
    """
    Configuration class for a PyTorch model.

    Attributes:
        pretrained_model (PreTrainedConfig): The configuration for a pre-trained model.
        classifier (PartialFunctionConfig): The configuration for the
            classifier model.
        calculate_predictions (PartialFunctionConfig): The configuration for the
            function to calculate predictions.
    """

    pretrained_model: PreTrainedConfig
    classifier: PartialFunctionConfig
    calculate_predictions: PartialFunctionConfig
    num_labels: int

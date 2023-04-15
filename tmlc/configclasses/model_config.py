from pydantic import BaseModel

from tmlc.configclasses import PartialFunctionConfig, PreTrainedConfig


class ModelConfig(BaseModel):
    """
    Configuration class for a PyTorch model.

    Attributes:
        pretrained_model (PreTrainedConfig): The configuration for a pre-trained model.
        dropout_prob (float): The dropout probability to use.
        hidden_size (int): The size of the hidden layer.
        num_labels (int): The number of output classes.
        calculate_predictions (PartialFunctionConfig): The configuration for the
            function to calculate predictions.
    """

    pretrained_model: PreTrainedConfig
    dropout_prob: float
    hidden_size: int
    num_labels: int
    calculate_predictions: PartialFunctionConfig

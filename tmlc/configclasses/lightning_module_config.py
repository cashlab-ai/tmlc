import yaml
from pydantic import BaseModel
from typing import List, Optional
from tmlc.configclasses import ModelConfig, PartialFunctionConfig


class LightningModuleConfig(BaseModel):
    """
    A data class for storing configuration parameters to create a partial function object.

    Attributes:
        model_name (str): The name of the model.
        model (ModelConfig): The configuration for the model to use.
        optimizer (PartialFunctionConfig): The configuration for the optimizer function.
        define_loss (PartialFunctionConfig): The configuration for the function to define the loss.
        predict (PartialFunctionConfig): The configuration for the function to generate predictions.
        calculate_best_thresholds (PartialFunctionConfig): The configuration for the function
            to calculate the best thresholds.
        calculate_metrics (PartialFunctionConfig): The configuration for the function to
            calculate the metrics.

    Class Methods:
        from_yaml(file_path: str) -> "LightningModuleConfig":
            Class method to create a `LightningModuleConfig` object from a YAML file.
    """

    model_name: str
    model: ModelConfig
    optimizer: PartialFunctionConfig
    define_loss: PartialFunctionConfig
    predict: PartialFunctionConfig
    calculate_best_thresholds: PartialFunctionConfig
    calculate_metrics: PartialFunctionConfig

    @classmethod
    def from_yaml(cls, file_path: str) -> "LightningModuleConfig":
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict["lightningmodule"])


from typing import Optional

import yaml
from pydantic import BaseModel

from tmlc.configclasses.model_config import ModelConfig
from tmlc.configclasses.partial_function_config import PartialFunctionConfig


class LightningModuleConfig(BaseModel):
    """
    A data class for storing configuration parameters to create a PyTorch Lightning module.

    Attributes:
        model_name (str): The name of the PyTorch Lightning module.
        model (ModelConfig): The configuration for the model to use in the PyTorch Lightning module.
        optimizer (PartialFunctionConfig): The configuration for the optimizer function to use in
            the PyTorch Lightning module.
        calculate_loss_weights (Optional[PartialFunctionConfig]): The configuration for the function to
            define the loss weights (optional).
        predict (PartialFunctionConfig): The configuration for the function to generate predictions in
            the PyTorch Lightning module.
        thresholds (PartialFunctionConfig): The configuration for the function to calculate the
            best thresholds in the PyTorch Lightning module.
        calculate_metrics (PartialFunctionConfig): The configuration for the function to calculate the metrics
            in the PyTorch Lightning module.
        pretrain_epochs (Optional[int]): The number of pretraining epochs to use for the PyTorch Lightning
            module (optional).

    Class Methods:
        from_yaml(file_path: str) -> LightningModuleConfig:
            Class method to create a `LightningModuleConfig` object from a YAML file.

    Example:
    
    >>> model_config = ModelConfig(**config_dict["model"])
    >>> optimizer_config = PartialFunctionConfig(**config_dict["optimizer"])
    >>> predict_config = PartialFunctionConfig(**config_dict["predict"])
    >>> best_thresholds_config = PartialFunctionConfig(**config_dict["thresholds"])
    >>> metrics_config = PartialFunctionConfig(**config_dict["calculate_metrics"])
    >>> module_config = LightningModuleConfig(
    ...     model_name="my_module",
    ...     model=model_config,
    ...     optimizer=optimizer_config,
    ...     predict=predict_config,
    ...     thresholds=best_thresholds_config,
    ...     calculate_metrics=metrics_config,
    ... )
    """

    model_name: str
    model: ModelConfig
    optimizer: PartialFunctionConfig
    calculate_loss_weights: Optional[PartialFunctionConfig] = None
    predict: PartialFunctionConfig
    thresholds: PartialFunctionConfig
    calculate_metrics: PartialFunctionConfig
    pretrain_epochs: Optional[int] = 0

    @classmethod
    def from_yaml(cls, file_path: str) -> "LightningModuleConfig":
        """
        Class method to create a `LightningModuleConfig` object from a YAML file.

        Args:
            file_path (str): The path to the YAML file containing the configuration parameters.

        Returns:
            A `LightningModuleConfig` object with the attributes specified in the YAML file.

        Example:
        >>> module_config = LightningModuleConfig.from_yaml("module_config.yaml")
        """
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict["lightningmodule"])

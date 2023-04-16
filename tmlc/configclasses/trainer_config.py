from typing import List, Optional

import yaml
from pydantic import BaseModel

from tmlc.configclasses import (
    DataModuleConfig,
    LightningModuleConfig,
    MLFlowConfig,
    PartialFunctionConfig,
)


class TrainerConfig(BaseModel):
    """
    Configuration class for the PyTorch Lightning trainer.

    Args:
        lightning_module_config (LightningModuleConfig):
            Configuration for the LightningModule.
        data_module_config (DataModuleConfig):
            Configuration for the DataModule.
        mlflow_config (MLFlowConfig):
            Configuration for MLflow tracking.
        callbacks (List[PartialFunctionConfig]):
            List of callbacks to use during training.
        loggers (List[PartialFunctionConfig]):
            List of loggers to use during training.
        kwargs (Optional[dict], optional):
            Additional keyword arguments for the trainer.
            Defaults to None.
        seed (Optional[int], optional):
            The random seed to use.
            Defaults to 42.
        config_path (Optional[str], optional):
            Path to the YAML file containing the configuration.
            Defaults to None.
    """

    lightning_module_config: LightningModuleConfig
    data_module_config: DataModuleConfig
    mlflow_config: MLFlowConfig
    callbacks: List[PartialFunctionConfig]
    loggers: List[PartialFunctionConfig]
    kwargs: Optional[dict] = None
    seed: Optional[int] = 42
    config_path: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Initializes any optional parameters if they were not specified.
        """
        self.kwargs = self.kwargs or {}

    @classmethod
    def from_yaml(cls, file_path: str) -> "TrainerConfig":
        """
        Loads the configuration from a YAML file.

        Args:
            file_path (str): The path to the YAML file.

        Returns:
            TrainerConfig: The configuration object.
        """
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)
            config_dict["trainer_config"]["config_path"] = file_path

        return cls(**config_dict["trainer_config"])

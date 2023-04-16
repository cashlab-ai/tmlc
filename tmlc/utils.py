import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pytorch_lightning as pl
import yaml
from loguru import logger
from pydantic import BaseModel

from tmlc.configclasses import PartialFunctionConfig, TrainerConfig
from tmlc.dataclasses import DataModule
from tmlc.exceptions import PartialFunctionError
from tmlc.model import (
    TextMultiLabelClassificationModel,
)


def json_to_nested_tags(data: Dict[str, Any]):
    """
    Converts a JSON object to a dictionary of nested tags using dot notation.

    Args:
        data: A JSON object.

    Returns:
        A dictionary of nested tags.
    """
    if not isinstance(data, dict):
        raise TypeError("Input data must be a dictionary.")

    tags = {}
    stack = [(key, value) for key, value in data.items()]

    while stack:
        key, value = stack.pop()
        if isinstance(value, dict):
            stack.extend((f"{key}.{sub_key}", sub_value) for sub_key, sub_value in value.items())
        else:
            tags[key] = str(value)

    return tags


def add_suffix_to_filename(filename: str, suffix: str) -> Path:
    """
    Add a suffix to the filename and return a Path object.

    Args:
        filename (str): The filename to modify.
        suffix (str): The suffix to add to the filename.

    Returns:
        Path: A Path object with the modified filename.

    Raises:
        ValueError: If filename is empty or None.
    """
    if not filename:
        raise ValueError("filename cannot be empty or None")
    path = Path(filename)
    new_name = f"{path.stem}_{suffix}{path.suffix}"
    return path.with_name(new_name)


def load_yaml_config(config_path: str, basemodel: BaseModel) -> BaseModel:
    """
    Load a configuration file from a YAML file and validate it against a Pydantic BaseModel.

    Args:
        config_path (str): The path to the YAML configuration file.
        basemodel (BaseModel): The Pydantic BaseModel to validate the
            configuration against.

    Returns:
        A validated instance of the Pydantic BaseModel.
    """
    try:
        config = basemodel.from_yaml(file_path=config_path)
    except FileNotFoundError as e:
        logger.error(f"Config file not found: {config_path}")
        raise e
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise e
    return config


def _partial_to_callable(partial: PartialFunctionConfig) -> Callable:
    """
    This function converts a partial function specified in the configuration file to a callable
    function.

    Args:
        partial (PartialFunctionConfig): The partial function to be converted
            to a callable.

    Returns:
        A callable function resulting from the conversion of the specified
            partial function.

    Raises:
        PartialFunctionError: If the specified partial function fails to be converted
            to a callable.
    """
    try:
        return partial.partial()
    except (TypeError, AttributeError) as e:
        raise PartialFunctionError(f"Failed to convert partial function '{partial.func}' to callable: {e}")


def to_partial_functions_dictionary(partials: List[PartialFunctionConfig]) -> Dict[str, Callable]:
    """
    Converts a list of partial functions specified in the config to a dictionary of callables.

    Args:
        partials: A list of partial functions specified in the config.

    Returns:
        A dictionary of partial functions as callables, with the keys being the
        function names.

    Raises:
        PartialFunctionError: If any partial function fails to be converted to a callable.
    """
    out_partials: Dict[str, Callable] = {}
    for partial in partials:
        out_partials[partial.func] = _partial_to_callable(partial)
    return out_partials


def load_model_data_trainer_config(
    file_path: str, check_point: bool = False
) -> Tuple[TextMultiLabelClassificationModel, DataModule, TrainerConfig]:
    """
    Load the model, data module, and trainer config from a YAML configuration file.

    Args:
        file_path (str): Path to the YAML configuration file.
        check_point (bool, optional): Whether to resume training from the
            checkpointed weights. Defaults to False.

    Returns:
        A tuple of the TextMultiLabelClassificationModel, DataModule, and
        TrainerConfig objects.
    """
    config: TrainerConfig = load_yaml_config(config_path=file_path, basemodel=TrainerConfig)

    if check_point:
        # If you want to resume training from the checkpointed weights,
        # load them back into the model
        model = TextMultiLabelClassificationModel.load_from_checkpoint(file_path)
    else:
        # Initialize model
        model = TextMultiLabelClassificationModel(config.lightning_module_config)

    # Initialize data module
    datamodule = DataModule(config.data_module_config)

    return model, datamodule, config


def setup_trainer(
    config: TrainerConfig,
    loggers: Dict[str, callable],
    callbacks: Dict[str, callable],
    tokenizers_parallelism: bool = False,
) -> Tuple[pl.Trainer, Dict[str, callable], Dict[str, callable]]:
    """
    Initializes and returns a PyTorch Lightning Trainer object with the given configuration,
    loggers, and callbacks.

    Args:
        config: The TrainerConfig object containing the training configuration parameters.
        loggers: A dictionary of loggers, where keys are the logger names
            and values are the corresponding logger objects.
        callbacks: A dictionary of callbacks, where keys are the callback
            names and values are the corresponding callback objects.
        tokenizers_parallelism: Whether to enable parallelism for tokenizers
            for huggingface False is expected. (default: False).

    Returns:
        A tuple containing the following objects:
            - A PyTorch Lightning Trainer object initialized with
                the given configuration, loggers, and callbacks.
            - A dictionary of loggers, where keys are the logger names
                and values are the corresponding logger callables.
            - A dictionary of callbacks, where keys are the callback names
                and values are the corresponding callback callables.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = str(tokenizers_parallelism).lower()

    # Set seed for reproducibility
    pl.seed_everything(config.seed)

    # Initialize trainer
    trainer = pl.Trainer(logger=list(loggers.values()), callbacks=list(callbacks.values()), **config.kwargs)
    return trainer

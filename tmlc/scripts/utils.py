import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import mlflow
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from loguru import logger
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

from tmlc.configclasses import PartialFunctionConfig, TrainerConfig
from tmlc.dataclasses import DataModule
from tmlc.exceptions import PartialFunctionError
from tmlc.model import (
    TextMultiLabelClassificationModel,
    TextMultiLabelClassificationModelWrapperPythonModel,
)


def set_mlflow_tags(config: TrainerConfig) -> None:
    """
    Set MLflow tags from configuration.

    Args:
        config: A configuration object containing training settings.
    """
    logger.info("Set tags to MLFlowLogger")
    tags = json_to_nested_tags(config.dict())
    mlflow.set_tags(tags)


def export_model_to_onnx(model: pl.LightningModule, config: TrainerConfig) -> None:
    """
    Convert a PyTorch model to ONNX format and save it as an artifact.

    Args:
        model: The trained PyTorch model.
        config: A configuration object containing training settings.
    """
    # Set the model to inference mode
    model.eval()

    # Define dummy input
    encoding = config.data_module_config.dataset.tokenizer(["Hello, world!"])
    dummy_input = {"data": {key: torch.tensor(value) for key, value in encoding.items()}}
    dynamic_axes = {
        "data": {0: "batch_size", 1: "sequence"},
        "output": {0: "batch_size", 1: "sequence"},
    }
    torch.onnx.export(
        model,
        dummy_input,
        config.mlflow_config.model_path,
        input_names=["data"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )


def _register_model(
    config: TrainerConfig,
    artifacts: Dict[str, str],
    model,
) -> str:
    """
    Register the model in the MLflow model registry.

    Args:
        config: A configuration object containing training settings.
        model_path: The path of the model file to register.
        artifacts: The artifacts to save with the model.
        model_signature: The signature to use for the model.
        input_example: An example input for the model.

    Returns:
        The URI of the registered model.
    """
    model_path = prepare_model_path(config)
    # Create a model wrapper and save the model
    model_wrapper = TextMultiLabelClassificationModelWrapperPythonModel(
        model_path=model_path,
        tokenizer_path=config.mlflow_config.tokenizer_path,
        thresholds=model.thresholds,
        tokenizer_config=config.data_module_config.dataset.tokenizer_config,
    )
    input_example, output_example = create_input_output_examples()
    model_signature = infer_signature(input_example, output_example)
    model_uri = mlflow.pyfunc.log_model(
        artifact_path=model_path,
        python_model=model_wrapper,
        artifacts=artifacts,
        registered_model_name=config.lightning_module_config.model_name,
        signature=model_signature,
        input_example=input_example,
    )

    # Update the version of the registered model
    models = MlflowClient().get_latest_versions(
        name=config.lightning_module_config.model_name, stages=["None"]
    )

    _update_model_version(config, models)

    return model_uri


def _update_model_version(config: TrainerConfig, models: list) -> None:
    """
    Update the version of the registered model.

    Args:
        config: A configuration object containing training settings.
        models: A list of models to update.
    """
    client = MlflowClient()

    client.update_model_version(
        name=config.lightning_module_config.model_name,
        version=models[0].version,
        description=config.mlflow_config.description,
    )

    for key, value in config.mlflow_config.tags.items():
        client.set_model_version_tag(
            name=config.lightning_module_config.model_name, version=models[0].version, key=key, value=value
        )


def create_artifacts(model: Any, config: TrainerConfig) -> Dict[str, str]:
    """
    Creates a dictionary of MLflow artifacts based on the given configuration.

    Args:
        model (Any): The PyTorch model to save as an artifact.
        config (TrainerConfig): The configuration for the trainer.

    Returns:
        Dict[str, str]: A dictionary of MLflow artifacts.
    """
    # Create artifact folder if it doesn't exist
    if not os.path.exists(config.mlflow_config.artifact_folder):
        os.makedirs(config.mlflow_config.artifact_folder)

    # Convert the PyTorch model to ONNX format and save it as an artifact
    export_model_to_onnx(model, config)

    # Save the tokenizer to a file
    config.data_module_config.dataset.tokenizer.tokenizer.save_pretrained(config.mlflow_config.tokenizer_path)

    # Create artifacts dictionary
    artifacts = {"model": config.mlflow_config.model_path, "tokenizer": config.mlflow_config.tokenizer_path}

    # Optionally include the score script in the artifacts
    if config.mlflow_config.score_script_path:
        artifacts.update({"score": config.mlflow_config.score_script_path})

    return artifacts


def prepare_model_path(config: TrainerConfig) -> str:
    """
    Prepares the path to the model directory based on the given configuration.

    Args:
        config (TrainerConfig): The trainer configuration.

    Returns:
        str: The path to the model directory.
    """
    model_path = f"{config.artifact_folder}{config.lightning_module_config.model_name}"
    try:
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
    except Exception as e:
        logger.error(f"Error removing directory {model_path}: {e}")
    return model_path


def create_input_output_examples() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates example input and output dataframes for testing purposes.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the input
            and output dataframes.
    """
    input_example = pd.DataFrame({"input_text": ["example text"]})
    output_example = pd.DataFrame({"output": [True]})
    return input_example, output_example


def register_model(model, config: TrainerConfig):
    """
    Register a trained model in the MLflow model registry.

    Args:
        model: The trained PyTorch model.
        config: A configuration object containing training settings.
    """
    # Set tags in the MLflow UI from the configuration
    set_mlflow_tags(config)

    artifacts = create_artifacts(model, config)

    model_uri = _register_model(config, artifacts, model)

    logger.info(f"model_uri: {model_uri}")

    return model_uri


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

import os
import shutil
from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
from loguru import logger
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from tmlc.configclasses import TrainerConfig
from tmlc.model import TextMultiLabelClassificationModelWrapperPythonModel
from tmlc.model.utils import export_onnx_and_tokenizer
from tmlc.utils import json_to_nested_tags


def set_mlflow_tags(config: TrainerConfig) -> None:
    """
    Set MLflow tags from configuration.

    Args:
        config: A configuration object containing training settings.
    """
    logger.info("Set tags to MLFlowLogger")
    tags = json_to_nested_tags(config.dict())
    mlflow.set_tags(tags)


def _register_model(
    config: TrainerConfig,
    artifacts: Dict[str, str],
    model,
) -> str:
    """
    Register the PyTorch model in the MLflow model registry and create a model wrapper to use for
    inference.

    This function registers the PyTorch model in the MLflow model registry and creates a
    model wrapper that can be used for inference. The model wrapper includes the path to
    the saved ONNX model, the tokenizer, and any other information needed for inference.

    Args:
        config: A configuration object containing training settings, such as the paths
            to the saved ONNX model and the tokenizer.
        artifacts: A dictionary of additional files to save with the model.
        model: The trained PyTorch model to register.

    Returns:
        The URI of the registered model.

    Example:
    >>> from my_package import _register_model, TrainerConfig
    >>> model = MyPyTorchModel()
    >>> config = TrainerConfig(model_path='my_model.onnx', tokenizer_path='my_tokenizer')
    >>> artifacts = {'requirements.txt': 'requirements.txt'}
    >>> model_uri = _register_model(config, artifacts, model)

    The function expects a trained PyTorch model, a `TrainerConfig` object containing the paths
    to the saved ONNX model and the tokenizer, and a dictionary of additional files to save with
    the model. The output of the function is the URI of the registered model.
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

    This function creates a dictionary of MLflow artifacts based on the given configuration, which includes
    the saved PyTorch model in the ONNX format and the tokenizer used by the model.

    Args:
        model: The trained PyTorch model to save as an artifact.
        config: The configuration for the trainer, including the paths to save the ONNX model and
            the tokenizer.

    Returns:
        A dictionary of MLflow artifacts.

    Example:
    >>> from my_package import create_artifacts, TrainerConfig
    >>> model = MyPyTorchModel()
    >>> config = TrainerConfig(model_path='my_model.onnx', tokenizer_path='my_tokenizer')
    >>> artifacts = create_artifacts(model, config)

    The function expects a trained PyTorch model and a `TrainerConfig` object containing the paths to save the
    ONNX model and the tokenizer. The output of the function is a dictionary of MLflow artifacts.
    """
    # Create artifact folder if it doesn't exist
    if not os.path.exists(config.mlflow_config.artifact_folder):
        os.makedirs(config.mlflow_config.artifact_folder)

    # Convert the PyTorch model to ONNX format and save it as an artifact
    # Save the tokenizer
    export_onnx_and_tokenizer(model, config)

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
    model_path = f"{config.mlflow_config.artifact_folder}{config.lightning_module_config.model_name}"
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
    Register a trained PyTorch model in the MLflow model registry.

    This function registers a trained PyTorch model in the MLflow model registry, including
    creating MLflow artifacts based on the configuration settings.

    Args:
        model: The trained PyTorch model to register.
        config: A configuration object containing training settings, such as the paths to save the ONNX
            model and the tokenizer.

    Returns:
        The URI of the registered model.

    Example:
    >>> from my_package import register_model, TrainerConfig
    >>> model = MyPyTorchModel()
    >>> config = TrainerConfig(model_path='my_model.onnx', tokenizer_path='my_tokenizer')
    >>> model_uri = register_model(model, config)

    The function expects a trained PyTorch model and a `TrainerConfig` object containing the paths to save the
    ONNX model and the tokenizer. The output of the function is the URI of the registered model.
    """
    # Set tags in the MLflow UI from the configuration
    set_mlflow_tags(config)

    artifacts = create_artifacts(model, config)

    model_uri = _register_model(config, artifacts, model)

    logger.info(f"model_uri: {model_uri}")

    return model_uri

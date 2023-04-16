import pytorch_lightning as pl
import torch

from tmlc.configclasses import TrainerConfig

def export_model_to_onnx(model: pl.LightningModule, config: TrainerConfig) -> None:
    """
    Convert a trained PyTorch model to the ONNX format and save it as an artifact.

    ONNX is an open format for representing machine learning models, which allows models to be exported from one
    framework and imported into another. This function exports a PyTorch model to the ONNX format, which can be used
    for inference on a variety of devices.

    Args:
        model: The trained PyTorch model to export to ONNX format.
        config: A configuration object containing training settings, such as the paths to save the ONNX model and the
            tokenizer.

    This function sets the model to inference mode using `model.eval()` to ensure that any dropout or batch normalization
    layers are disabled during the export process. It then defines a `dummy_input` variable and a `dynamic_axes` dictionary
    to define the input and output shapes of the ONNX model. The `torch.onnx.export()` function is used to export the model
    to the ONNX format and save it as an artifact.

    Example:
    >>> from my_package import export_model_to_onnx, TrainerConfig
    >>> model = MyPyTorchModel()
    >>> config = TrainerConfig(model_path='my_model.onnx', tokenizer_path='my_tokenizer')
    >>> export_model_to_onnx(model, config)

    The function expects a trained PyTorch model and a `TrainerConfig` object containing the paths to save the ONNX model and
    the tokenizer. The output of the function is a saved ONNX model and tokenizer.
    """

    # Set the model to inference mode
    model.eval()

    # Define dummy input
    encoding = config.data_module_config.dataset.tokenizer(["Hello, world!"])
    dummy_input = {key: torch.tensor(value) for key, value in encoding.items()}
    dynamic_axes = {key: {0: "batch_size", 1: "sequence"} for key in encoding.keys()}
    dynamic_axes["output"] = {0: "batch_size", 1: "sequence"}
    torch.onnx.export(
        model,
        dummy_input,
        config.mlflow_config.model_path,
        input_names=list(encoding.keys()),
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=False
    )


def export_onnx_and_tokenizer(model: pl.LightningModule, config: TrainerConfig) -> None:
    """
    Export a trained PyTorch model to the ONNX format and save the tokenizer used by the model.

    This function exports the PyTorch model to the ONNX format and saves the tokenizer used by the model for later use during
    inference. The tokenizer is needed to convert text inputs into numerical inputs that the model can process during
    inference.

    Args:
        model: The trained PyTorch model to export to ONNX format.
        config: A configuration object containing training settings, such as the paths to save the ONNX model and the
            tokenizer.

    Example:
    >>> from my_package import export_onnx_and_tokenizer, TrainerConfig
    >>> model = MyPyTorchModel()
    >>> config = TrainerConfig(model_path='my_model.onnx', tokenizer_path='my_tokenizer')
    >>> export_onnx_and_tokenizer(model, config)

    The function expects a trained PyTorch model and a `TrainerConfig` object containing the paths to save the ONNX model and
    the tokenizer. The output of the function is a saved ONNX model and tokenizer.
    """
    config.data_module_config.dataset.tokenizer.tokenizer.save_pretrained(config.mlflow_config.tokenizer_path)
    export_model_to_onnx(model, config)

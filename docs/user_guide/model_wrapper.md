# Model Wrapper

This documentation page provides an extensive overview of the `TextMultiLabelClassificationModelWrapperPythonModel` class, which is a wrapper for multi-label text classification models in the project.

## Overview

The `TextMultiLabelClassificationModelWrapperPythonModel` class wraps an ONNX model for multi-label text classification tasks. The class provides methods to predict class probabilities and labels for input text data. It also handles the process of loading the model and tokenizer configurations.

## Usage

To use the `TextMultiLabelClassificationModelWrapperPythonModel` class, you need to provide the following arguments:

- `model_path` (str): Path to the ONNX model file.
- `tokenizer_config` (TokenizerConfig): Tokenizer configuration object.
- `tokenizer_path` (str): Path to the tokenizer model file.
- `thresholds` (torch.Tensor): Threshold values for class probabilities.

### Example

```python
wrapper = TextMultiLabelClassificationModelWrapperPythonModel(
    model_path="path/to/onnx/model",
    tokenizer_config=tokenizer_config,
    tokenizer_path="path/to/tokenizer",
    thresholds=torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
)
```

## Implementation Details

The `TextMultiLabelClassificationModelWrapperPythonModel` class inherits from the `mlflow.pyfunc.PythonModel` class. This allows for easy integration with MLflow for logging, registering, and serving models.

The class uses ONNX Runtime (`onnxruntime`) for inference, which provides a fast and efficient way to run ONNX models across different platforms and devices.

The tokenizer is configured using the TokenizerConfig class. The tokenizer configuration and path are set in the `_set_tokenizer_config` method.

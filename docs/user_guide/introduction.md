# Full Training and Deployment Guide

This extensive documentation guide combines the training and deployment of the multi-label text classification models using the `TextMultiLabelClassificationModelWrapperPythonModel` wrapper class.

## Overview

The guide will cover the following steps:

1. Preparing the dataset
2. Training script
3. Creating a model wrapper
4. Saving and registering the model

## 1. Preparing the Dataset

Before training the model, ensure that the dataset is formatted properly. The dataset should be in a CSV format with the following columns:

- `id`: Unique identifier for each data point. Is optional
- `comment_text`: Text data for classification.
- One column for each label, with binary values (0 or 1) indicating the presence or absence of the label.

For more information on how to preprocess the data, refer to the [Data Requirements](user_guide/dataset_requirements.md) user guide.

## 2. Training script

Train the multi-label text classification model using the provided training script. Adjust the model and training parameters as needed.

For more information on training the model, refer to the [Training](user_guide/training.md) user guide.

## 3. Creating a Model Wrapper

Once the model has been trained, create an instance of the `TextMultiLabelClassificationModelWrapperPythonModel` class to wrap the model for deployment.

```python
wrapper = TextMultiLabelClassificationModelWrapperPythonModel(
    model_path="path/to/onnx/model",
    tokenizer_config=tokenizer_config,
    tokenizer_path="path/to/tokenizer",
)
```

For more information on the model wrapper, refer to the [Model Wrapper](user_guide/model_wrapper.md) documentation page.

## 4. Saving and Registering the Model

Save the model wrapper using from `tmlc.script.utils.register_model` function. This will create an MLflow artifact containing the model and all associated files, including the ONNX model and tokenizer.

Once the model is saved as an MLflow artifact, it can be deployed using various deployment options supported by MLflow.

For more information on the model saving, refer to the [Training](user_guide/training.md) documentation page.

## 5. Training Job
TODO

## Summary

By following this full training and deployment guide, you can train a multi-label text classification model, create a model wrapper using the `TextMultiLabelClassificationModelWrapperPythonModel` class, and deploy the model using MLflow. This allows you to efficiently use the model for inference and serve it through various deployment options.

For more details on each part of the process, refer to the specific user guides provided in the documentation:

- [Data Requirements](user_guide/dataset_requirements.md)
- [Training](user_guide/training.md)
- [Model Wrapper](user_guide/model_wrapper.md)
- [Model Evaluation](user_guide/model_evaluation.md)

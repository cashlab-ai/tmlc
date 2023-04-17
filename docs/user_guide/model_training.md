# Model Training

This guide explains how to train the `TextMultiLabelClassificationModel` using the PyTorch Lightning framework. The model is designed for multi-label text classification tasks, and it leverages the Transformers library for the underlying backbone.

## Prerequisites

Before you start, make sure you have installed the required libraries:

```bash
poetry install
```

## Training Script

The model training script is responsible for loading the model, data module, and trainer configuration from a YAML file or a checkpoint file. It sets up the trainer with the specified configuration and trains the model using the given dataset.

The training script also logs important metrics during training and testing using MLflow. After training, the model is registered, and its URI is printed for easy access.

### Running the Training Script

To run the training script, you'll need to provide a YAML configuration file that includes the necessary hyperparameters for the model, data module, and trainer. You can also provide a checkpoint file to start training from an existing checkpoint.

To execute the training script, run the following command in your terminal or command prompt:

```bash
python tmlc/scripts/train.py --file-path /path/to/config_or_checkpoint.yaml [--check-point True|False]
```

Replace `filename: checkpoint  # file name for the checkpoint` in your YAML configuration file or checkpoint file. If you're starting from an existing checkpoint, set the --check-point flag to True.

## Example Configuration

Here's an example configuration file for the LightningModuleConfig. This file includes settings for the trainer, model, data module, loggers, and callbacks.

```yaml
# Example configuration file for LightningModuleConfig
trainer_config:
  seed: 42  # seed for reproducibility
  kwargs:
      max_epochs: 1  # maximum number of epochs to train for
      accelerator: cpu  # use CPU for training
      log_every_n_steps: 5  # log after every 5 steps
      enable_progress_bar: True  # show progress bar during training
  mlflow_config:
    artifact_folder: model/
    model_path: model/mode.onnx
    tokenizer_path: model/tokenizer
    description: "This is a description"
    tags:
      one: 1

  callbacks:
      - module: "pytorch_lightning.callbacks.early_stopping"  # early stopping callback to stop training if val_loss doesn't improve
        func: "EarlyStopping"
        kwargs:
            monitor: val_loss  # monitor validation loss
            patience: 5  # number of epochs with no improvement after which training will be stopped
            mode: min  # minimize the monitored quantity
      - module: "pytorch_lightning.callbacks.model_checkpoint"  # model checkpoint callback to save the best model
        func: "ModelCheckpoint"
        kwargs:
            dirpath: 'dirpath'  # directory to save the checkpoint
            filename: checkpoint  # file name for the checkpoint
            monitor: val_loss  # monitor validation loss
            mode: min  # minimize the monitored quantity
            save_top_k: 1  # save the top 1 models
            save_last: True  # save the last checkpoint

  loggers:
      - module: "lightning.pytorch.loggers"  # logger to log the experiment to MLFlow
        func: "MLFlowLogger"
        kwargs:
          experiment_name: experiment_name  # name of the experiment
          tracking_uri: "mlruns"  # URI of the MLFlow server
      - module: "lightning.pytorch.loggers"  # logger to log the experiment to TensorBoard
        func: "TensorBoardLogger"
        kwargs:
          save_dir: tensorboard_logdir  # directory to save the logs
          name: run_name  # name of the run

  lightning_module_config:
    model_name: TMLC  # name of the model
    model:
      pretrained_model:
        path: bert-base-cased
      dropout_prob: 0.1  # dropout probability
      hidden_size: 768  # size of the hidden layer
      num_labels: 1  # number of classes
      calculate_predictions:
        module: "tmlc.components.predictions"  # predictions module
        func: "calculate_predictions"  # calculate predictions function
    calculate_metrics:
        module: "tmlc.components.metrics"  # metrics module
        func: "calculate_metrics"  # calculate metrics function
        kwargs:
          n_iterations: 1000 # Number of iterations for the bootstrap error calculation
          percentile: 2.5 # Lower percentile to calculate symetric error bars in the bootstrap error calculation
    optimizer:
        module: "torch.optim"  # optimizer module
        func: "AdamW"  # optimizer function
        kwargs:
          lr: 0.000001  # learning rate
    calculate_loss_weights:
        module: tmlc.components.loss  # loss module
        func: bceloss_inverse_frequency_weighted  # loss function
    predict:
      module: "torch"  # module to use for prediction
      func: "sigmoid"  # function to use for prediction
    thresholds:
      module: "tmlc.components.thresholds"  # module to use for calculating best thresholds
      func: "thresholds"  # function to use for calculating best thresholds
      kwargs:
          vmin: 0.1  # minimum value for the threshold
          vmax: 0.9  # maximum value for the threshold
          step: 0.05  # step size for the threshold

  data_module_config:
    state_file: "state.bin" # Location of state file
    dataset:
      batch_size: 5 # Batch size for data loader
      tokenizer_config:
        model_name: "bert-base-cased"
        path: "bert-base-cased"
        max_length: 128
        output_keys: ["input_ids", "attention_mask"]
        kwargs:
          add_special_tokens: true
          padding: "max_length"
          truncation: true
      kwargs:
        num_workers: 8 # Number of workers for data loader
    load_data:
      module: "tmlc.components.load_data"
      func: "load_data"
      kwargs:
        file_path: "data/toxic_multi_class.csv" # File path for input data
    process_data:
      module: "tmlc.components.process_data"
      func: "process_data"
      kwargs:
        text_column: "comment_text" # Column name for input text
        labels_columns: ["toxic"] # List of column names for label(s)
    split:
      module: "tmlc.components.split_data"
      func: "split_data"
      kwargs:
        train_ratio: 0.8 # Train split
        val_ratio: 0.1 # Validation split
        test_ratio: 0.1 # Test split
        random_state: 42 # Random seed for splitting data
        labels_columns: ["toxic"] # List of column names for label(s)
```

### Configuration Explanation

The configuration file is organized into several sections:

1. `trainer_config`: Contains settings for the training process, including seed, maximum epochs, and MLflow configuration.
2. `callbacks`: Configures early stopping and model checkpoint callbacks.
3. `loggers`: Sets up loggers for MLflow and TensorBoard.
4. `lightning_module_config`: Defines the model, optimizer, loss function, and other settings related to the model itself.
5. `data_module_config`: Specifies data-related settings such as dataset, tokenizer, data loading, and data splitting.

Each section includes several parameters, which are briefly explained in the comments in the example configuration above. For more details on each parameter, please refer to the documentation of the respective library or module (e.g., PyTorch Lightning, torch.optim, etc.).

When creating your own configuration file, ensure to include all the required sections and parameters, adjusting their values as needed for your specific use case. This configuration file should then be passed to the training script to initialize and train the `TextMultiLabelClassificationModel`.

## Saving the Model

In the model training script `tmlc/scripts/train.py` saves the model, it also registers it to MLflow. The model saving process occurs mainly within the `register_model` function. This function takes the trained model and the trainer configuration as inputs and logs the model to MLflow. It also sets MLflow tags, creates artifacts, and registers the model.

### Training

This code snippet

```python
  from tmlc.configclasses import LightningModuleConfig, TrainConfig
  from tmlc.model import TextMultiLabelClassificationModel
  from tmlc.utils import load_yaml
  from tmlc.data import EmailDataLoader, EmailDataModule

  # Load the configuration YAML file.
  config = load_yaml("config.yaml")

  # Create an instance of the LightningModuleConfig class.
  module_config = LightningModuleConfig(**config["lightning_module_config"])

  # Create an instance of the TextMultiLabelClassificationModel class.
  model = TextMultiLabelClassificationModel(module_config)

  # Create an instance of the TrainConfig class.
  train_config = TrainConfig(**config["train_config"])

  # Create an instance of the EmailDataModule class.
  datamodule = EmailDataModule(train_dataloader=dataloader)

  # Train the model.
  trainer = pl.Trainer(**train_config.to_dict())
  trainer.fit(model, datamodule)
```

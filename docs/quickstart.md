# Quickstart

This quickstart guide provides an overview of the basic usage of the TextMultiLabelClassificationModel project.

## Train a model

To train a model, you need to create a configuration file (in YAML format) with the necessary hyperparameters and settings. You can find an example configuration file in the model documentation.

After creating the configuration file, you can run the training script:

```bash
python train.py --file-path path/to/your/config.yaml
```

Replace path/to/your/config.yaml with the path to your configuration file.

## Evaluate a model

To evaluate a model, you can use the scoring script. First, you need to register the trained model with MLflow. You can find the model name and version in the training script's output.

Run the scoring script with the registered model name and version:

```bash
python score.py --model-name <registered_model_name> --version <model_version> --texts "Text1|Text2|Text3"
```

Replace <registered_model_name> with the name of the registered model and <model_version> with the desired model version. The input texts should be separated by the '|' character.

## Next steps

For more detailed information on the project components and usage, refer to the user guides provided in the documentation:

[Introduction](/user_guide/introduction.md)
[Dataset Requirements](/user_guide/dataset_requirements.md)
[Model Training](/user_guide/model_training.md)
[Model Evaluation](/user_guide/model_evaluation.md)
[Model Wrapper](/user_guide/model_wrapper.md)

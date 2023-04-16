import click
import mlflow
from loguru import logger

from tmlc.utils import (
    load_model_data_trainer_config
)
from tmlc.model.explainability import InterpretabilityModule

file_path = "config.yml"

model, datamodule, config = load_model_data_trainer_config(file_path=file_path)

model.load("/Users/wave/github/toxic/tmlc/model.pt")

# Create an instance of InterpretabilityModule
interpretability_module = InterpretabilityModule(model=model, tokenizer=config.data_module_config.dataset.tokenizer)

# Define a sample input text and target label
input_text = ["This is a sample input text for interpretation."]
target_label = 0

# Attribute the model's predictions to the input features
attributions = interpretability_module.explain(data=input_text, target=target_label)
print(attributions)
import click
from loguru import logger

from tmlc.model import InterpretabilityModule, TextMultiLabelClassificationModel
from tmlc.utils import load_model_data_trainer_config


@click.command()
@click.option(
    "--file-path",
    type=str,
    help="Path to the YAML config file and trained model.",
)
@click.option(
    "--model-path",
    type=str,
    help="Path to the checkpoint of trained model.",
)
@click.option(
    "--input-text",
    type=str,
    default="This is a sample input text for interpretation.",
    help="Input text to explain.",
)
@click.option(
    "--target-label",
    type=int,
    default=0,
    help="Target label to explain.",
)
def explain(
    file_path: str,
    model_path: str,
    input_text: str,
    target_label: int,
) -> None:
    """
    Explains the predictions of the TextMultiLabelClassificationModel on the given input text using
    the specified target label.

    Args:
        file_path: Path to the YAML config file and trained model.
        model_path: Path to the checkpoint of trained model.
        input_text: Input text to explain.
        target_label: Target label to explain.

    Example:
        To explain the predictions of the TextMultiLabelClassificationModel on a sample input text,
        run the following command in the terminal:

        ```
        python tmlc/scripts/interpretability.py --file-path configs/training.yml --model-path model.pt \
              --input-text "This is a sample input text for interpretation." --target-label 0
        ```
    """

    logger.info(f"Loading model and trainer configuration from {file_path}")
    _, _, config = load_model_data_trainer_config(file_path=file_path)

    model = TextMultiLabelClassificationModel.load(model_path)

    # Create an instance of InterpretabilityModule
    interpretability_module = InterpretabilityModule(
        model=model, tokenizer=config.data_module_config.dataset.tokenizer
    )

    # Attribute the model's predictions to the input features
    attributions = interpretability_module.explain(data=[input_text], target=target_label)

    logger.info(f"Attributions: {attributions}")


if __name__ == "__main__":
    explain()

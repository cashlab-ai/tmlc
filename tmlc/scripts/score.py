from typing import List, Optional

import click
import mlflow
from loguru import logger


def score(model_name: str, version: Optional[int] = None, texts: Optional[str] = None) -> List[str]:
    """
    Loads the registered MLflow model by name and version, and makes predictions on the input texts.

    Args:
        model_name : str
            Name of the registered MLflow model to use.
        version : int, optional
            Version of the registered MLflow model to use. If not specified, the
            latest version is used.
        texts : str, optional
            Texts to predict, separated by the '|' character. If not provided, the function
            returns an empty list.

    Returns:
        List[str]
            Predictions made by the model on the input texts.

    Example:
        To make predictions using a registered MLflow model, run the following command
        in the terminal:

        ```
        python your_script.py score --model-name "my_model" --version 1 \
            --texts "text1|text2|text3"
        ```

        This will load the registered MLflow model with the name "my_model" and version 1,
        and make predictions on the texts "text1", "text2", and "text3".
        The predictions will be printed to the console.
    """
    model = mlflow.pytorch.load_model(model_name=model_name, version=version)

    if texts is None:
        logger.info("No texts provided for prediction")
        return []

    # Split the input texts by the '|' character
    messages = texts.split("|")

    # Make predictions
    predictions = model.predict(messages)
    logger.info(f"Predictions: {predictions}")
    return predictions


@click.command()
@click.option("--model-name", type=str, help="Name of the registered MLflow model to use")
@click.option("--version", type=int, default=None, help="Version of the registered MLflow model to use")
@click.option("--texts", type=str, help="Texts to predict, separated by | character")
def cscore(model_name: str, version: Optional[int] = None, texts: Optional[str] = None) -> List[str]:
    return score(model_name, version, texts)


if __name__ == "__main__":
    cscore()

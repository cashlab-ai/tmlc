import click
import mlflow
from loguru import logger

from tmlc.utils import (
    load_model_data_trainer_config,
    setup_trainer,
    to_partial_functions_dictionary,
)
from tmlc.mlflow.register_model import register_model

def train(file_path: str, check_point: bool = False) -> None:
    """
    Trains the TextMultiLabelClassificationModel using the hyperparameters specified in the given
    YAML config file.

    Args:
        file_path: Path to the YAML config file or to the checkpoint if check-point flag
        is True. check_point: Flag to start from existing checkpoint.

    Example:
        To train the TextMultiLabelClassificationModel using a YAML config file, run the
        following command in the terminal:

        ```
        python your_script.py train --file-path /path/to/config.yaml
        ```

        To resume training from an existing checkpoint, set the check-point flag to
        True and provide the path to the checkpoint file:

        ```
        python your_script.py train --file-path /path/to/checkpoint.pt --check-point True
        ```
    """

    logger.info(f"Loading model, data module, and trainer configuration from {file_path}")
    model, datamodule, config = load_model_data_trainer_config(file_path=file_path, check_point=check_point)
    loggers = to_partial_functions_dictionary(config.loggers)
    callbacks = to_partial_functions_dictionary(config.callbacks)

    logger.info("Setting up trainer")
    trainer = setup_trainer(config=config, loggers=loggers, callbacks=callbacks)

    with mlflow.start_run(experiment_id=loggers["MLFlowLogger"].experiment_id) as run:
        run_id = run.info.run_id
        logger.info(f"Run ID: {run_id}")
        loggers["MLFlowLogger"].experiment.delete_run(loggers["MLFlowLogger"]._run_id)
        loggers["MLFlowLogger"]._run_id = run_id

        logger.info("Starting model training")
        trainer.fit(model, datamodule=datamodule)
        model.save("model.pt")

        logger.info("Starting model testing")
        trainer.test(datamodule=datamodule)

        model_uri = register_model(model, config)
        logger.info(f"Model uri: {model_uri}")

@click.command()
@click.option("--file-path", type=str,
    help="Path to the YAML config file or to the checkpoint if check-point flag is True",
)
@click.option("--check-point", type=bool, default=False, help="Flag to start from existing checkpoint")
def ctrain(file_path: str, check_point: bool = False) -> None:
    return train(file_path, check_point)

if __name__ == "__main__":
    ctrain()

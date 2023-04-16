from typing import Any, Dict, Optional

from pydantic import BaseModel


class MLFlowConfig(BaseModel):
    """
    Configuration class for logging a model and tokenizer with MLflow.

    Attributes:
        model_path (str): The path to the saved model.
        tokenizer_path (str): The path to the saved tokenizer.
        score_script_path (Optional[str]): The path to the
            Python script to use for scoring.
        description (str): A description of the model.
        tags (Dict[str, Any]): A dictionary of key-value pairs
            to use as tags when logging the model.
        artifact_folder (str):
            Path to the folder where artifacts will be stored.
    """

    model_path: str
    tokenizer_path: str
    score_script_path: Optional[str] = None
    description: str
    tags: Dict[str, Any]
    artifact_folder: str

from typing import Any, Dict, List

import yaml
from pydantic import BaseModel

from tmlc.configclasses import PartialFunctionConfig, PreTrainedConfig, TokenizerConfig


class SklearnClassifiersConfig(BaseModel):
    clf: PartialFunctionConfig
    hyperparams: Dict[str, Any]


class TransformerModelConfig(BaseModel):
    pretrainedmodel: PreTrainedConfig
    tokenizer: TokenizerConfig


class EDAClassifiersEvaluationConfig(BaseModel):
    output_file: str
    message_column: str
    labels_columns: List[str]
    get_data: PartialFunctionConfig
    split_data: PartialFunctionConfig
    transformer_models: List[TransformerModelConfig]
    classifiers: List[SklearnClassifiersConfig]

    @classmethod
    def from_yaml(cls, file_path: str) -> "EDAClassifiersEvaluationConfig":
        with open(file_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict["exploratory"])

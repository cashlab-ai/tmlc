import pytest
from pydantic import ValidationError

from tmlc.configclasses import DatasetConfig, TokenizerConfig


class TestDatasetConfig:
    def test_valid_config(self):
        config_dict = {
            "tokenizer_config": {"model_name": "bert-base-uncased", "path": "./data"},
            "batch_size": 32,
            "kwargs": {"shuffle": True},
        }
        config = DatasetConfig(**config_dict)
        assert isinstance(config.tokenizer, TokenizerConfig)
        assert config.batch_size == 32
        assert config.kwargs == {"shuffle": True}

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            config_dict = {"batch_size": 32, "kwargs": {"shuffle": True}}
            DatasetConfig(**config_dict)

    def test_invalid_field(self):
        with pytest.raises(ValidationError):
            config_dict = {
                "tokenizer_config": {"model_name": "bert-base-uncased", "path": 42},
                "batchs_size": 32,
            }
            DatasetConfig(**config_dict)

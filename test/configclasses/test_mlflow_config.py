import pytest
from tmlc.configclasses import MLFlowConfig

class TestMLFlowConfig:
    @pytest.fixture
    def mlflow_config_data(self):
        return {
            "model_path": "models/model.bin",
            "tokenizer_path": "tokenizers/tokenizer.json",
            "score_script_path": "scripts/score.py",
            "description": "Test model",
            "tags": {"type": "test", "version": 1},
            "artifact_folder": "artifacts/",
        }

    def test_mlflow_config(self, mlflow_config_data):
        mlflow_config = MLFlowConfig(**mlflow_config_data)
        assert mlflow_config.model_path == mlflow_config_data["model_path"]
        assert mlflow_config.tokenizer_path == mlflow_config_data["tokenizer_path"]
        assert mlflow_config.score_script_path == mlflow_config_data["score_script_path"]
        assert mlflow_config.description == mlflow_config_data["description"]
        assert mlflow_config.tags == mlflow_config_data["tags"]
        assert mlflow_config.artifact_folder == mlflow_config_data["artifact_folder"]

    def test_missing_fields(self):
        with pytest.raises(ValueError):
            MLFlowConfig()


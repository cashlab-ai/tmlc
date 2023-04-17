import pytest
from transformers import AutoModel, PreTrainedModel
from tmlc.configclasses import PreTrainedConfig

class TestPreTrainedConfig:
    @pytest.fixture
    def model_path(self):
        return "bert-base-uncased"

    @pytest.fixture
    def kwargs(self):
        return {"return_dict": True}

    def test_pretrained_config(self, model_path, kwargs):
        pretrained_config = PreTrainedConfig(path=model_path, kwargs=kwargs)
        assert pretrained_config.path == model_path
        assert pretrained_config.kwargs == kwargs
        assert pretrained_config.instance is None

    def test_pretrained_config_model_property(self, model_path, kwargs):
        pretrained_config = PreTrainedConfig(path=model_path, kwargs=kwargs)
        model = pretrained_config.model
        assert isinstance(model, PreTrainedModel)
        assert pretrained_config.instance is model

    def test_missing_fields(self):
        with pytest.raises(ValueError):
            PreTrainedConfig()


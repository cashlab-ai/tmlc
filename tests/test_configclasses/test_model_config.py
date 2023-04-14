from hypothesis import given
from hypothesis import strategies as st

from tmlc.configclasses import ModelConfig


class TestModelConfig:
    def test_create_model_config(self):
        # Test creating a ModelConfig instance
        config = ModelConfig(model_name="test_model", dropout_prob=0.5, hidden_size=100)
        assert isinstance(config, ModelConfig)
        assert config.model_name == "test_model"
        assert config.dropout_prob == 0.5
        assert config.hidden_size == 100

    @given(
        model_name=st.text(),
        dropout_prob=st.floats(min_value=0.0, max_value=1.0),
        hidden_size=st.integers(min_value=0),
    )
    def test_create_model_config_with_hypothesis(self, model_name, dropout_prob, hidden_size):
        # Test creating a ModelConfig instance with Hypothesis
        config = ModelConfig(model_name=model_name, dropout_prob=dropout_prob, hidden_size=hidden_size)
        assert isinstance(config, ModelConfig)
        assert config.model_name == model_name
        assert config.dropout_prob == dropout_prob
        assert config.hidden_size == hidden_size

import pytest
from unittest.mock import MagicMock
from hypothesis import given, strategies as st
from your_module import DatasetConfig, TokenizerConfig
from transformers import PreTrainedTokenizer

class TestDatasetConfig:
    @pytest.fixture(autouse=True)
    def mock_auto_tokenizer_from_pretrained(self, monkeypatch):
        # Mock the `AutoTokenizer.from_pretrained` method
        mock_from_pretrained = MagicMock(spec=PreTrainedTokenizer)
        monkeypatch.setattr("your_module.AutoTokenizer.from_pretrained", mock_from_pretrained)
        return mock_from_pretrained

    @given(
        tokenizer_config=st.builds(
            TokenizerConfig,
            model_name=st.text(min_size=1, max_size=20),
            path=st.text(min_size=1, max_size=20),
            max_length=st.integers(min_value=1, max_value=1000),
            output_keys=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5),
            kwargs=st.fixed_dictionaries({"return_dict": st.booleans()}),
        ),
        batch_size=st.integers(min_value=1, max_value=100),
        kwargs=st.fixed_dictionaries({"shuffle": st.booleans()}),
    )
    def test_dataset_config(
        self, mock_auto_tokenizer_from_pretrained, tokenizer_config, batch_size, kwargs
    ):
        config = DatasetConfig(
            tokenizer_config=tokenizer_config, batch_size=batch_size, kwargs=kwargs
        )

        assert config.tokenizer_config == tokenizer_config
        assert config.batch_size == batch_size
        assert config.kwargs == kwargs

        _ = config.tokenizer

        mock_auto_tokenizer_from_pretrained.assert_called_once_with(tokenizer_config.path, model_name=tokenizer_config.model_name)
        assert isinstance(tokenizer_config.instance, PreTrainedTokenizer)

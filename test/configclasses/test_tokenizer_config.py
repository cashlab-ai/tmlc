import pytest
from unittest.mock import MagicMock
from hypothesis import given, strategies as st
from your_module import TokenizerConfig
from transformers import PreTrainedTokenizer

class TestTokenizerConfig:
    @pytest.fixture(autouse=True)
    def mock_auto_tokenizer_from_pretrained(self, monkeypatch):
        # Mock the `AutoTokenizer.from_pretrained` method
        mock_from_pretrained = MagicMock(spec=PreTrainedTokenizer)
        monkeypatch.setattr("your_module.AutoTokenizer.from_pretrained", mock_from_pretrained)
        return mock_from_pretrained

    @given(
        model_name=st.text(min_size=1, max_size=20),
        path=st.text(min_size=1, max_size=20),
        max_length=st.integers(min_value=1, max_value=1000),
        output_keys=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5),
        kwargs=st.fixed_dictionaries({"return_dict": st.booleans()}),
    )
    def test_tokenizer_config(
        self, mock_auto_tokenizer_from_pretrained, model_name, path, max_length, output_keys, kwargs
    ):
        config = TokenizerConfig(
            model_name=model_name, path=path, max_length=max_length, output_keys=output_keys, kwargs=kwargs
        )

        assert config.model_name == model_name
        assert config.path == path
        assert config.max_length == max_length
        assert config.output_keys == output_keys
        assert config.kwargs == kwargs
        assert config.instance == None

        _ = config.tokenizer

        mock_auto_tokenizer_from_pretrained.assert_called_once_with(path, model_name=model_name)
        assert isinstance(config.instance, PreTrainedTokenizer)

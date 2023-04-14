from unittest.mock import patch

import pytest

from tmlc.configclasses import TokenizerConfig


class TestTokenizerConfig:
    @pytest.fixture()
    def mock_tokenizer(self):
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_auto_tokenizer:
            yield mock_auto_tokenizer

    def test_tokenizer(self, mock_tokenizer):
        tokenizer_config = TokenizerConfig(model_name="bert-base-uncased", path="./data")
        tokenizer_config.tokenizer("hello world")
        mock_tokenizer.assert_called_once_with("./data", model_name="bert-base-uncased")

    def test_tokenizer_response(self, mock_tokenizer):
        tokenizer = TokenizerConfig(model_name="bert-base-uncased", path="path/to/tokenizer")

        # Set up the mock tokenizer to return a specific encoding
        mock_encoding = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        mock_tokenizer.return_value = lambda x: {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        # Call the tokenizer with some input data
        input_data = "This is a test."

        encoding = tokenizer(input_data)

        # Check that the encoding returned by the tokenizer matches the expected encoding
        assert encoding == mock_encoding

import pytest
from typing import List
from transformers import PreTrainedTokenizer

from tmlc.configclasses import TokenizerConfig


@pytest.fixture(scope="module")
def tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig(
        model_name="bert-base-uncased",
        path="bert-base-uncased",
        max_length=512,
        output_keys=["input_ids", "attention_mask"],
    )


class TestTokenizerConfig:
    def test_call(self, tokenizer_config: TokenizerConfig):
        sentences = ["Hello world!", "This is a test."]
        tokenized_data = tokenizer_config(sentences)

        assert isinstance(tokenized_data, dict)
        assert set(tokenized_data.keys()) == {"input_ids", "attention_mask"}
        assert isinstance(tokenized_data["input_ids"], List)
        assert isinstance(tokenized_data["attention_mask"], List)

        input_ids = tokenized_data["input_ids"]
        attention_mask = tokenized_data["attention_mask"]

        assert input_ids == [
            [101, 7592, 2088, 999, 102],
            [101, 2023, 2003, 1037, 3231, 1012, 102],
        ]
        assert attention_mask == [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ]

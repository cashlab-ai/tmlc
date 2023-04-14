import pytest
import torch
from transformers import AutoTokenizer

from tmlc.configclasses import DatasetConfig
from tmlc.dataclasses import Dataset, Message


@pytest.fixture
def sample_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer


@pytest.fixture
def sample_messages():
    messages = [
        Message(text="Hello, how are you?", labels=[0]),
        Message(text="I'm doing great, thank you!", labels=[1]),
        Message(text="That's fantastic to hear!", labels=[2]),
    ]
    return messages


class TestDataset:
    @pytest.fixture
    def dataset(self):
        config = DatasetConfig(tokenizer=sample_tokenizer)
        return Dataset(messages=sample_messages, config=config)

    def test_init(self, dataset):
        assert isinstance(dataset.config, DatasetConfig)
        assert len(dataset.messages) == len(sample_messages)

    def test_len(self, dataset):
        assert len(dataset) == len(sample_messages)

    def test_getitem(self, dataset):
        index = 1
        item = dataset[index]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)

        assert item["input_ids"].shape == (len(sample_messages[index].text),)
        assert item["attention_mask"].shape == (len(sample_messages[index].text),)
        assert item["labels"].shape == (len(sample_messages[index].labels),)

from unittest.mock import MagicMock, patch

import pytest

from tmlc.configclasses import DataModuleConfig
from tmlc.dataclasses import DataModule, Dataset


class TestDataModule:
    @pytest.fixture
    def data_module_config(self):
        config = DataModuleConfig()
        config.state_file = None
        return config

    @pytest.fixture
    def sample_data(self):
        return [
            ("This is a sample text.", [1, 0, 0]),
            ("Another sample text here.", [0, 1, 0]),
            ("One more sample text.", [0, 0, 1]),
        ]

    @pytest.fixture
    def data_module(self, data_module_config):
        return DataModule(config=data_module_config)

    def test_init(self, data_module, data_module_config):
        assert data_module.config == data_module_config
        assert data_module.train_data is None
        assert data_module.val_data is None
        assert data_module.test_data is None

    def test_prepare_data(self, data_module):
        data_module.prepare_data()
        # No exception should be raised, as the method is intentionally left empty.

    @patch("torch.load")
    def test_setup_with_state_file(self, mock_torch_load, data_module_config, sample_data):
        data_module_config.state_file = "fake_state_file"
        data_module = DataModule(config=data_module_config)

        state_dict = {
            "config": data_module_config,
            "train_data": sample_data[:1],
            "val_data": sample_data[1:2],
            "test_data": sample_data[2:],
        }
        mock_torch_load.return_value = state_dict

        data_module.setup()
        assert data_module._data["train"] == sample_data[:1]
        assert data_module._data["val"] == sample_data[1:2]
        assert data_module._data["test"] == sample_data[2:]

    def test_setup_without_state_file(self, data_module, sample_data):
        with patch.object(data_module.config, "get_data") as mock_get_data:
            with patch.object(data_module.config, "split") as mock_split:
                mock_get_data.partial.return_value = sample_data
                mock_split.partial.return_value = (sample_data[:1], sample_data[1:2], sample_data[2:])

                data_module.setup()

        assert data_module._data["train"] == sample_data[:1]
        assert data_module._data["val"] == sample_data[1:2]
        assert data_module._data["test"] == sample_data[2:]

    def test_dataloader(self, data_module, sample_data):
        with patch.object(data_module.config, "process_data") as mock_process_data:
            with patch("tmlc.datamodule.DataLoader") as mock_dataloader:
                mock_process_data.partial.return_value = sample_data
                dataset = Dataset(messages=sample_data, config=data_module.config.dataset)

                data_module._data["train"] = sample_data
                _ = data_module._dataloader(element="train")

        mock_process_data.partial.assert_called_once_with(sample_data)
        mock_dataloader.assert_called_once_with(dataset, batch_size=data_module.config.dataset.batch_size)

    def test_train_dataloader(self, data_module, sample_data):
        data_module._dataloader = MagicMock()
        data_module._data["train"] = sample_data
        data_module.train_dataloader()
        data_module._dataloader.assert_called_once_with(element="train")

    def test_val_dataloader(self, data_module, sample_data):
        data_module._dataloader = MagicMock()
        data_module._data["val"] = sample_data
        data_module.val_dataloader()
        data_module._dataloader.assert_called_with(element="val")

    def test_test_dataloader(self, data_module, sample_data):
        data_module._dataloader = MagicMock()
        data_module._data["test"] = sample_data
        data_module.test_dataloader()
        data_module._dataloader.assert_called_with(element="test")

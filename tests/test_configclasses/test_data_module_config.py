import pytest

from tmlc.configclasses import DataModuleConfig, DatasetConfig, PartialFunctionConfig


class TestDataModuleConfig:
    @pytest.fixture()
    def dataset_config(self):
        return DatasetConfig(tokenizer_config={"model_name": "bert-base-uncased", "path": "./data"})

    @pytest.fixture()
    def get_data_config(self):
        return PartialFunctionConfig(module="tmlc.components.get_data", func="get_data")

    @pytest.fixture()
    def split_config(self):
        return PartialFunctionConfig(module="tmlc.components.split_data", func="split_data")

    @pytest.fixture()
    def process_data(self):
        return PartialFunctionConfig(module="tmlc.components.process_data", func="process_data")

    def test_data_module_config(self, dataset_config, get_data_config, split_config, process_data):
        config = DataModuleConfig(
            state_file="./data_module_state.pkl",
            dataset=dataset_config,
            get_data=get_data_config,
            split=split_config,
            process_data=process_data,
        )

        assert config.state_file == "./data_module_state.pkl"
        assert config.dataset == dataset_config
        assert config.get_data == get_data_config
        assert config.split == split_config
        assert config.process_data == process_data

    def test_data_module_config_invalid(self):
        with pytest.raises(ValueError):
            _ = DataModuleConfig(
                state_file="./data_module_state.pkl",
                dataset=None,
                get_data=None,
                split=None,
                process_data=None,
            )

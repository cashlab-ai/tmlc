import pytest
from unittest.mock import MagicMock
from hypothesis import given, strategies as st
from your_module import DataModuleConfig, DatasetConfig, PartialFunctionConfig, TokenizerConfig
from transformers import PreTrainedTokenizer

class Test_DataModuleConfig:
    @given(
        state_file=st.text(min_size=1, max_size=20),
        dataset=st.builds(
            DatasetConfig,
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
        ),
        load_data=st.builds(
            PartialFunctionConfig,
            module=st.text(min_size=1, max_size=20),
            func=st.text(min_size=1, max_size=20),
            args=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5),
            kwargs=st.fixed_dictionaries({"key": st.text(min_size=1, max_size=20)}),
        ),
        split=st.builds(
            PartialFunctionConfig,
            module=st.text(min_size=1, max_size=20),
            func=st.text(min_size=1, max_size=20),
            args=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5),
            kwargs=st.fixed_dictionaries({"key": st.text(min_size=1, max_size=20)}),
        ),
        process_data=st.builds(
            PartialFunctionConfig,
            module=st.text(min_size=1, max_size=20),
            func=st.text(min_size=1, max_size=20),
            args=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5),
            kwargs=st.fixed_dictionaries({"key": st.text(min_size=1, max_size=20)}),
        ),
    )
    def test_datamodule_config(
        self,
        state_file,
        dataset,
        load_data,
        split,
        process_data,
    ):
        config = DataModuleConfig(
            state_file=state_file,
            dataset=dataset,
            load_data=load_data,
            split=split,
            process_data=process_data,
        )

        assert config.state_file == state_file
        assert config.dataset == dataset
        assert config.load_data == load_data
        assert config.split == split
        assert config.process_data == process_data

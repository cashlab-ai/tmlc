import pytest
from unittest.mock import MagicMock
from hypothesis import given, strategies as st
from your_module import ModelConfig, PartialFunctionConfig, PreTrainedConfig

class Test_ModelConfig:
    @given(
        pretrained_model=st.builds(
            PreTrainedConfig,
            path=st.text(min_size=1, max_size=20),
            kwargs=st.fixed_dictionaries({"return_dict": st.booleans()}),
            instance=st.none(),
        ),
        classifier=st.builds(
            PartialFunctionConfig,
            module=st.text(min_size=1, max_size=20),
            func=st.text(min_size=1, max_size=20),
            args=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5),
            kwargs=st.fixed_dictionaries({"key": st.text(min_size=1, max_size=20)}),
        ),
        calculate_predictions=st.builds(
            PartialFunctionConfig,
            module=st.text(min_size=1, max_size=20),
            func=st.text(min_size=1, max_size=20),
            args=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=5),
            kwargs=st.fixed_dictionaries({"key": st.text(min_size=1, max_size=20)}),
        ),
    )
    def test_model_config(
        self,
        pretrained_model,
        classifier,
        calculate_predictions,
    ):
        config = ModelConfig(
            pretrained_model=pretrained_model,
            classifier=classifier,
            calculate_predictions=calculate_predictions,
        )

        assert config.pretrained_model == pretrained_model
        assert config.classifier == classifier
        assert config.calculate_predictions == calculate_predictions

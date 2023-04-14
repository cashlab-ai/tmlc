from unittest.mock import MagicMock

import onnxruntime as rt
import pandas as pd
import pytest
import torch

from tmlc.configclasses import TokenizerConfig
from tmlc.model.modelwrapper import TextMultiLabelClassificationModelWrapperPythonModel


class TestTextMultiLabelClassificationModelWrapperPythonModel:
    @pytest.fixture
    def mock_model(self):
        return MagicMock(spec=TextMultiLabelClassificationModelWrapperPythonModel)

    @pytest.fixture
    def thresholds(self):
        return torch.tensor([0.5] * 3)

    @pytest.fixture
    def sample_input_data(self):
        return pd.DataFrame({"input_text": ["This is a sample text."]})

    def test_init(self, mock_model, thresholds):
        model = TextMultiLabelClassificationModelWrapperPythonModel(
            model_path="fake_model_path",
            tokenizer_config=TokenizerConfig(),
            tokenizer_path="fake_tokenizer_path",
            thresholds=thresholds,
        )
        assert model.model_path == "fake_model_path"
        assert model.tokenizer_config.path == "fake_tokenizer_path"
        assert torch.equal(model.thresholds, thresholds)

    def test_load_context(self, mock_model, thresholds):
        model = TextMultiLabelClassificationModelWrapperPythonModel(
            model_path="fake_model_path",
            tokenizer_config=TokenizerConfig(),
            tokenizer_path="fake_tokenizer_path",
            thresholds=thresholds,
        )
        model.load_context(None)
        assert isinstance(model.tokenizer, TokenizerConfig)
        assert isinstance(model.model, rt.InferenceSession)

    def test_predict_logits(self, mock_model, sample_input_data):
        model = MagicMock(spec=TextMultiLabelClassificationModelWrapperPythonModel)
        model.predict_logits(None, input_data=sample_input_data)
        model.assert_called_once()

    def test_predict(self, mock_model, sample_input_data, thresholds):
        model = MagicMock(spec=TextMultiLabelClassificationModelWrapperPythonModel)
        model.predict(None, input_data=sample_input_data, thresholds=thresholds)
        model.assert_called_once()

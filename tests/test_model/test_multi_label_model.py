from unittest.mock import MagicMock

import pytest
import torch

from tmlc.configclasses import LightningModuleConfig
from tmlc.model import TextMultiLabelClassificationModel


class TestTextMultiLabelClassificationModel:
    @pytest.fixture(scope="class")
    def config(self):
        config = LightningModuleConfig(
            model=dict(pretrained_model_name="distilbert-base-uncased", hidden_size=768, num_labels=3),
            optimizer=dict(class_name="torch.optim.Adam", params=dict(lr=0.001)),
            define_loss=MagicMock(),
            predict=MagicMock(),
            calculate_best_thresholds=MagicMock(),
        )
        return config

    @pytest.fixture(scope="class")
    def model(self, config):
        return TextMultiLabelClassificationModel(config)

    def test_init(self, model):
        assert model.backbone is not None
        assert model.classifier is not None
        assert model.config is not None

    def test_forward(self, model):
        input_ids = torch.randint(0, 10000, (2, 10))
        attention_mask = torch.ones((2, 10))
        logits = model.forward(input_ids, attention_mask)
        assert logits.shape == (2, model.config.model.num_labels)

    def test_training_step(self, model):
        batch = {
            "input_ids": torch.randint(0, 10000, (2, 10)),
            "attention_mask": torch.ones((2, 10)),
            "labels": torch.randint(0, 2, (2, model.config.model.num_labels)),
        }
        loss = model.training_step(batch, 0)
        assert loss is not None

    def test_validation_step(self, model):
        batch = {
            "input_ids": torch.randint(0, 10000, (2, 10)),
            "attention_mask": torch.ones((2, 10)),
            "labels": torch.randint(0, 2, (2, model.config.model.num_labels)),
        }
        loss = model.validation_step(batch, 0)
        assert loss is not None

    def test_test_step(self, model):
        batch = {
            "input_ids": torch.randint(0, 10000, (2, 10)),
            "attention_mask": torch.ones((2, 10)),
            "labels": torch.randint(0, 2, (2, model.config.model.num_labels)),
        }
        loss = model.test_step(batch, 0)
        assert loss is not None

    def test_configure_optimizers(self, model):
        optimizer = model.configure_optimizers()
        assert optimizer is not None

    def test_save_and_load(self, model, config, tmp_path):
        filename = tmp_path / "model.pt"
        model.save(filename)
        loaded_model = TextMultiLabelClassificationModel.load(filename)
        assert loaded_model is not None
        assert isinstance(loaded_model, TextMultiLabelClassificationModel)
        assert loaded_model.config == config

    def test_predict_logits(self, model):
        input_ids = torch.randint(0, 10000, (2, 10))
        attention_mask = torch.ones((2, 10))
        logits = model.predict_logits(input_ids, attention_mask)
        assert logits.shape == (2, model.config.model.num_labels)

    def test_predict(self, model):
        input_ids = torch.randint(0, 10000, (2, 10))
        attention_mask = torch.ones((2, 10))
        predictions = model.predict(input_ids, attention_mask)
        assert predictions.shape == (2, model.config.model.num_labels)

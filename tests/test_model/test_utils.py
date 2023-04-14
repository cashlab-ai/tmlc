from typing import Dict, List

import pytest
import torch

from tmlc.model import aggregate_outputs, calculate_metrics, calculate_predictions


class TestMetrics:
    @pytest.fixture
    def sample_outputs(self) -> List[Dict[str, torch.Tensor]]:
        outputs = [
            {
                "loss": torch.tensor(0.6),
                "logits": torch.tensor([[0.1, 0.9], [0.7, 0.3]]),
                "labels": torch.tensor([[0, 1], [1, 0]]),
            },
            {
                "loss": torch.tensor(0.7),
                "logits": torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
                "labels": torch.tensor([[1, 0], [0, 1]]),
            },
        ]
        return outputs

    def test_aggregate_outputs(self, sample_outputs):
        mean_loss, logits, labels = aggregate_outputs(sample_outputs)

        assert isinstance(mean_loss, torch.Tensor)
        assert mean_loss.item() == 0.65
        assert logits.shape == (4, 2)
        assert labels.shape == (4, 2)

    def test_calculate_predictions(self):
        probabilities = torch.tensor([[0.1, 0.9], [0.7, 0.3], [0.8, 0.2], [0.1, 0.9]])
        thresholds = torch.tensor([0.5, 0.5])
        predictions = calculate_predictions(probabilities, thresholds)

        assert predictions.shape == (4, 2)
        predictions_expected = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=torch.float32)
        assert torch.all(predictions == predictions_expected)

    def test_calculate_metrics(self):
        labels = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=torch.float32)
        predictions = torch.tensor([[0, 1], [1, 0], [1, 0], [0, 1]], dtype=torch.float32)
        metrics = calculate_metrics(labels, predictions, prefix="test")

        assert isinstance(metrics, dict)
        for key in ["test_f1", "test_precision", "test_recall", "test_accuracy"]:
            assert key in metrics
            assert isinstance(metrics[key], float)

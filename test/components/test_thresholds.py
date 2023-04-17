import pytest
import torch
from torchmetrics.classification import BinaryF1Score

from tmlc.components.thresholds import calculate_best_thresholds

class TestCalculateBestThresholds:

    @pytest.fixture
    def probs_labels(self):
        probabilities = torch.Tensor([[0.6, 0.4], [0.7, 0.3], [0.8, 0.2]])
        labels = torch.Tensor([[1, 0], [0, 1], [1, 1]])
        return probabilities, labels

    def test_thresholds_default_metric(self, probs_labels):
        probabilities, labels = probs_labels
        thresholds = calculate_best_thresholds(probabilities, labels, 0.1, 0.9, 0.1)
        assert torch.isclose(thresholds, torch.tensor([0.5, 0.1])).all()

    def test_thresholds_custom_metric(self, probs_labels):
        probabilities, labels = probs_labels
        metric = BinaryF1Score()
        thresholds = calculate_best_thresholds(probabilities, labels, 0.1, 0.9, 0.1, metric)
        assert torch.isclose(thresholds, torch.tensor([0.5, 0.1])).all()

    def test_thresholds_vmin_vmax_out_of_range(self, probs_labels):
        probabilities, labels = probs_labels
        with pytest.raises(ValueError, match=r"vmin must be within the range \(0, 1\)"):
            calculate_best_thresholds(probabilities, labels, -0.1, 0.9, 0.1)
        with pytest.raises(ValueError, match=r"vmax must be within the range \(0, 1\)"):
            calculate_best_thresholds(probabilities, labels, 0.1, 1.1, 0.1)

    def test_thresholds_vmax_less_than_vmin(self, probs_labels):
        probabilities, labels = probs_labels
        with pytest.raises(ValueError, match=r"vmax \(.+\) must be greater than vmin \(.+\)"):
            calculate_best_thresholds(probabilities, labels, 0.6, 0.5, 0.1)

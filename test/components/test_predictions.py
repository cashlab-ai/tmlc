import pytest
import torch
from hypothesis import given, settings
from hypothesis.strategies import lists, floats
from hypothesis.extra.numpy import arrays
from your_module import calculate_predictions


class TestCalculatePredictions:
    @given(
        probabilities=arrays(
            dtype=torch.float32,
            shape=(integers(min_value=1, max_value=50), integers(min_value=1, max_value=10)),
            elements=floats(min_value=0, max_value=1),
        ),
        thresholds=lists(elements=floats(min_value=0, max_value=1), min_size=1, max_size=10),
    )
    @settings(deadline=None)
    def test_calculate_predictions(self, probabilities, thresholds):
        probabilities = torch.tensor(probabilities, dtype=torch.float32)
        num_labels = probabilities.shape[1]

        if len(thresholds) != num_labels:
            pytest.skip("Number of thresholds does not match the number of labels")

        thresholds = torch.tensor(thresholds, dtype=torch.float32)
        predictions = calculate_predictions(probabilities, thresholds)

        assert predictions.shape == probabilities.shape

        for i in range(probabilities.shape[0]):
            for j in range(num_labels):
                assert predictions[i, j] == (probabilities[i, j] > thresholds[j]).float()


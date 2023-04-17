import pytest
import torch
from tmlc.components.predictions import calculate_predictions

class TestCalculatePredictions:
    @pytest.fixture
    def probabilities(self):
        return torch.tensor([[0.3, 0.7, 0.1], [0.1, 0.2, 0.9], [0.7, 0.3, 0.4]])

    @pytest.fixture
    def thresholds(self):
        return torch.tensor([0.5, 0.3, 0.8])

    @pytest.fixture
    def expected_output(self):
        return torch.tensor([[False, True, False], [False, False, True], [True, False, False]])

    def test_calculate_predictions_example_A(self, probabilities, thresholds, expected_output):
        result = calculate_predictions(probabilities, thresholds)
        assert torch.all(result.eq(expected_output))

    @pytest.fixture
    def probabilities_example_B(self):
        return torch.tensor([[0.3, 0.7], [0.1, 0.2], [0.7, 0.3], [0.5, 0.4]])

    @pytest.fixture
    def thresholds_example_B(self):
        return torch.tensor([0.4, 0.6])

    @pytest.fixture
    def expected_output_example_B(self):
        return torch.tensor([[False, True], [False, False], [True, False], [True, False]])

    def test_calculate_predictions_example_B(self, probabilities_example_B, thresholds_example_B, expected_output_example_B):
        result = calculate_predictions(probabilities_example_B, thresholds_example_B)
        assert torch.all(result.eq(expected_output_example_B))

    def test_calculate_predictions_with_empty_input(self):
        probabilities_empty = torch.tensor([[]])
        thresholds_empty = torch.tensor([])
        expected_output_empty = torch.tensor([[]])
        result = calculate_predictions(probabilities_empty, thresholds_empty)
        assert torch.all(result.eq(expected_output_empty))

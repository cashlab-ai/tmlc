import pytest
import torch
import torchmetrics
from hypothesis import given, strategies as st

from calculate_best_thresholds import calculate_best_thresholds

class TestCalculateBestThresholds:

    @given(
        probs=st.lists(st.lists(st.floats(0, 1), min_size=1, max_size=10), min_size=1, max_size=1000),
        labels=st.lists(st.lists(st.integers(0, 1), min_size=1, max_size=10), min_size=1, max_size=1000),
        vmin=st.floats(0.01, 0.99),
        vmax=st.floats(0.01, 0.99),
        step=st.floats(0.01, 0.99),
    )
    def test_calculate_best_thresholds_valid_input(self, probs, labels, vmin, vmax, step):
        # Ensure that the input satisfies the input constraints
        if vmin >= vmax:
            return

        probs_tensor = torch.tensor(probs)
        labels_tensor = torch.tensor(labels)

        assert probs_tensor.shape == labels_tensor.shape, "Probabilities and labels must have the same shape"

        thresholds = calculate_best_thresholds(probs_tensor, labels_tensor, vmin, vmax, step)

        assert thresholds.shape == (labels_tensor.shape[1],), "Output shape must be (num_labels,)"

    @given(
        vmin=st.floats(-1, 2),
        vmax=st.floats(-1, 2),
    )
    def test_calculate_best_thresholds_invalid_vmin_vmax(self, vmin, vmax):
        probs = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.8, 0.2]])
        labels = torch.tensor([[1, 0], [0, 1], [1, 1]])
        step = 0.1

        if 0 < vmin < 1 and 0 < vmax < 1 and vmax > vmin:
            return  # Skip valid input

        with pytest.raises(ValueError):
            calculate_best_thresholds(probs, labels, vmin, vmax, step)

    def test_calculate_best_thresholds_example(self):
        probs = torch.tensor([[0.6, 0.4], [0.7, 0.3], [0.8, 0.2]])
        labels = torch.tensor([[1, 0], [0, 1], [1, 1]])
        vmin = 0.1
        vmax = 0.9
        step = 0.1

        thresholds = calculate_best_thresholds(probs, labels, vmin, vmax, step)

        assert torch.allclose(thresholds, torch.tensor([0.5, 0.2]), atol=1e-6)

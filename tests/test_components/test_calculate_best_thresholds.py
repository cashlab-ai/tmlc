import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st

from tmlc.components.calculate_best_thresholds import calculate_best_thresholds


class TestCalculateBestThresholds:
    @given(
        probabilities=st.lists(
            st.lists(st.floats(min_value=0, max_value=1), min_size=5, max_size=5), min_size=10, max_size=10
        ),
        labels=st.lists(
            st.lists(st.integers(min_value=0, max_value=1), min_size=5, max_size=5), min_size=10, max_size=10
        ),
        num_labels=st.integers(min_value=10, max_value=10),
        vmin=st.floats(min_value=0.1, max_value=0.4),
        vmax=st.floats(min_value=0.5, max_value=0.9),
        step=st.floats(min_value=0.2, max_value=0.4),
    )
    def test_calculate_best_thresholds(self, probabilities, num_labels, labels, vmin, vmax, step):
        probabilities = torch.tensor(probabilities).reshape(-1, num_labels)
        labels = torch.tensor(probabilities).reshape(-1, num_labels)

        best_thresholds = calculate_best_thresholds(probabilities, labels, vmin, vmax, step)

        assert isinstance(best_thresholds, np.ndarray)
        assert best_thresholds.shape == (num_labels,)

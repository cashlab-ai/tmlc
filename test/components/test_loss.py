import pytest
import torch
from hypothesis import given, strategies as st

from bceloss_inverse_frequency_weighted import bceloss_inverse_frequency_weighted

class TestBCELossInverseFrequencyWeighted:

    @given(
        labels=st.lists(st.lists(st.integers(0, 1), min_size=1, max_size=10), min_size=1, max_size=1000),
    )
    def test_bceloss_inverse_frequency_weighted_valid_input(self, labels):
        labels_tensor = torch.tensor(labels)
        pos_weight = bceloss_inverse_frequency_weighted(labels_tensor)
        assert isinstance(pos_weight, dict) and "pos_weight" in pos_weight
        assert pos_weight["pos_weight"].shape == (labels_tensor.shape[1],)

    def test_bceloss_inverse_frequency_weighted_example(self):
        labels = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        pos_weight = bceloss_inverse_frequency_weighted(labels)
        expected_weights = torch.tensor([1.6667, 1.25, 1.6667])
        assert torch.allclose(pos_weight["pos_weight"], expected_weights, atol=1e-4)

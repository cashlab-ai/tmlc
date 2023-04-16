import pytest
import torch
import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays
from sklearn.metrics import f1_score, precision_score, recall_score
from your_module import calculate_metrics


class TestCalculateMetrics:
    @given(
        labels=arrays(
            shape=(4, 3),
            dtype=bool,
            elements=bool,
        ),
        predictions=arrays(
            shape=(4, 3),
            dtype=float,
            elements=floats(0, 1),
        ),
        element=str,
    )
    @settings(deadline=None)
    def test_calculate_metrics(self, labels, predictions, element):
        # Convert labels to torch.Tensor
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        predictions_tensor = torch.tensor(predictions, dtype=torch.float32)

        metrics = calculate_metrics(
            labels=labels_tensor,
            predictions=predictions_tensor,
            element=element,
        )

        # Assert that the calculated metrics match the expected values
        assert np.isclose(
            metrics[f"{element}_f1"], f1_score(labels, predictions, average="macro")
        )
        assert np.isclose(
            metrics[f"{element}_precision"],
            precision_score(labels, predictions, average="macro", zero_division=0),
        )
        assert np.isclose(
            metrics[f"{element}_recall"], recall_score(labels, predictions, average="macro")
        )

        # Assert that the rates are within a valid range
        for rate_name in ["tpr", "fpr", "fnr", "tnr"]:
            assert 0 <= metrics[f"{element}_{rate_name}"] <= 1

import pytest
import torch
import numpy as np
from tmlc.components.metrics import calculate_metrics, evaluate_rates, bootstrap_rates

class TestMetrics:
    @pytest.fixture()
    def labels_and_predictions(self):
        labels = torch.tensor([
            [1],
            [0],
            [1],
            [0]
        ])
        predictions = torch.tensor([
            [0.8],
            [0.3],
            [0.7],
            [0.2]
        ]) > 0.5
        return labels, predictions

    def test_calculate_metrics(self, labels_and_predictions):
        labels, predictions = labels_and_predictions
        metrics = calculate_metrics(labels, predictions, element="test", n_iterations=50, percentile=5.0)
        assert metrics is not None
        assert isinstance(metrics, dict)
        assert len(metrics) == 16

    def test_evaluate_rates(self):
        y_true = np.array([1, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        rates = evaluate_rates(y_true, y_pred)
        assert rates is not None
        assert isinstance(rates, dict)
        assert len(rates) == 5

    def test_bootstrap_rates(self):
        y_true = np.random.randint(0, 2, size=100)
        y_pred = np.random.randint(0, 2, size=100)
        results = bootstrap_rates(y_true, y_pred)
        assert results is not None
        assert isinstance(results, dict)
        assert len(results) == 13
        for key in results.keys():
            assert key in [
                'tpr',
                'fpr',
                'fnr',
                'tnr',
                'tpr_avg',
                'fpr_avg',
                'fnr_avg',
                'tnr_avg',
                'tpr_error_bars_lower',
                'fpr_error_bars_lower',
                'fnr_error_bars_lower',
                'tnr_error_bars_lower',                
                'tpr_error_bars_upper',
                'fpr_error_bars_upper',
                'fnr_error_bars_upper',
                'tnr_error_bars_upper',
                'number_samples']

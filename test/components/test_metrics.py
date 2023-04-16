import pytest
import torch
import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats
from hypothesis.extra.numpy import arrays
from sklearn.metrics import f1_score, precision_score, recall_score
from your_module import 
from sklearn.metrics import confusion_matrix
from your_module import evaluate_rates, calculate_rates, calculate_metrics, bootstrap_rates, evaluate_rates

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




class TestEvaluateRates:
    @given(
        y_true=lists(binary(), min_size=1, max_size=10),
        y_pred=lists(binary(), min_size=1, max_size=10),
    )
    @settings(deadline=None)
    def test_evaluate_rates(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            return

        rates = evaluate_rates(y_true, y_pred)
        cm = rates["confusion_matrix"]
        tpr, fpr, fnr, tnr = calculate_rates(cm)

        # Assert that the rates match the expected values
        assert np.isclose(rates["tpr"], tpr)
        assert np.isclose(rates["fpr"], fpr)
        assert np.isclose(rates["fnr"], fnr)
        assert np.isclose(rates["tnr"], tnr)

        # Assert that the confusion matrix matches the expected values
        assert np.array_equal(cm, confusion_matrix(y_true, y_pred))

        # Assert that the rates are within a valid range
        for rate_name in ["tpr", "fpr", "fnr", "tnr"]:
            assert 0 <= rates[rate_name] <= 1




class TestCalculateRates:
    @given(
        cm=arrays(
            dtype=np.int32,
            shape=(2, 2),
            elements=integers(min_value=0, max_value=1000),
        )
    )
    @settings(deadline=None)
    def test_calculate_rates(self, cm):
        tpr, fpr, fnr, tnr = calculate_rates(cm)
        tp = np.diag(cm)
        fn = np.sum(cm, axis=1) - tp
        fp = np.sum(cm, axis=0) - tp
        tn = np.sum(cm) - tp - fn - fp

        eps = 0.00000001

        # Calculate expected rates
        expected_tpr = tp / (tp + fn + eps)
        expected_fpr = fp / (fp + tn + eps)
        expected_fnr = fn / (fn + tp + eps)
        expected_tnr = tn / (tn + fp + eps)

        # Assert that the calculated rates match the expected values
        assert np.allclose(tpr, expected_tpr)
        assert np.allclose(fpr, expected_fpr)
        assert np.allclose(fnr, expected_fnr)
        assert np.allclose(tnr, expected_tnr)

        # Assert that the rates are within a valid range
        for rate in [tpr, fpr, fnr, tnr]:
            assert np.all(0 <= rate) and np.all(rate <= 1)



class TestBootstrapRates:
    @given(
        y_true=arrays(
            dtype=np.int32,
            shape=integers(min_value=50, max_value=200),
            elements=integers(min_value=0, max_value=1),
        ),
        y_pred=arrays(
            dtype=np.int32,
            shape=integers(min_value=50, max_value=200),
            elements=integers(min_value=0, max_value=1),
        ),
        n_iterations=integers(min_value=100, max_value=1000),
        percentile=floats(min_value=0.1, max_value=99.9),
    )
    @settings(deadline=None)
    def test_bootstrap_rates(self, y_true, y_pred, n_iterations, percentile):
        if y_true.shape != y_pred.shape:
            pytest.skip("y_true and y_pred shapes don't match")

        results = bootstrap_rates(y_true, y_pred, n_iterations, percentile)
        eval_results = evaluate_rates(y_true, y_pred)

        for rate in ["tpr", "fpr", "fnr", "tnr"]:
            avg_key = f"{rate}_avg"
            lower_key = f"{rate}_error_bars_lower"
            upper_key = f"{rate}_error_bars_upper"

            assert avg_key in results
            assert lower_key in results
            assert upper_key in results

            assert results[lower_key] <= eval_results[rate] <= results[upper_key]

            assert 0 <= results[lower_key] <= results[avg_key] <= results[upper_key] <= 1

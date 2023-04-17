from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def calculate_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    element: str,
    n_iterations: int = 1000,
    percentile: float = 2.5,
) -> Dict[str, Any]:
    """
    Calculates evaluation metrics given the true labels and predicted probabilities.

    Args:
        labels (torch.Tensor): True labels, of shape (n_samples, num_labels).
        predictions (torch.Tensor): Predicted probabilities, of shape (n_samples, num_labels).
        element (str): Name of the element to include in metric names, such as "train" or "test".
        n_iterations (int, optional): Number of bootstrap iterations to use for calculating rates.
            Defaults to 1000.
        percentile (float, optional): Percentile for the bootstrap confidence interval. Defaults to 2.5.

    Returns:
        Dict[str, Any]: A dictionary containing the following evaluation metrics:
            - element_f1 (float): Macro-average F1-score.
            - element_precision (float): Macro-average precision score.
            - element_recall (float): Macro-average recall score.
            - element_tpr (float): True positive rate.
            - element_fpr (float): False positive rate.
            - element_fnr (float): False negative rate.
            - element_tnr (float): True negative rate.

    Example:
        Here's an example of how to use the `calculate_metrics` function:
        >>> import torch
        >>> from sklearn.metrics import f1_score, precision_score, recall_score
        >>> from tmlc.metrics import calculate_metrics

        >>> # Create a tensor of true labels and predicted probabilities
        >>> true_labels = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 1, 1]])
        >>> predicted_probs = torch.tensor(
        ... [[0.8, 0.1, 0.9], [0.3, 0.7, 0.4], [0.7, 0.6, 0.8], [0.2, 0.9, 0.7]])

        >>> # Calculate the evaluation metrics
        >>> metrics = calculate_metrics(
        ... true_labels, predicted_probs, element="test", n_iterations=100, percentile=5.0)

        >>> # Print the evaluation metrics
        >>> print(metrics)
        {'test_f1': 0.717948717948718,
        'test_precision': 0.6666666666666666,
        'test_recall': 0.8,
        'test_tpr': 0.8,
        'test_fpr': 0.16666666666666666,
        'test_fnr': 0.2,
        'test_tnr': 0.8333333333333334}
    """
    # Convert input tensors to numpy arrays
    labels, predictions = labels.detach().numpy(), predictions.detach().numpy()

    # Calculate macro-average F1-score, precision, recall, and accuracy
    metrics = {
        f"{element}_f1": f1_score(labels, predictions, average="macro"),
        f"{element}_precision": precision_score(labels, predictions, average="macro", zero_division=0),
        f"{element}_recall": recall_score(labels, predictions, average="macro"),
    }

    # Calculate bootstrap estimates for true positive rate (tpr), false positive rate (fpr),
    # false negative rate (fnr), and true negative rate (tnr) using the calculate_rates function
    rates = bootstrap_rates(labels, predictions, n_iterations=n_iterations, percentile=percentile)

    # Add bootstrap estimates to the metrics dictionary
    for key, value in rates.items():
        metrics.update({f"{element}_{key}": value})

    # Return the metrics dictionary
    return metrics


def evaluate_rates(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate true positive rate (tpr), false positive rate (fpr), false negative rate (fnr), and
    true negative rate (tnr) given the true labels and predicted labels.

    The function first calculates the confusion matrix using scikit-learn's `confusion_matrix` function.
    It then uses the `calculate_rates` function to calculate the tpr, fpr, fnr, and tnr. The output is
    returned in a dictionary containing the confusion matrix and each of the rates.

    Args:
        y_true (array-like): True labels, of shape (n_samples,).
        y_pred (array-like): Predicted labels, of shape (n_samples,).

    Returns:
        Dict[str, array-like]: A dictionary containing the confusion matrix and each of the rates.

    Example:
        >>> y_true = [1, 0, 1, 1, 0, 1]
        >>> y_pred = [0, 0, 1, 1, 0, 1]
        >>> evaluate_rates(y_true, y_pred)
        {
            "confusion_matrix": array([[2, 1],
                                    [1, 3]]),
            "tpr": 0.75,
            "fpr": 0.3333333333333333,
            "fnr": 0.25,
            "tnr": 0.6666666666666666
        }
    """

    cm = confusion_matrix(y_true, y_pred)
    tpr, fpr, fnr, tnr = calculate_rates(cm)
    return {
        "confusion_matrix": cm,
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "tnr": tnr,
    }


def calculate_rates(cm: np.ndarray) -> Tuple[np.ndarray]:
    """
    Calculate true positive rate (tpr), false positive rate (fpr), false negative rate (fnr), and
    true negative rate (tnr) given a confusion matrix.

    A confusion matrix is a table used to evaluate the performance of a classification algorithm.
    The rows of the matrix represent the true labels, while the columns represent the predicted labels.
    Each cell of the matrix represents the number of samples that fall into a particular category.

    Args:
        cm (array-like): Confusion matrix, of shape (num_labels, num_labels).

    Returns:
        Tuple[array-like]: A tuple containing the tpr, fpr, fnr, and tnr.

    The tpr, fpr, fnr, and tnr can be calculated from the confusion matrix as follows:

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    tnr = tn / (tn + fp)

    where tp, fn, fp, and tn are the numbers of true positives, false negatives, false positives,
    and true negatives, respectively.

    Example:
        >>> import numpy as np
        >>> cm = np.array([[50, 10], [20, 70]])
        >>> tpr, fpr, fnr, tnr = calculate_rates(cm)
        >>> tpr
        array([0.83333333, 0.77777778])
        >>> fpr
        array([0.22222222, 0.16666667])
        >>> fnr
        array([0.16666667, 0.22222222])
        >>> tnr
        array([0.77777778, 0.83333333])
    """

    eps = 0.00000001

    # cm is the confusion matrix
    num_labels = cm.shape[0]

    # Initialize arrays to store true positives, false negatives, false positives,
    # and true negatives for each class
    tp = np.zeros(num_labels)
    fn = np.zeros(num_labels)
    fp = np.zeros(num_labels)
    tn = np.zeros(num_labels)

    # Calculate true positives, false negatives, false positives, and true negatives
    # for each class
    for i in range(num_labels):
        tp[i] = cm[i, i]
        fn[i] = np.sum(cm[i, :]) - tp[i]
        fp[i] = np.sum(cm[:, i]) - tp[i]
        tn[i] = np.sum(cm) - tp[i] - fn[i] - fp[i]

    # Calculate true positive rate, false positive rate, false negative rate, and true
    # negative rate for each class
    tpr = tp / (tp + fn + eps)
    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)
    tnr = tn / (tn + fp + eps)

    # Return arrays containing tpr, fpr, fnr, and tnr for each class
    return tpr, fpr, fnr, tnr


def bootstrap_rates(
    y_true: np.ndarray, y_pred: np.ndarray, n_iterations: int = 1000, percentile: float = 2.5
) -> Dict[str, Dict[str, float]]:
    """
    Estimate average and error bars for true positive rate (tpr), false positive rate (fpr), false
    negative rate (fnr), and true negative rate (tnr) using bootstrapping.

    Bootstrapping is a resampling technique that estimates the sampling distribution of an estimator
    by generating multiple "bootstrap samples" from the original data. In this function, we generate
    multiple bootstrap samples of the true labels and predicted labels, and then calculate the tpr,
    fpr, fnr, and tnr for each sample. Finally, we estimate the average and error bars for each rate
    using the bootstrap samples.

    To generate a bootstrap sample, we randomly sample the original data with replacement, resulting
    in a new dataset of the same size as the original, but with some samples potentially repeated and
    others left out. By repeatedly generating these bootstrap samples, we can estimate the distribution
    of the estimator, and use this distribution to estimate the average and error bars.

    The benefit of bootstrapping is that it allows us to estimate the distribution of an estimator without
    making assumptions about the underlying distribution of the data. This can be particularly useful in
    cases where the sample size is small or the underlying distribution is unknown.

    Args:
        y_true (array-like): True labels, of shape (n_samples,)
        y_pred (array-like): Predicted labels, of shape (n_samples,)
        n_iterations (int): Number of bootstrap iterations.
        percentile (float): Percentile for error bar calculation.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary containing the average values and error bars
        for tpr, fpr, fnr, tnr, n_samples.

    Raises:
        ValueError: If percentile is outside the range of (0, 100).

    Example:
        For single-label classification:

        >>> import numpy as np
        >>> np.random.seed(123)
        >>> y_true = np.random.randint(0, 2, size=100)
        >>> y_pred = np.random.rand(100)
        >>> results = bootstrap_rates(y_true, y_pred)
        >>> set(results.keys())
        {'tpr', 'fpr', 'fnr', 'tnr'}
        >>> results['tpr'].keys()
        dict_keys(['avg', 'error_bars'])
        >>> results['tpr']['avg']
        0.4734470264099684
        >>> results['tpr']['error_bars']
        (0.36764705882352944, 0.5789473684210527)

        For multi-label classification:

        >>> import numpy as np
        >>> np.random.seed(123)
        >>> y_true = np.random.randint(0, 2, size=(100, 3))
        >>> y_pred = np.random.rand(100, 3)
        >>> results = bootstrap_rates(y_true, y_pred)
        >>> set(results.keys())
        {'tpr', 'fpr', 'fnr', 'tnr'}
        >>> results['tpr'].keys()
        dict_keys(['avg', 'error_bars'])
        >>> results['tpr']['avg']
        array([0.47186573, 0.42935668, 0.44356321])
        >>> results['tpr']['error_bars']
        (array([0.3761165 , 0.31934306, 0.33100715]),
         array([0.57416268, 0.5308642 , 0.56390977]))
    """
    # Raise an error if percentile is outside the range of (0, 100).
    if (percentile >= 100) or (percentile <= 0):
        raise ValueError(f"Percentile needs to be between (0, 100); it is {percentile}.")

    # Set the number of samples to the length of y_true.
    n_samples = len(y_true)

    # Initialize a dictionary to store errors for each rate.
    errors = {"tpr": [], "fpr": [], "fnr": [], "tnr": []}

    # Iterate over the number of bootstrap iterations.
    for _ in range(n_iterations):

        # Generate bootstrap samples of true labels and predicted labels.
        idx = np.random.choice(n_samples, n_samples, replace=True)
        y_true_bootstrap = y_true[idx]
        y_pred_bootstrap = y_pred[idx]

        # Evaluate rates for the bootstrap sample.
        eval_bootstrap = evaluate_rates(y_true_bootstrap, y_pred_bootstrap)

        # Append errors to the errors dictionary.
        for key in errors.keys():
            errors[key].extend(eval_bootstrap[key])

    # Initialize a dictionary to store the output.
    output = {}

    # Calculate the average and error bars for each rate.
    for key in errors.keys():
        output[f"{key}_avg"] = np.mean(errors[key])
        output[f"{key}_error_bars_lower"] = np.percentile(errors[key], percentile)
        output[f"{key}_error_bars_upper"] = np.percentile(errors[key], 100 - percentile)
    output["number_samples"] = float(n_samples)
    return output

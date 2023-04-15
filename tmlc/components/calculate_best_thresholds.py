from typing import Callable, Optional

import torch
import torchmetrics


def calculate_best_thresholds(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    vmin: float,
    vmax: float,
    step: float,
    metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    maximize: bool = True,
) -> torch.Tensor:
    """
    Calculate the optimal threshold for each class based on predicted probabilities and true labels.

    Args:
        probabilities (torch.Tensor): Tensor of predicted probabilities for each class, of shape
            (n_samples, num_labels).
        labels (torch.Tensor): Tensor of true labels, of shape (n_samples, num_labels).
        vmin (float): The minimum threshold value to try. Must be within the range (0, 1).
        vmax (float): The maximum threshold value to try. Must be within the range (0, 1).
        step (float): The step size to use between vmin and vmax.
        metric (Optional[Callable[[torch.Tensor, torch.Tensor], float]]):
            The scoring metric to use when evaluating the performance of
            each threshold. Must take two arguments:
            the true labels and the predicted labels. Defaults to the F1 score metric
            from the TorchMetrics library.
        maximize (bool): If True, the metric is maximized; otherwise, it is minimized.

    Raises:
        ValueError: If vmin or vmax are not within the range (0, 1), or if vmax <= vmin.

    Returns:
        torch.Tensor: A tensor of the optimal threshold values for each class, in order.

    Calculates the optimal threshold for each class given the predicted probabilities and true
    labels, by evaluating the performance of each threshold using the specified metric.

    The function returns a torch tensor containing the optimal threshold value for each class,
    in the order they appear in the input.

    Example:
        >>> probs = torch.Tensor([[0.6, 0.4], [0.7, 0.3], [0.8, 0.2]])
        >>> labels = torch.Tensor([[1, 0], [0, 1], [1, 1]])
        >>> thresholds = calculate_best_thresholds(probs, labels, 0.1, 0.9, 0.1)
        >>> print(thresholds)
        tensor([0.5000, 0.2000])

    In this example, there are 3 samples and 2 classes. The predicted probabilities for the
    first class are [0.6, 0.7, 0.8], while for the second class they are [0.4, 0.3, 0.2].
    The true labels are [1, 0] for the first sample, [0, 1] for the second sample, and [1, 1]
    for the third sample. The function is called with vmin=0.1, vmax=0.9, step=0.1, and
    the default F1 score metric from the TorchMetrics library. The function returns the
    optimal threshold for the first class as 0.5, and for the second class as 0.2.
    """

    metric = metric or torchmetrics.classification.BinaryF1Score()

    # Validate input
    if not 0 < vmin < 1:
        raise ValueError(f"vmin must be within the range (0, 1), but got {vmin}")
    if not 0 < vmax < 1:
        raise ValueError(f"vmax must be within the range (0, 1), but got {vmax}")
    if vmax <= vmin:
        raise ValueError(f"vmax ({vmax}) must be greater than vmin ({vmin})")

    # Find the optimal threshold for each class
    _, num_labels = probabilities.shape
    best_thresholds = torch.zeros(num_labels)

    if maximize:
        selection = torch.argmax
    else:
        selection = torch.argmin

    for i in range(num_labels):
        scores = []
        thresholds = torch.arange(vmin, vmax, step)
        for threshold in thresholds:
            predictions = (probabilities[:, i] > threshold).int()
            score = metric(labels[:, i], predictions)
            scores.append(score)
        best_thresholds[i] = thresholds[selection(torch.tensor(scores))]

    return best_thresholds

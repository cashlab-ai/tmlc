import torch


def calculate_predictions(probabilities: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """
    Generate predictions for a multi-class classification task.

    Args:
        probabilities: A tensor of shape (batch_size, num_labels) containing model predictions.
        thresholds: A list of length num_labels containing the optimal thresholds for each class.

    Returns:
        A tensor of shape (batch_size, num_labels) with predictions.

    Example:
        Example A:
        >>> import torch
        >>> probabilities = torch.tensor([[0.3, 0.7, 0.1], [0.1, 0.2, 0.9], [0.7, 0.3, 0.4]])
        >>> thresholds = [0.5, 0.3, 0.8]
        >>> calculate_predictions(probabilities, thresholds)
        tensor([[0., 1., 0.],
                [0., 0., 1.],
                [1., 0., 0.]])

        Example B:
        >>> probabilities = torch.tensor([[0.3, 0.7], [0.1, 0.2], [0.7, 0.3], [0.4, 0.6]])
        >>> thresholds = [0.4, 0.6]
        >>> calculate_predictions(probabilities, thresholds)
        tensor([[1., 1.],
                [0., 0.],
                [1., 0.],
                [1., 0.]])
    """
    # create a copy of the tensor to store predictions
    predictions = probabilities.clone()
    # get the number of classes from the tensor
    _, num_labels = probabilities.shape
    for i in range(num_labels):
        # iterate through the classes and assign 1 or 0 to the predictions depending on the threshold
        predictions[:, i] = (probabilities[:, i] > thresholds[i]).float()
    return predictions

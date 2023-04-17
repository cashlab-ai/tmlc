import torch


def inverse_frequency_weighted(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute inverse frequency weights for each class in a dataset.

    Args:
        labels (torch.Tensor): A tensor of shape (batch_size, num_labels) representing the true labels.

    Returns:
        torch.Tensor: A tensor representing the inverse frequency weights for each class.

    The function first counts the number of samples in each class, then calculates the frequency
    of each class in the dataset. The weight for each class is then calculated as the inverse of
    the class frequency, where the frequency is computed as follows:
    (number of samples in the class + 1) / (total number of samples in the dataset + 1).
    Adding 1 to both the numerator and denominator is a form of smoothing that avoids division
    by zero errors in case a class has zero samples. In this case, we are assuming that the
    class has appeared once in the dataset.

    Example:
        >>> import torch
        >>> from my_module import calculate_inverse_frequency_weights
        >>> labels = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        >>> pos_weight = calculate_inverse_frequency_weights(labels)

        In this example, the input `labels` has shape (4, 3) and contains 4 samples with 3 classes each.
        The first class appears twice, the second class appears three times, and the third class appears
        twice. The frequency of each class is [0.6, 0.8, 0.6], calculated as (2 + 1) / (4 + 1),
        (3 + 1) / (4 + 1), and (2 + 1) / (4 + 1), respectively.
        The weight for each class is [1.6667, 1.25, 1.6667], calculated as 1.0 / 0.6, 1.0 / 0.8,
        and 1.0 / 0.6, respectively.
    """

    # count the number of samples in each class
    class_count = torch.sum(labels, axis=0)

    # calculate the total number of samples
    num_samples, _ = labels.shape

    # calculate the frequency of each class in the dataset
    class_freq = (class_count + 1) / (num_samples + 1)

    # calculate the weight for each class
    class_weights = 1.0 / class_freq

    # convert the class_weights to a tensor
    return {"pos_weight": torch.as_tensor(class_weights, dtype=torch.float32)}

import torch


def bceloss_inverse_frequency_weighted(labels: torch.Tensor) -> torch.Tensor:
    """
    Compute a weighted binary cross-entropy loss based on the frequency of each class.

    Args:
        labels (torch.Tensor): A tensor of shape (batch_size, num_classes) representing the true labels.

    Returns:
        torch.Tensor: A tensor representing the binary cross-entropy loss.

    The function first counts the number of samples in each class, then calculates the frequency
    of each class in the dataset. The weight for each class is then calculated as the inverse of
    the class frequency, where the frequency is computed as follows:
    (number of samples in the class + 1) / (total number of samples in the dataset + 1).
    Adding 1 to both the numerator and denominator is a form of smoothing that avoids division
    by zero errors in case a class has zero samples. In this case, we are assuming that the
    class has appeared once in the dataset.
    Finally, the class weights are converted to a tensor and passed as the `pos_weight` argument
    to the `torch.nn.BCEWithLogitsLoss` function, which computes the binary cross-entropy loss.

    Example:
        >>> import torch
        >>> from my_module import bceloss_inverse_frequency_weighted
        >>> labels = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
        >>> loss_fn = bceloss_inverse_frequency_weighted(labels)
        >>> loss = loss_fn(logits, labels)
    In this example, the input `labels` has shape (4, 3) and contains 4 samples with 3 classes each.
    The first class appears twice, the second class appears three times, and the third class appears
    twice. The frequency of each class is [0.5, 0.75, 0.5], calculated as (2 + 1) / (4 + 1),
    (3 + 1) / (4 + 1), and (2 + 1) / (4 + 1), respectively.
    The weight for each class is [1.3333, 1.25, 1.3333], calculated as 1.0 / 0.6, 1.0 / 0.8,
    and 1.0 / 0.6, respectively.
    These weights are then passed as the `pos_weight` argument to the `torch.nn.BCEWithLogitsLoss`
    function, which computes the binary cross-entropy loss.
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
    class_weights = torch.as_tensor(class_weights, dtype=torch.float32)

    return torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

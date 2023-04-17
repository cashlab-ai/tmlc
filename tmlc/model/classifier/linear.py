import torch

from tmlc.model.classifier.basemodel import GeneralizedClassifier


class LinearClassifier(GeneralizedClassifier):
    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = 0.0):
        """
        A linear classifier with dropout for multi-label text classification.

        Args:
            hidden_size (int): The size of the input features.
            num_labels (int): The number of labels to predict.
            dropout_prob (float): The dropout probability. Default value 0.0
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.linear = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output: torch.Tensor, classifier_additional: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear classifier.

        Args:
            pooled_output (torch.Tensor): The output of the pooling layer with
                shape (batch_size, hidden_size).
            classifier_additional (torch.Tensor): Additional features to concatenate
                with pooled_output, with shape (batch_size, num_additional_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_labels).
        """
        # Concatenate the pooled_output with any additional features before passing to
        # the classifier linear layer.
        if classifier_additional is not None:
            classifier_inputs = torch.cat((pooled_output, classifier_additional), dim=-1)
        else:
            classifier_inputs = pooled_output

        # Apply dropout and pass through the linear layer.
        classifier_inputs = self.dropout(classifier_inputs)
        return self.linear(classifier_inputs)

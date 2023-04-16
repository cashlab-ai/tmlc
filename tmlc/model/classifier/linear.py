import torch
from tmlc.model.classifier.basemodel import GeneralizedClassifier

class LinearClassifier(GeneralizedClassifier):
    def __init__(self, hidden_size: int, num_labels: int):
        """
        A linear classifier for multi-label text classification.
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the linear classifier.

        Args:
            x (Tensor): Input tensor of shape (batch_size, hidden_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_labels).
        """
        return self.linear(x)
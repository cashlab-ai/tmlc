import torch
from tmlc.model.classifier.basemodel import GeneralizedClassifier

class MLPClassifier(GeneralizedClassifier):
    def __init__(self, hidden_size: int, num_labels: int, hidden_layer_size: int, num_hidden_layers: int, dropout_prob: float = 0.1):
        """
        A multi-layer perceptron (MLP) classifier with dropout for multi-label text classification.

        Args:
            hidden_size (int): The size of the input features.
            num_labels (int): The number of labels to predict.
            hidden_layer_size (int): The size of the hidden layers.
            num_hidden_layers (int): The number of hidden layers.
            dropout_prob (float): The dropout probability.
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.layers = torch.nn.ModuleList()
        for i in range(num_hidden_layers):
            if i == 0:
                self.layers.append(torch.nn.Linear(hidden_size, hidden_layer_size))
            else:
                self.layers.append(torch.nn.Linear(hidden_layer_size, hidden_layer_size))
        self.linear = torch.nn.Linear(hidden_layer_size, num_labels)

    def forward(self, pooled_output: torch.Tensor, classifier_additional: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP classifier.

        Args:
            pooled_output (torch.Tensor): The output of the pooling layer with shape (batch_size, hidden_size).
            classifier_additional (torch.Tensor): Additional features to concatenate with pooled_output, with shape 
                                                   (batch_size, num_additional_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_labels).
        """
        # Concatenate the pooled_output with any additional features before passing to the MLP layers.
        if classifier_additional is not None:
            classifier_inputs = torch.cat((pooled_output, classifier_additional), dim=-1)
        else:
            classifier_inputs = pooled_output

        # Pass through the MLP layers
        for layer in self.layers:
            classifier_inputs = layer(classifier_inputs)
            classifier_inputs = torch.relu(classifier_inputs)
            classifier_inputs = self.dropout(classifier_inputs)
        return self.linear(classifier_inputs)

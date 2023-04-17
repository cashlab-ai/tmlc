import torch
from tmlc.model.classifier.basemodel import GeneralizedClassifier

class CNNClassifier(GeneralizedClassifier):
    def __init__(self, hidden_size: int, num_labels: int, num_filters: int, filter_sizes: list, dropout_prob: float = 0.1):
        """
        A convolutional neural network (CNN) classifier with dropout for multi-label text classification.

        Args:
            hidden_size (int): The size of the input features.
            num_labels (int): The number of labels to predict.
            num_filters (int): The number of filters in each convolutional layer.
            filter_sizes (list): The sizes of the filters in each convolutional layer.
            dropout_prob (float): The dropout probability.
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.conv_layers = torch.nn.ModuleList()
        for filter_size in filter_sizes:
            self.conv_layers.append(torch.nn.Conv1d(in_channels=hidden_size, out_channels=num_filters, kernel_size=filter_size))
        self.linear = torch.nn.Linear(len(filter_sizes) * num_filters, num_labels)

    def forward(self, pooled_output: torch.Tensor, classifier_additional: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN classifier.

        Args:
            pooled_output (torch.Tensor): The output of the pooling layer with shape (batch_size, hidden_size).
            classifier_additional (torch.Tensor): Additional features to concatenate with pooled_output, with shape 
                                                   (batch_size, num_additional_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_labels).
        """
        # Reshape the input for the convolutional layers.
        x = pooled_output.permute(0, 2, 1)

        # Apply the convolutional layers and pooling layers.
        conv_outputs = []
        for conv_layer in self.conv_layers:
            conv_output = torch.relu(conv_layer(x))
            conv_output = torch.nn.functional.max_pool1d(conv_output, kernel_size=conv_output.shape[-1])
            conv_outputs.append(conv_output.squeeze(-1))
        conv_outputs = torch.cat(conv_outputs, dim=-1)

        # Concatenate the pooled_output with any additional features before passing to the linear layer.
        if classifier_additional is not None:
            classifier_inputs = torch.cat((conv_outputs, classifier_additional), dim=-1)
        else:
            classifier_inputs = conv_outputs

        # Apply dropout and pass through the linear layer.
        classifier_inputs = self.dropout(classifier_inputs)
        return self.linear(classifier_inputs)

import torch

from tmlc.model.classifier.basemodel import GeneralizedClassifier

class TransformerClassifier(GeneralizedClassifier):
    def __init__(self, hidden_size: int, num_labels: int, num_layers: int, attn_dropout_prob: float = 0.1, ff_dropout_prob: float = 0.1, activation: str = 'gelu'):
        """
        A transformer classifier for multi-label text classification.

        Args:
            hidden_size (int): The size of the input features.
            num_labels (int): The number of labels to predict.
            num_layers (int): The number of transformer layers.
            attn_dropout_prob (float): The dropout probability for the attention mechanism.
            ff_dropout_prob (float): The dropout probability for the feedforward layers.
            activation (str): The activation function to use in the transformer encoder layer. Possible values are 'relu', 'gelu', 'silu', 'swish', 'tanh', 'sigmoid', 'hardshrink', 'hardtanh', 'softsign', 'softplus', 'elu', 'selu', 'leaky_relu', 'rrelu'.
        """
        super().__init__()
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(hidden_size, nhead=8,
                                       dropout=attn_dropout_prob,
                                       activation=activation,
                                       batch_first=True,
                                       dim_feedforward=hidden_size*4),
            num_layers=num_layers
        )
        self.linear = torch.nn.Linear(hidden_size, num_labels)
        self.dropout = torch.nn.Dropout(ff_dropout_prob)

    def forward(self, pooled_output: torch.Tensor, classifier_additional: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer classifier.

        Args:
            pooled_output (torch.Tensor): The output of the pooling layer with shape (batch_size, hidden_size).
            classifier_additional (torch.Tensor): Additional features to concatenate with pooled_output, with shape 
                                                   (batch_size, num_additional_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_labels).
        """
        # Concatenate the pooled_output with any additional features before passing to the Transformer.
        if classifier_additional is not None:
            classifier_inputs = torch.cat((pooled_output, classifier_additional), dim=-1)
        else:
            classifier_inputs = pooled_output

        # Reshape the input for the Transformer.
        classifier_inputs = classifier_inputs.unsqueeze(1)

        # Pass through the Transformer.
        transformer_output = self.transformer(classifier_inputs)

        # Collapse the sequence dimension and pass through the linear layer.
        transformer_output = transformer_output.squeeze(1)
        transformer_output = self.dropout(transformer_output)
        return self.linear(transformer_output)

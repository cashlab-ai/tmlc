import torch
from tmlc.model.classifier.basemodel import GeneralizedClassifier

class RNNClassifier(GeneralizedClassifier):
    def __init__(self, hidden_size: int, num_labels: int, num_layers: int, bidirectional: bool = False, rnn_type: str = "gru"):
        """
        A recurrent neural network (RNN) classifier for multi-label text classification.

        Args:
            hidden_size (int): The size of the hidden states.
            num_labels (int): The number of labels to predict.
            num_layers (int): The number of RNN layers.
            bidirectional (bool): Whether to use bidirectional RNNs.
            rnn_type (str): The type of RNN to use (gru or lstm).
        """
        super().__init__()
        if rnn_type == "gru":
            self.rnn = torch.nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        elif rnn_type == "lstm":
            self.rnn = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            raise ValueError("Invalid RNN type: {}".format(rnn_type))
        self.linear = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), num_labels)

    def forward(self, pooled_output: torch.Tensor, classifier_additional: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RNN classifier.

        Args:
            pooled_output (torch.Tensor): The output of the pooling layer with shape (batch_size, hidden_size).
            classifier_additional (torch.Tensor): Additional features to concatenate with pooled_output, with shape 
                                                   (batch_size, num_additional_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_labels).
        """
        # Concatenate the pooled_output with any additional features before passing to the RNN.
        if classifier_additional is not None:
            classifier_inputs = torch.cat((pooled_output, classifier_additional), dim=-1)
        else:
            classifier_inputs = pooled_output

        # Reshape the input for the RNN and pass through the RNN.
        classifier_inputs = classifier_inputs.unsqueeze(1)
        rnn_output, _ = self.rnn(classifier_inputs)
        rnn_output = rnn_output[:, -1, :]

        # Pass through the linear layer.
        return self.linear(rnn_output)

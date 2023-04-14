from typing import Optional

from pydantic import BaseModel
from transformers import AutoModel, PreTrainedModel


class PreTrainedConfig(BaseModel):
    """
    A configuration class for a pre-trained model.

    Attributes:
        path (str): The path to the pre-trained model files.
        kwargs (Optional[dict]): A dictionary of additional keyword arguments to pass
            to the tokenizer.
        instance (Optional[PreTrainedModel]): Instance of the pre-trained model.

    Example:
        >>> # create a configuration object
        >>> config = PreTrainedConfig(path='bert-base-uncased', kwargs={'return_dict': True})
        >>> # apply the pre-trained model on some input data
        >>> input_data = 'Hello world!'
        >>> output = config(input_data)
        >>> # print the output
        >>> print(output)
        {'last_hidden_state': tensor([[[...]]]),
         'pooler_output': tensor([[...]]),
         'hidden_states': (tensor([[[...]]]), ...)}
    """

    path: str
    kwargs: Optional[dict] = None
    instance: Optional[str] = None

    @property
    def model(self):
        """
        Initialize the Pre-Trained Model.
        """
        if ~isinstance(self.instance, PreTrainedModel):
            self.instance = AutoModel.from_pretrained(self.path)
        return self.instance

    def __call__(self, data):
        """
        Apply the Pre-Trained Model on the input data.
        """
        kwargs = self.kwargs or {}
        return self.model(data, **kwargs)

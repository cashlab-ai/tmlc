from typing import List, Optional

from pydantic import BaseModel
from transformers import AutoTokenizer, PreTrainedTokenizer


class TokenizerConfig(BaseModel):
    """
    A configuration class for a tokenizer.

    Attributes:
        model_name (str): The name of the pre-trained model to use.
        path (str): The path to the tokenizer files.
        max_length (int): The maximum length of input sequences.
        kwargs (Optional[Dict[str, Any]]): A dictionary of additional keyword arguments to pass
            to the tokenizer.
        instance (Optional[Any]): Instance of the pre-trained tokenizer.
        output_keys (List[str]): A list of output keys to include in the returned dictionary.

    Example:
        >>> # Initialize the tokenizer configuration object
        >>> tokenizer_config = TokenizerConfig(
        ...     model_name='bert-base-uncased',
        ...     path='bert-base-uncased',
        ...     max_length=512,
        ...     output_keys=['input_ids', 'attention_mask']
        ... )
        >>> # Tokenize a list of sentences
        >>> sentences = ['Hello world!', 'This is a test.']
        >>> tokenized_data = tokenizer_config(sentences)
        >>> # Print the tokenized data
        >>> print(tokenized_data)
        Output:
        {
            'input_ids': [[101, 7592, 2088, 999, 102], [101, 2023, 2003, 1037, 3231, 1012, 102]],
            'attention_mask': [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]
        }
    """

    model_name: str
    path: str
    max_length: int
    output_keys: List[str]
    kwargs: Optional[dict] = None
    instance: Optional[str] = None

    @property
    def tokenizer(self):
        """
        Initialize the tokenizer.
        """
        if ~isinstance(self.instance, PreTrainedTokenizer):
            self.instance = AutoTokenizer.from_pretrained(self.path, model_name=self.model_name)
        return self.instance

    def __call__(self, data):
        """
        Tokenize the input data using the tokenizer.
        """
        kwargs = self.kwargs or {}
        output = self.tokenizer(data, **kwargs)
        return {key: output[key] for key in self.output_keys}

from typing import Optional

from pydantic import BaseModel

from tmlc.configclasses import TokenizerConfig


class DatasetConfig(BaseModel):
    """
    A configuration class for a dataset.

    This class is used to store configuration information for a dataset. It includes a `tokenizer_config`
    attribute that specifies the configuration for the tokenizer to use.

    Attributes:
        tokenizer_config (TokenizerConfig): The configuration for the tokenizer to use.
        batch_size (int): The batch size to use when processing the dataset.
        kwargs (Optional[dict]): Additional keyword arguments to pass to the dataset class constructor.
    """

    tokenizer_config: TokenizerConfig
    batch_size: int
    kwargs: Optional[dict] = None

    @property
    def tokenizer(self):
        """
        Return the tokenizer from the tokenizer configuration.

        Returns:
            PreTrainedTokenizer: The tokenizer instance.
        """
        return self.tokenizer_config

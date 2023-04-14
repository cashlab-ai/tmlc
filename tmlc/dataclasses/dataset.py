from typing import List

import torch
from torch.utils.data import Dataset
from tmlc.dataclasses import Message
from tmlc.configclasses import DatasetConfig


class Dataset(Dataset):
    """
    A custom dataset class that can be used with PyTorch DataLoader to load email data.

    Attributes:
        messages (List[Message]): A list of Message objects containing the email data.
        config (DatasetConfig): A DatasetConfig object containing configuration information.

    Examples:
        >>> from tmlc.dataclasses import Message
        >>> from tmlc.configclasses import DatasetConfig
        >>> from tmlc.datasets import Dataset
        >>> import transformers
        >>> messages = [Message(text='This is a test email', labels=['spam']),
        ...             Message(text='Another email for testing', labels=['not_spam'])]
        >>> tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        >>> dataset_config = DatasetConfig(tokenizer=tokenizer, batch_size=2, kwargs={'num_workers': 4})
        >>> dataset = Dataset(messages, dataset_config)
        >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset_config.batch_size,
        ...                                          num_workers=dataset_config.kwargs.get('num_workers', 0))
        >>> for batch in dataloader:
        ...     print(batch['input_ids'], batch['attention_mask'], batch['labels'])
        tensor([[  101,  2023,  2003,  1037,  3231,  2758,   102],
                [ 2066,  2758,  2005,  5607,   102,     0,     0]])
        tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]])
        tensor([[1., 0.], [0., 1.]])
    """

    def __init__(self, messages: List[Message], config: DatasetConfig):
        """
        Initializes the dataset.

        Args:
            messages (List[Message]): A list of Message objects containing the email data.
            config (DatasetConfig): A DatasetConfig object containing configuration information.
        """
        self.messages = messages
        self.config = config

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.messages)

    def __getitem__(self, index: int) -> dict:
        """
        Gets an item from the dataset at the specified index.

        Args:
            index (int): The index of the item to get.

        Returns:
            dict: A dictionary containing the input_ids, attention_mask, and labels for the item.
        """

        message = self.messages[index]
        encoding = self.config.tokenizer(message.text)
        output = {key: torch.tensor(value) for key, value in encoding.items()}

        # Convert bool to int.
        labels = list(map(int, message.labels))
        output.update({"labels": torch.FloatTensor(labels)})
        return output

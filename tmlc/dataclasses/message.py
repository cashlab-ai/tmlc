from dataclasses import dataclass
from typing import List


@dataclass
class Message:
    """
    A dataclass that holds a message text and its labels.

    Attributes:
        text (str): The message text.
        labels (List[int]): The list of labels for the message.
    """

    text: str
    labels: List[bool]


@dataclass
class Messages:
    """
    A dataclass that holds a list of messages and their labels.

    Attributes:
        messages (List[Message]):
            The list of messages.
        labels_names (List[str]):
            The list of label names for the messages.
    """

    messages: List[Message]
    labels_names: List[str]

    def __len__(self):
        """
        Returns the number of messages in the dataset.

        Returns:
            int:
                The number of messages.
        """
        return len(self.messages)

    def __getitem__(self, index: int) -> dict:
        """
        Returns the message at the given index.

        Args:
            index (int):
                The index of the message to return.

        Returns:
            Message:
                The message at the given index.
        """
        return self.messages[index]

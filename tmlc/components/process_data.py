from typing import List

import pandas as pd

from tmlc.dataclasses import Message, Messages


def process_data(data: pd.DataFrame, text_column: str, labels_columns: List[str]) -> Messages:
    """
    Processes the input data by converting it to a list of messages.

    Args:
        data (pd.DataFrame): The data to preprocess.
        text_column (str): The name of the column containing the message text.
        labels_columns (List[str]): The names of the columns containing the labels.

    Returns:
        Messages: A Messages object containing the preprocessed messages.
    """
    messages = data.apply(
        lambda row: Message(row[text_column], row[labels_columns].tolist()), axis=1
    ).tolist()
    return Messages(messages, labels_names=labels_columns)

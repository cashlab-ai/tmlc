import pandas as pd


def get_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a file and preprocess it.

    Args:
        file_path (str): The path to the file containing the data.

    Returns:
        pd.DataFrame: Containing the raw data.
    """

    return pd.read_csv(file_path)

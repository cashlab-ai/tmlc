from typing import List, Tuple

import numpy as np
import pandas as pd
from skmultilearn.model_selection import iterative_train_test_split


def combine_frames(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    labels_columns: List[str],
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Combines the train, validation, and test data into a single dataframe for each dataset.

    Args:
        X_train (np.ndarray): Array of training data.
        X_val (np.ndarray): Array of validation data.
        X_test (np.ndarray): Array of testing data.
        y_train (np.ndarray): Array of training labels.
        y_val (np.ndarray): Array of validation labels.
        y_test (np.ndarray): Array of testing labels.
        labels_columns (List[str]): List of columns containing the labels.
        feature_columns (List[str]): List of columns containing the features.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the combined train, validation,
            and test DataFrames.

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
        >>> X_val = np.array([[7, 8]])
        >>> X_test = np.array([[9, 10], [11, 12]])
        >>> y_train = np.array([[0], [1], [0]])
        >>> y_val = np.array([[1]])
        >>> y_test = np.array([[0], [1]])
        >>> labels_columns = ["label"]
        >>> feature_columns = ["feature_1", "feature_2"]
        >>> train_data, val_data, test_data = combine_frames(
        ... X_train, X_val, X_test, y_train, y_val, y_test, labels_columns, feature_columns)
        >>> print("Train data shape:", train_data.shape)
        >>> print("Train data:")
        >>> print(train_data)
        >>> print("Validation data shape:", val_data.shape)
        >>> print("Validation data:")
        >>> print(val_data)
        >>> print("Test data shape:", test_data.shape)
        >>> print("Test data:")
        >>> print(test_data)
        Train data shape: (3, 3)
        Train data:
        feature_1  feature_2  label
        0          1          2      0
        1          3          4      1
        2          5          6      0
        Validation data shape: (1, 3)
        Validation data:
        feature_1  feature_2  label
        0          7          8      1
        Test data shape: (2, 3)
        Test data:
        feature_1  feature_2  label
        0          9         10      0
        1         11         12      1
    """
    if not all(X.shape[0] == y.shape[0] for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]):
        raise ValueError("The number of rows in X and y must be the same for all sets.")

    # Convert arrays back to dataframes
    train_df = pd.DataFrame(X_train, columns=feature_columns)
    train_df[labels_columns] = y_train
    val_df = pd.DataFrame(X_val, columns=feature_columns)
    val_df[labels_columns] = y_val
    test_df = pd.DataFrame(X_test, columns=feature_columns)
    test_df[labels_columns] = y_test
    return train_df, val_df, test_df


def split_data(
    data: pd.DataFrame,
    labels_columns: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a given data set into training, validation, and test sets based on user-specified ratios.

    Args:
        data (pd.DataFrame): DataFrame containing features and labels.
        labels_columns (List[str]): List of columns containing the labels.
        train_ratio (float, optional): Proportion of data to use for training.
        val_ratio (float, optional): Proportion of data to use for validation.
        test_ratio (float, optional): Proportion of data to use for testing.
        random_state (int, optional): Random state to use for reproducibility. Defaults to 42.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the train, validation,
            and test DataFrames.

    Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ...     "feature_2": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ...     "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        ... })
        >>> labels_columns = ["label"]
        >>> train_ratio = 0.6
        >>> val_ratio = 0.2
        >>> test_ratio = 0.2
        >>> train_data, val_data, test_data = split_data(
        ... data, labels_columns, train_ratio, val_ratio, test_ratio)
        >>> print("Train data shape:", train_data.shape)
        >>> print("Train data:")
        >>> print(train_data)
        >>> print("Validation data shape:", val_data.shape)
        >>> print("Validation data:")
        >>> print(val_data)
        >>> print("Test data shape:", test_data.shape)
        >>> print("Test data:")
        >>> print(test_data)
        Train data shape: (8, 3)
        Train data:
            feature_1  feature_2  label
        0          1          6      0
        2          3          8      0
        3          4          9      1
        5          6         11      1
        7          8         13      1
        9         10         15      1
        10        11         16      0
        12        13         18      0
        Validation data shape: (2, 3)
        Validation data:
            feature_1  feature_2  label
        1          2          7      1
        11        12         17      0
        Test data shape: (4, 3)
        Test data:
            feature_1  feature_2  label
        4          5         10      0
        6          7         12      0
        8          9         14      0
        13        14         19      1
    """
    np.random.seed(random_state)

    # Check that ratios add up to 1.0
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must add up to 1.0")

    # Check for missing values
    if data.isnull().values.any():
        raise ValueError("data contains missing values")

    # Check that labels_columns is not empty
    if not labels_columns:
        raise ValueError("labels_columns cannot be empty")

    # Split data into X and y
    X = data.drop(columns=labels_columns)
    y = data[labels_columns]

    # Split into train and test
    X_train, y_train, X_test, y_test = iterative_train_test_split(X.values, y.values, test_size=test_ratio)

    # Split train into train and val
    X_train, y_train, X_val, y_val = iterative_train_test_split(
        X_train, y_train, test_size=val_ratio / (train_ratio + val_ratio)
    )

    return combine_frames(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        labels_columns=labels_columns,
        feature_columns=X.columns,
    )

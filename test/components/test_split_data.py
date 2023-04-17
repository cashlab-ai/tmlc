import pytest
import numpy as np
import pandas as pd
from tmlc.components.split_data import combine_frames, split_data

class TestCombineFrames:
    @pytest.fixture
    def X_train(self):
        return np.array([[1, 2], [3, 4], [5, 6]])

    @pytest.fixture
    def X_val(self):
        return np.array([[7, 8]])

    @pytest.fixture
    def X_test(self):
        return np.array([[9, 10], [11, 12]])

    @pytest.fixture
    def y_train(self):
        return np.array([[0], [1], [0]])

    @pytest.fixture
    def y_val(self):
        return np.array([[1]])

    @pytest.fixture
    def y_test(self):
        return np.array([[0], [1]])

    @pytest.fixture
    def labels_columns(self):
        return ["label"]

    @pytest.fixture
    def feature_columns(self):
        return ["feature_1", "feature_2"]

    def test_combine_frames(self, X_train, X_val, X_test, y_train, y_val, y_test, labels_columns, feature_columns):
        train_df, val_df, test_df = combine_frames(X_train, X_val, X_test, y_train, y_val, y_test, labels_columns, feature_columns)
        assert train_df.shape == (3, 3)
        assert val_df.shape == (1, 3)
        assert test_df.shape == (2, 3)
        assert list(train_df.columns) == feature_columns + labels_columns
        assert list(val_df.columns) == feature_columns + labels_columns
        assert list(test_df.columns) == feature_columns + labels_columns


class TestSplitData:
    @pytest.fixture
    def data(self):
        return pd.DataFrame({
            "feature_1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "feature_2": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })

    @pytest.fixture
    def labels_columns(self):
        return ["label"]

    def test_split_data(self, data, labels_columns):
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2
        train_df, val_df, test_df = split_data(data, labels_columns, train_ratio, val_ratio, test_ratio)
        
        assert train_df.shape == (8, 3)
        assert val_df.shape == (3, 3)
        assert test_df.shape == (3, 3)
        assert list(train_df.columns) == data.columns.tolist()
        assert list(val_df.columns) == data.columns.tolist()
        assert list(test_df.columns) == data.columns.tolist()

    def test_split_data_with_invalid_ratios(self, data, labels_columns):
        with pytest.raises(ValueError):
            train_ratio = 0.4
            val_ratio = 0.4
            test_ratio = 0.4
            split_data(data, labels_columns, train_ratio, val_ratio, test_ratio)

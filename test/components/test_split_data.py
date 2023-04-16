import pytest
import pandas as pd
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import data as st_data, integers, lists, floats
from your_module import split_data, combine_frames

class TestSplitData:
    @given(
        st_data(),
        n_rows=integers(min_value=10, max_value=50),
        n_features=integers(min_value=1, max_value=10),
        n_labels=integers(min_value=1, max_value=5),
        train_ratio=floats(min_value=0.5, max_value=0.9),
        val_ratio=floats(min_value=0.1, max_value=0.4),
        test_ratio=floats(min_value=0.1, max_value=0.4),
    )
    @settings(deadline=None)
    def test_split_data(
        self, data, n_rows, n_features, n_labels, train_ratio, val_ratio, test_ratio
    ):
        if train_ratio + val_ratio + test_ratio > 1.0:
            pytest.skip("Invalid ratios")

        feature_columns = [f"feature_{i}" for i in range(1, n_features + 1)]
        label_columns = [f"label_{i}" for i in range(1, n_labels + 1)]
        df = pd.DataFrame(
            {
                **{col: data.draw(lists(integers(min_value=0, max_value=100), min_size=n_rows, max_size=n_rows)) for col in feature_columns},
                **{col: data.draw(lists(integers(min_value=0, max_value=1), min_size=n_rows, max_size=n_rows)) for col in label_columns},
            }
        )

        train, val, test = split_data(df, label_columns, train_ratio, val_ratio, test_ratio)

        assert train.shape[1] == val.shape[1] == test.shape[1] == len(feature_columns) + len(label_columns)

        total_rows = train.shape[0] + val.shape[0] + test.shape[0]
        assert total_rows == n_rows

        for col in feature_columns + label_columns:
            assert col in train.columns
            assert col in val.columns
            assert col in test.columns


class TestCombineFrames:
    @given(
        st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=10),
        st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=10),
        st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=10),
    )
    def test_combine_frames(self, feature_columns, label_columns, row_sizes):
        if sum(row_sizes) != len(row_sizes) * len(feature_columns):
            pytest.skip("Invalid row sizes")

        X_train = np.array(row_sizes[: len(row_sizes) // 3]).reshape((-1, len(feature_columns)))
        X_val = np.array(row_sizes[len(row_sizes) // 3 : 2 * len(row_sizes) // 3]).reshape((-1, len(feature_columns)))
        X_test = np.array(row_sizes[2 * len(row_sizes) // 3 :]).reshape((-1, len(feature_columns)))
        y_train = np.array([0] * (len(row_sizes) // 3)).reshape((-1, len(label_columns)))
        y_val = np.array([0] * (len(row_sizes) // 3)).reshape((-1, len(label_columns)))
        y_test = np.array([0] * (len(row_sizes) // 3)).reshape((-1, len(label_columns)))

        train, val, test = combine_frames(
            X_train, X_val, X_test, y_train, y_val, y_test, label_columns, feature_columns
        )

        assert train.shape == (X_train.shape[0], X_train.shape[1] + y_train.shape[1])
        assert val.shape == (X_val.shape[0], X_val.shape[1] + y_val.shape[1])
        assert test.shape == (X_test.shape[0], X_test.shape[1] + y_test.shape[1])

        assert set(train.columns) == set(feature_columns + label_columns)
        assert set(val.columns) == set(feature_columns + label_columns)
        assert set(test.columns) == set(feature_columns + label_columns)

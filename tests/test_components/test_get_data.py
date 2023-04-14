import unittest
from unittest.mock import patch

import pandas as pd

from tmlc.components.get_data import get_data


class TestGetData(unittest.TestCase):
    @patch("pandas.read_csv")
    def test_get_data(self, mock_read_csv):
        # Mock the pd.read_csv() function to return a DataFrame
        data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        mock_read_csv.return_value = data

        # Call the function and check that the output is correct
        result = get_data("path/to/file.csv")
        mock_read_csv.assert_called_once_with("path/to/file.csv")
        self.assertIsInstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, data)

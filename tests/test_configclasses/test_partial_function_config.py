from typing import Any, Dict

import numpy as np
import pytest

from tmlc.configclasses.partial_function_config import (
    PartialFunctionConfig,
    get_partial,
)


class TestGetPartial:
    def test_get_partial_with_no_args(self):
        module_name = "math"
        func_name = "pow"
        func = get_partial(module_name, func_name)
        assert func(2, 3) == 8

    def test_get_partial_with_args(self):
        module_name = "math"
        func_name = "pow"
        func_args = (2,)
        func = get_partial(module_name, func_name, args=func_args)
        assert func(3) == 8

    def test_get_partial_with_kwargs(self):
        module_name = "numpy"
        func_name = "arange"
        func_kwargs = {"start": 0, "stop": 10, "step": 1, "dtype": int}
        func = get_partial(module_name, func_name, kwargs=func_kwargs)
        assert np.array_equal(func(), np.arange(**func_kwargs))


class TestPartialFunctionConfig:
    @pytest.fixture
    def config(self):
        module = "numpy"
        func = "arange"
        kwargs = {"start": 0, "stop": 10, "step": 1, "dtype": int}
        return dict(module=module, func=func, kwargs=kwargs)

    def test_creation(self, config: Dict[str, Any]):
        partial_config = PartialFunctionConfig(**config)
        assert isinstance(partial_config, PartialFunctionConfig)

    def test_func_keywords(self, config: Dict[str, Any]):
        partial_config = PartialFunctionConfig(**config)
        func = partial_config.partial
        assert func.keywords == config["kwargs"]

    def test_partial(self, config: Dict[str, Any]):
        partial_config = PartialFunctionConfig(**config)

        # assert that the original function behaves as expected
        assert np.array_equal(partial_config.partial(), np.arange(**config["kwargs"]))

import pytest
from functools import partial
from tmlc.configclasses.partial_function_config import PartialFunctionConfig, get_partial

class TestPartialFunctionConfig:
    @pytest.fixture
    def module_name(self):
        return "torch"

    @pytest.fixture
    def func_name(self):
        return "sigmoid"

    @pytest.fixture
    def args(self):
        return [2, 3]

    @pytest.fixture
    def kwargs(self):
        return {}

    @pytest.fixture
    def partial_function_config_data(self, module_name, func_name, args, kwargs):
        return {
            "module": module_name,
            "func": func_name,
            "args": args,
            "kwargs": kwargs,
        }

    def test_get_partial(self, module_name, func_name, args, kwargs):
        partial_func = get_partial(module_name, func_name, args, kwargs)
        assert isinstance(partial_func, partial)
        assert partial_func.func.__name__ == func_name
        assert partial_func.args == tuple(args)
        assert partial_func.keywords == kwargs

    def test_get_partial_with_invalid_module(self, func_name, args, kwargs):
        with pytest.raises(ValueError):
            get_partial("invalid_module", func_name, args, kwargs)

    def test_get_partial_with_invalid_function(self, module_name, args, kwargs):
        with pytest.raises(ValueError):
            get_partial(module_name, "invalid_function", args, kwargs)

    def test_partial_function_config(self, partial_function_config_data):
        partial_function_config = PartialFunctionConfig(**partial_function_config_data)
        assert partial_function_config.module == partial_function_config_data["module"]
        assert partial_function_config.func == partial_function_config_data["func"]
        assert partial_function_config.args == partial_function_config_data["args"]
        assert partial_function_config.kwargs == partial_function_config_data["kwargs"]
        assert isinstance(partial_function_config.partial, partial)

    def test_missing_fields(self):
        with pytest.raises(ValueError):
            PartialFunctionConfig()

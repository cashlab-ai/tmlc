import pytest
from hypothesis import given, strategies as st
from your_module import get_partial, PartialFunctionConfig

class TestPartialFunction:
    def test_get_partial(self):
        partial_sum = get_partial("builtins", "sum", kwargs={"start": 10})
        result = partial_sum([1, 2, 3, 4])
        assert result == 20

        with pytest.raises(ValueError):
            get_partial("unknown_module", "unknown_function")

    @given(
        module=st.sampled_from(["builtins"]),
        func=st.sampled_from(["sum"]),
        args=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=5),
        kwargs=st.fixed_dictionaries({"start": st.integers(min_value=1, max_value=10)}),
    )
    def test_partial_function_config(self, module, func, args, kwargs):
        config = PartialFunctionConfig(module=module, func=func, args=args, kwargs=kwargs)

        assert config.module == module
        assert config.func == func
        assert config.args == args
        assert config.kwargs == kwargs

        partial_function = config.partial
        result = partial_function(args)
        assert result == sum(args, **kwargs)

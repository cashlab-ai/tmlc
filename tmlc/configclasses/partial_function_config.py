import importlib
from functools import partial
from typing import Any, Dict, Optional

from pydantic import BaseModel


def get_partial(
    module: str, func: str, args: Optional[Any] = None, kwargs: Optional[Dict[str, Any]] = None
) -> partial:
    """
    Create and return a partial function object.

    Args:
        module (str): Name of the module that contains the function to create the partial object.
        func (str): Name of the function to create the partial object.
        args (Optional[Any], optional): Arguments to pass to the function. Defaults to None.
        kwargs (Optional[Dict[str, Any]], optional): Keyword arguments to pass to the function.
            Defaults to None.

    Raises:
        ValueError: If function object is not found.

    Returns:
        partial: A partial function object.
    """
    try:
        # Get the function object from the module and create a partial object
        func = getattr(importlib.import_module(module), func)
    except (AttributeError, ModuleNotFoundError):
        raise ValueError(f"Could not find function {func} in module {module}")

    args = args or []
    kwargs = kwargs or {}
    return partial(func, *args, **kwargs)


class PartialFunctionConfig(BaseModel):
    """
    Configuration class for creating a partial function object.

    Attributes:
        module (str): Name of the module that contains the function.
        func (str): Name of the function to create the partial object.
        args (Optional[list], optional): Arguments to pass to the function. Defaults to None.
        kwargs (Optional[dict], optional): Keyword arguments to pass to the function.
            Defaults to None.

    Properties:
        partial (partial): A partial function object.
    """

    module: str
    func: str
    args: Optional[list] = None
    kwargs: Optional[dict] = None

    @property
    def partial(self):
        """
        Creates and returns a partial function object.

        Returns:
            partial: A partial function object.
        """
        return get_partial(self.module, self.func, self.args, self.kwargs)
    
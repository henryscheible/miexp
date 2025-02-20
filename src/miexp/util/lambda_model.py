from collections.abc import Callable, Collection
from typing import Any

from torch import nn


class LambdaModel(nn.Module):
    """A PyTorch model that wraps a given function and allows it to be used as a module.

    Methods:
        __init__(func: Callable):
            Initializes the LambdaModel with a given function.
        __forward___(*args: Sequence[Any], **kwargs: Mapping[Any, Any]) -> Any:
            Forward method that calls the stored function with the provided arguments.
    """

    def __init__(self, func: Callable, submodules: Collection[nn.Module] = set()):
        """Initializes the LambdaModel with a given function.

        Args:
            func (Callable): The function to be stored in the LambdaModel.
            submodules (Collection[nn.Module]): Other modules which should be moved to a device with this module.
        """
        super().__init__()
        self.func = func
        self.submodules = submodules

    def forward(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Forward method that calls the stored function with the provided arguments.

        Args:
            *args (Sequence[Any]): Positional arguments to pass to the function.
            **kwargs (Mapping[Any, Any]): Keyword arguments to pass to the function.

        Returns:
            Any: The result of the function call.
        """
        return self.func(*args, **kwargs)

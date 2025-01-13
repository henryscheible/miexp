from typing import Any

from torch import nn
from torchviz import make_dot


def visualize_model(
    model: nn.Module,
    example_input: Any,  # noqa: ANN401
    file_name: str,
    format: str = "pdf",
) -> None:
    """Create a flowchart depicting the model's computational graph and save it to the file_name and format provided."""
    model.train()
    yhat = model(example_input)
    make_dot(yhat, params=dict(list(model.named_parameters()))).render(
        file_name, format=format
    )

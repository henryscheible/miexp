import plotly.graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from torch import Tensor


def visualize_state_dict_at_epoch(
    state_dict: dict[str, Tensor],
) -> Figure:
    """Visualizes the state dictionary at a given epoch by creating a subplot of heatmaps.

    Parameters:
        state_dict (dict[str, Tensor]): A dictionary where keys are matrix names and values are tensors representing the matrices.
        epoch (int): The epoch number for which the visualization is being created.

    Returns:
        Figure: A Plotly Figure object containing the heatmaps of the matrices in the state dictionary, as well as the effective QKV and QKVO matrices.
    """
    num_epochs = next(iter(state_dict.values())).shape[0]
    num_matrices = len(state_dict)
    fig = make_subplots(
        rows=num_matrices // 3 + 1,
        cols=3,
        subplot_titles=list(state_dict.keys()),
    )

    for epoch in range(num_epochs):
        matrices = []
        for i, (_, matrix) in enumerate(state_dict.items()):
            fig.add_trace(
                go.Heatmap(z=matrix[epoch, :, :].cpu().numpy(), coloraxis="coloraxis"),
                row=(i // 3) + 1,
                col=(i % 3) + 1,
            )
            matrices.append(matrix[epoch, :, :].cpu().numpy())

    # Make first epoch trace visible
    for i in range(len(state_dict)):
        fig.data[i].visible = True  # type: ignore

    # Create and add slider
    steps = []
    for epoch in range(num_epochs):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},  # type: ignore
                {"title": "Slider switched to step: " + str(epoch)},
            ],  # layout attribute
        )
        for i in range(epoch * len(state_dict), (epoch + 1) * len(state_dict)):
            step["args"][0]["visible"][i] = True  # type: ignore
        steps.append(step)

    sliders = [
        dict(
            active=0,
            currentvalue={"prefix": "Frequency: "},
            pad={"t": 50},
            steps=steps,
        )
    ]

    fig.update_layout(sliders=sliders)

    fig.update_layout(
        height=300 * num_matrices / 3,  # Adjust height based on the number of matrices
        coloraxis={"colorscale": "Viridis"},
        showlegend=False,
    )

    i = len(state_dict)
    fig.update_traces(texttemplate="%{z:.2f}")

    return fig

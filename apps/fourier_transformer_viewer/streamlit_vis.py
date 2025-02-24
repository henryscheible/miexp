import json
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots
from pydantic import ConfigDict, TypeAdapter
from torch import Tensor

from miexp.models.interptransformer import SingleHeadTransformerOneHotPositionalNoMLP

torch.classes.__path__ = []


st.title("Fourier Component Training Results")


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
        rows=num_matrices // 3 + 1, cols=3, subplot_titles=list(state_dict.keys())
    )
    min_val = min([torch.min(mat).item() for mat in state_dict.values()])
    max_val = max([torch.max(mat).item() for mat in state_dict.values()])

    for epoch in range(0, num_epochs, 5):
        for i, (name, matrix) in enumerate(state_dict.items()):
            trace = go.Heatmap(
                z=matrix[epoch, :, :].cpu().numpy(),
                coloraxis="coloraxis",
            )
            trace.name = f"{epoch}/{name}"
            trace.showlegend = True
            fig.add_trace(trace, row=(i // 3) + 1, col=(i % 3) + 1)

    # Make first epoch trace visible

    for i in range(len(fig.data)):  # type: ignore
        fig.data[i].visible = False  # type: ignore

    for i in range(len(state_dict)):
        fig.data[i].visible = True  # type: ignore

    # Create and add slider
    steps = []
    for epoch in range(0, num_epochs, 5):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(fig.data)},  # type: ignore
                {"title": "Slider switched to step: " + str(epoch)},
            ],  # layout attribute
            label=epoch,
        )
        for i in range(
            (epoch // 5) * len(state_dict), ((epoch // 5) + 1) * len(state_dict)
        ):
            step["args"][0]["visible"][i] = True  # type: ignore
        steps.append(step)

    sliders = [
        dict(
            active=0,
            # currentvalue={"prefix": "Frequency: "},
            # pad={"t": 50},
            steps=steps,
        )
    ]

    fig.update_layout(sliders=sliders)

    fig.update_layout(
        height=300 * num_matrices / 3,  # Adjust height based on the number of matrices
        coloraxis={"colorscale": "Viridis"},
        coloraxis_cmin=min_val,
        coloraxis_cmax=max_val,
        showlegend=False,
    )
    fig.update_traces(texttemplate="%{z:.2f}", showlegend=True)

    return fig


# Read the CSV file
@st.cache_data()
def get_metadata_df() -> pd.DataFrame:
    return pd.read_csv("../../data/results_2_23_25/bulk_metadata.csv")


@st.cache_data()
def get_events_df() -> pd.DataFrame:
    return pd.read_csv("../../data/results_2_23_25/bulk_events.csv", index_col=0)


@st.cache_data()
def get_model_dict() -> dict[str, list[dict[str, Any]]]:
    return torch.load("../../data/results_2_23_25/bulk_models.pt", map_location="cpu")


@st.cache_data()
def get_state_dict_type_adapter() -> TypeAdapter:
    return TypeAdapter(
        list[dict[str, torch.Tensor]],
        config=ConfigDict(arbitrary_types_allowed=True),
    )


@st.cache_data()
def load_specific_models(
    _bulk_models_dict: dict[str, list[dict[str, Any]]], uuid: str
) -> dict[str, torch.Tensor]:
    specific_model_params: list[dict[str, torch.Tensor]] = [
        SingleHeadTransformerOneHotPositionalNoMLP.load_from_dict(
            model_dict, map_device=torch.device("cpu")
        ).state_dict()
        for model_dict in _bulk_models_dict[uuid]
    ]
    specific_model_params = get_state_dict_type_adapter().validate_python(
        specific_model_params
    )
    return {
        name: torch.stack(
            [
                specific_model_param[name]
                for specific_model_param in specific_model_params
            ]
        )
        for name in specific_model_params[0].keys()
    }


metadata_df = get_metadata_df()
events_df = get_events_df()
bulk_model_dict = get_model_dict()

head_dims = len(list(filter(lambda string: "eval_acc/" in string, events_df.columns)))

# Display the dataframe
st.write("Training Runs: (SELECT ONE)")

shown_df = metadata_df.sort_values(by="max_eval_acc", ascending=False).drop_duplicates(
    ["function_uuid"]
)

selection = st.dataframe(
    shown_df,
    selection_mode="single-row",
    on_select="rerun",
)
if len(selection["selection"]["rows"]) > 0:  # type: ignore
    row = selection["selection"]["rows"][0]  # type: ignore
    uuid = shown_df.iloc[selection["selection"]["rows"]]["run_uuid"].item()  # type: ignore
    st.write(uuid)

    st.dataframe(events_df.loc[events_df["uuid"] == uuid])

    comps = json.loads(metadata_df.loc[row, "comps"])  # type: ignore
    coeffs = json.loads(metadata_df.loc[row, "coeffs"])  # type: ignore

    st.header("Function Components")
    st.plotly_chart(
        px.imshow(
            torch.tensor(comps), y=[f"Component 1: {coeff:.2f}" for coeff in coeffs]
        )
    )

    st.header("Model Performance")

    head_dim_selection: str = st.segmented_control(
        "Attention Head Dimensions",
        ["All Dims", *[f"Head Dim {i}" for i in range(head_dims)]],
        default="All Dims",
        selection_mode="single",
    )  # type: ignore

    if head_dim_selection == "All Dims":
        st.plotly_chart(
            px.line(
                events_df.loc[events_df["uuid"] == uuid],
                x="epoch",
                y=[
                    "loss",
                    "train_acc",
                    "eval_acc",
                    "eval_acc_0",
                    "eval_acc_1",
                    "eval_acc_2",
                    "eval_acc_3",
                ],
            )
        )
    else:
        dim = int(head_dim_selection.split("Head Dim ")[1])
        st.plotly_chart(
            px.line(
                events_df.loc[events_df["uuid"] == uuid],
                x="epoch",
                y=[
                    "loss",
                    f"eval_acc/head_{dim}",
                    f"eval_acc_0/head_{dim}",
                    f"eval_acc_1/head_{dim}",
                    f"eval_acc_2/head_{dim}",
                    f"eval_acc_3/head_{dim}",
                ],
            )
        )
    state_dict = load_specific_models(bulk_model_dict, uuid)

    st.plotly_chart(visualize_state_dict_at_epoch(state_dict))

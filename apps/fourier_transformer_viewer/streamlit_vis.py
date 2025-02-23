import json

import pandas as pd
import plotly.express as px
import streamlit as st
import torch

st.title("Fourier Component Training Results")


# Read the CSV file
@st.cache_data()
def get_metadata_df() -> pd.DataFrame:
    return pd.read_csv("../../data/results_2_23_25/bulk_metadata.csv")


def get_events_df() -> pd.DataFrame:
    return pd.read_csv("../../data/results_2_23_25/bulk_events.csv", index_col=0)


metadata_df = get_metadata_df()
events_df = get_events_df()

head_dims = len(list(filter(lambda string: "eval_acc/" in string, events_df.columns)))

# Display the dataframe
st.write("Training Runs: (SELECT ONE)")

selection = st.dataframe(
    metadata_df.sort_values(by="max_eval_acc", ascending=False).drop_duplicates(
        ["function_uuid"]
    ),
    selection_mode="single-row",
    on_select="rerun",
)
if len(selection["selection"]["rows"]) > 0:  # type: ignore
    row = selection["selection"]["rows"][0]  # type: ignore
    uuid = metadata_df.loc[selection["selection"]["rows"], "run_uuid"].item()  # type: ignore
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
                    "train_acc",
                    f"eval_acc/head_{dim}",
                    f"eval_acc_0/head_{dim}",
                    f"eval_acc_1/head_{dim}",
                    f"eval_acc_2/head_{dim}",
                    f"eval_acc_3/head_{dim}",
                ],
            )
        )

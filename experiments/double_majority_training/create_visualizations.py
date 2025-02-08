import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
from plotly.subplots import make_subplots

from miexp.models.interptransformer import (
    SingleHeadTransformerNoEmbeddingNoMLP,
)

pio.kaleido.scope.mathjax = None

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

results = pd.read_csv("./results.csv")

train_results_fig = px.line(results, y=["loss", "acc", "eval_acc", "low_eval_acc"])
train_results_fig.write_image("train_results_fig.svg")

state_dict = torch.load("./model.pth", map_location=device, weights_only=False)

model = SingleHeadTransformerNoEmbeddingNoMLP(vocab_size=2, head_dim=2)
model.load_state_dict(state_dict)

# Filter out biases from state_dict
filtered_state_dict = (
    state_dict  # {k: v for k, v in state_dict.items() if 'bias' not in k}
)


# Create a subplot grid
num_matrices = len(filtered_state_dict)
fig = make_subplots(
    rows=num_matrices // 3 + 1,
    cols=3,
    subplot_titles=list(state_dict.keys()) + ["Effective QKV", "Effective QKVO"],
)

fig.update_layout(
    height=300 * num_matrices / 3,  # Adjust height based on the number of matrices
    coloraxis={"colorscale": "Viridis"},
    showlegend=False,
)

with open("numbers.json", "w") as f:
    contributions = (
        state_dict["attention_head.w_q.weight"][0, 2]
        * state_dict["attention_head.w_k.weight"]
        * state_dict["attention_head.w_v.weight"]
    ).cpu()
    json.dump(contributions.tolist(), f)

final_contributions = state_dict["attention_head.w_o.weight"] @ contributions

# Add a heatmap for each matrix in state_dict
for i, (name, matrix) in enumerate(filtered_state_dict.items()):
    fig.add_trace(
        go.Heatmap(z=matrix.cpu().numpy(), coloraxis="coloraxis"),
        row=(i // 3) + 1,
        col=(i % 3) + 1,
    )

i = len(filtered_state_dict)
fig.add_trace(
    go.Heatmap(
        z=contributions.cpu().numpy(),
        coloraxis="coloraxis",
    ),
    row=(i // 3) + 1,
    col=(i % 3) + 1,
)

fig.add_trace(
    go.Heatmap(
        z=final_contributions.cpu().numpy(),
        coloraxis="coloraxis",
    ),
    row=((i + 1) // 3) + 1,
    col=((i + 1) % 3) + 1,
)


fig.update_traces(texttemplate="%{z:.2f}")

fig.write_image("parameter_view_fig.svg", width=1000)

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import torch
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader
from train_transformer import eval_epoch

from miexp.models.interptransformer import (
    SingleHeadTransformerNoEmbedding,
)

pio.kaleido.scope.mathjax = None

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

results = pd.read_csv("./results.csv")

train_results_fig = px.line(results, y=["loss", "acc", "eval_acc"])
train_results_fig.write_image("train_results_fig.svg")

state_dict = torch.load("./model.pth", map_location=device, weights_only=False)

dataset = torch.load("./dataset.pt", map_location=device, weights_only=False)

model = SingleHeadTransformerNoEmbedding(vocab_size=2, head_dim=1)
model.load_state_dict(state_dict)

eval_dl = DataLoader(dataset["eval"], batch_size=32, shuffle=False)
res = eval_epoch(model, eval_dl, device=device)

eval_results = eval_epoch(model, eval_dl, device)
eval_acc = torch.argmax(
    torch.tensor(eval_results["probabilities"]), dim=1
) == torch.tensor(eval_results["correct_outputs"])
eval_acc.float().mean().item()

torch.set_printoptions(sci_mode=False)

# Filter out biases from state_dict
filtered_state_dict = (
    state_dict  # {k: v for k, v in state_dict.items() if 'bias' not in k}
)


# Create a subplot grid
num_matrices = len(filtered_state_dict)
fig = make_subplots(
    rows=num_matrices // 3 + 1, cols=3, subplot_titles=list(state_dict.keys())
)

fig.update_layout(
    height=300 * num_matrices / 3,  # Adjust height based on the number of matrices
    coloraxis={"colorscale": "Viridis"},
    showlegend=False,
)

# Add a heatmap for each matrix in state_dict
for i, (name, matrix) in enumerate(filtered_state_dict.items()):
    fig.add_trace(
        go.Heatmap(z=matrix.cpu().numpy(), coloraxis="coloraxis"),
        row=(i // 3) + 1,
        col=(i % 3) + 1,
    )

fig.write_image("parameter_view_fig.svg", width=1000)

with open("numbers.json", "w") as f:
    contributions = (
        (
            state_dict["attention_head.w_q.weight"][0, 2]
            * state_dict["attention_head.w_k.weight"]
            * state_dict["attention_head.w_v.weight"]
        )
        .cpu()
        .flatten()
    )
    json.dump(contributions.tolist(), f)

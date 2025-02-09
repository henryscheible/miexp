import pandas as pd
import plotly.express as px
import plotly.io as pio
import torch

from miexp.models.interptransformer import (
    SingleHeadTransformerOneHotPositionalNoMLP,
)

pio.kaleido.scope.mathjax = None

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

results = pd.read_csv("./results.csv")

train_results_fig = px.line(
    results,
    y=["lr", "loss", "acc", *[item for item in results.columns if "eval_acc" in item]],
)
train_results_fig.show()
train_results_fig.write_image("train_results_fig.svg")

model = SingleHeadTransformerOneHotPositionalNoMLP.load_from_checkpoint("./model.pth")


# # Create a subplot grid
# num_matrices = len(filtered_state_dict)
# fig = make_subplots(
#     rows=num_matrices // 3 + 1,
#     cols=3,
#     subplot_titles=list(state_dict.keys()) + ["Effective QKV", "Effective QKVO"],
# )

# fig.update_layout(
#     height=300 * num_matrices / 3,  # Adjust height based on the number of matrices
#     coloraxis={"colorscale": "Viridis"},
#     showlegend=False,
# )

# with open("numbers.json", "w") as f:
#     contributions = (
#         state_dict["attention_head.w_q.weight"][0, 2]
#         * state_dict["attention_head.w_k.weight"]
#         * state_dict["attention_head.w_v.weight"]
#     ).cpu()
#     json.dump(contributions.tolist(), f)

# final_contributions = state_dict["attention_head.w_o.weight"] @ contributions

# # Add a heatmap for each matrix in state_dict
# for i, (name, matrix) in enumerate(filtered_state_dict.items()):
#     fig.add_trace(
#         go.Heatmap(z=matrix.cpu().numpy(), coloraxis="coloraxis"),
#         row=(i // 3) + 1,
#         col=(i % 3) + 1,
#     )

# i = len(filtered_state_dict)
# fig.add_trace(
#     go.Heatmap(
#         z=contributions.cpu().numpy(),
#         coloraxis="coloraxis",
#     ),
#     row=(i // 3) + 1,
#     col=(i % 3) + 1,
# )

# fig.add_trace(
#     go.Heatmap(
#         z=final_contributions.cpu().numpy(),
#         coloraxis="coloraxis",
#     ),
#     row=((i + 1) // 3) + 1,
#     col=((i + 1) % 3) + 1,
# )


# fig.update_traces(texttemplate="%{z:.2f}")

# fig.write_image("parameter_view_fig.svg", width=1000)

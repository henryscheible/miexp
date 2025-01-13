import plotly.express as px
import streamlit as st
import torch

from miexp.models.btransformer import BooleanTransformer

st.title("Transformer Viewer")


@st.cache_data()
def load_transformer():
    return BooleanTransformer.load_from_checkpoint(
        "./checkpoints/example_transformer.ckpt"
    )


transformer = load_transformer()

input = st.text_input("Input Sequence")

input_tensor = torch.tensor([[int(c) for c in input]])

B, N = input_tensor.shape
st.write(B, N)

int_input = torch.cat(
    [
        2 * torch.ones((B, 1), dtype=torch.int).to(input_tensor.device),
        input_tensor.type(torch.int),
    ],
    dim=1,
)
embedding = transformer.embedding(int_input)
output_sequence = transformer.transformer_layer(embedding)

st.write(int_input)
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.imshow(embedding.squeeze().detach().cpu().numpy()[:, :10].T))
with c2:
    st.plotly_chart(
        px.imshow(output_sequence.squeeze().detach().cpu().numpy()[:, :10].T)
    )


st.write(transformer(input_tensor))

for name, p in transformer.named_parameters():
    st.write(f"Parameter: {name}")
    st.write(p.detach().numpy())

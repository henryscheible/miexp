import pytest
import torch

from miexp.transformer import Transformer


@pytest.fixture
def transformer():
    return Transformer(
        embed_dim=8,
        num_heads=2,
        P=7,
        device="cpu",
        mlp_neurons=4,
    )


def test_transformer_inits(transformer):
    assert isinstance(transformer, Transformer)

def test_forward_runs(transformer: Transformer):
    example_input = transformer.get_sample_input(
        batch_size=2, 
    )
    output = transformer.forward(example_input)
    assert isinstance(output, torch.Tensor)
    assert output.shape == (2, 7)
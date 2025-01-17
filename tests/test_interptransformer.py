import torch

from miexp.models.interptransformer import AttentionHead, AttentionLayer


def test_attention_layer_initialization():
    n_heads = 4
    hidden_dim = 64
    head_dim = 16
    attention_layer = AttentionLayer(n_heads, head_dim, hidden_dim)

    # Check if the number of attention heads is correct
    assert len(attention_layer.attention_heads) == n_heads

    # Check if each attention head is an instance of AttentionHead
    for head in attention_layer.attention_heads:
        assert isinstance(head, AttentionHead)

    # Check if the MLP is a sequential model with the correct layers
    assert isinstance(attention_layer.mlp, torch.nn.Sequential)
    assert len(attention_layer.mlp) == 3
    assert isinstance(attention_layer.mlp[0], torch.nn.Linear)
    assert attention_layer.mlp[0].in_features == hidden_dim
    assert attention_layer.mlp[0].out_features == hidden_dim
    assert isinstance(attention_layer.mlp[1], torch.nn.ReLU)
    assert isinstance(attention_layer.mlp[2], torch.nn.Linear)
    assert attention_layer.mlp[2].in_features == hidden_dim
    assert attention_layer.mlp[2].out_features == hidden_dim


def test_attention_layer_forward():
    n_heads = 4
    hidden_dim = 64
    head_dim = 16
    attention_layer = AttentionLayer(n_heads, head_dim, hidden_dim)

    # Create a random input tensor with the appropriate shape
    batch_size = 8
    seq_length = 10
    x = torch.randn(batch_size, seq_length, hidden_dim)

    # Perform a forward pass
    y = attention_layer(x)

    # Check if the output tensor has the correct shape
    assert y.shape == (batch_size, seq_length, hidden_dim)

    # Check if the output tensor is a torch.Tensor
    assert isinstance(y, torch.Tensor)


def test_attention_head_initialization():
    hidden_dim = 64
    head_dim = 16
    attention_head = AttentionHead(head_dim, hidden_dim)

    # Check if the hidden_dim is set correctly
    assert attention_head.hidden_dim == hidden_dim

    # Check if the linear layers are initialized correctly
    assert isinstance(attention_head.w_q, torch.nn.Linear)
    assert attention_head.w_q.in_features == hidden_dim
    assert attention_head.w_q.out_features == head_dim

    assert isinstance(attention_head.w_k, torch.nn.Linear)
    assert attention_head.w_k.in_features == hidden_dim
    assert attention_head.w_k.out_features == head_dim

    assert isinstance(attention_head.w_v, torch.nn.Linear)
    assert attention_head.w_v.in_features == hidden_dim
    assert attention_head.w_v.out_features == head_dim

    assert isinstance(attention_head.w_o, torch.nn.Linear)
    assert attention_head.w_o.in_features == head_dim
    assert attention_head.w_o.out_features == hidden_dim


def test_attention_head_forward():
    hidden_dim = 64
    head_dim = 16

    attention_head = AttentionHead(head_dim, hidden_dim)

    # Create a random input tensor with the appropriate shape
    batch_size = 8
    seq_length = 10
    x = torch.randn(batch_size, seq_length, hidden_dim)

    # Perform a forward pass
    y = attention_head(x)

    # Check if the output tensor has the correct shape
    assert y.shape == (batch_size, seq_length, hidden_dim)

    # Check if the output tensor is a torch.Tensor
    assert isinstance(y, torch.Tensor)

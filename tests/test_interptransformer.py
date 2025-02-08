import torch

from miexp.models.interptransformer import (
    AttentionHead,
    AttentionLayer,
    SingleHeadTransformer,
    SingleHeadTransformerNoEmbedding,
    SingleHeadTransformerOneHotPositionalNoMLP,
)


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


def test_single_head_transformer_initialization():
    vocab_size = 100
    head_dim = 16
    hidden_dim = 64
    transformer = SingleHeadTransformer(vocab_size, head_dim, hidden_dim)

    # Check if the embedding layer is initialized correctly
    assert isinstance(transformer.embedding, torch.nn.Embedding)
    assert transformer.embedding.num_embeddings == vocab_size + 1
    assert transformer.embedding.embedding_dim == hidden_dim

    # Check if the attention head is an instance of AttentionHead
    assert isinstance(transformer.attention_head, AttentionHead)
    assert transformer.attention_head.head_dim == head_dim
    assert transformer.attention_head.hidden_dim == hidden_dim

    # Check if the MLP is a sequential model with the correct layers
    assert isinstance(transformer.mlp, torch.nn.Sequential)
    assert len(transformer.mlp) == 3
    assert isinstance(transformer.mlp[0], torch.nn.Linear)
    assert transformer.mlp[0].in_features == hidden_dim
    assert transformer.mlp[0].out_features == hidden_dim
    assert isinstance(transformer.mlp[1], torch.nn.ReLU)
    assert isinstance(transformer.mlp[2], torch.nn.Linear)
    assert transformer.mlp[2].in_features == hidden_dim
    assert transformer.mlp[2].out_features == hidden_dim

    # Check if the unembedding layer is initialized correctly
    assert isinstance(transformer.unembedding, torch.nn.Linear)
    assert transformer.unembedding.in_features == hidden_dim
    assert transformer.unembedding.out_features == vocab_size + 1


def test_single_head_transformer_forward():
    vocab_size = 100
    head_dim = 16
    hidden_dim = 64
    transformer = SingleHeadTransformer(vocab_size, head_dim, hidden_dim)

    # Create a random input tensor with the appropriate shape
    batch_size = 8
    seq_length = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Perform a forward pass
    y = transformer(x)

    # Check if the output tensor has the correct shape
    assert y.shape == (batch_size, vocab_size + 1)

    # Check if the output tensor is a torch.Tensor
    assert isinstance(y, torch.Tensor)


def test_single_head_transformer_no_embedding_initialization():
    vocab_size = 100
    head_dim = 16
    transformer = SingleHeadTransformerNoEmbedding(vocab_size, head_dim)

    hidden_dim = vocab_size + 1

    # Check if the embedding layer is initialized correctly
    assert isinstance(transformer.embedding, torch.nn.Embedding)
    assert transformer.embedding.num_embeddings == hidden_dim
    assert transformer.embedding.embedding_dim == hidden_dim
    assert torch.equal(transformer.embedding.weight, torch.eye(hidden_dim))
    assert not transformer.embedding.weight.requires_grad

    # Check if the attention head is an instance of AttentionHead
    assert isinstance(transformer.attention_head, AttentionHead)
    assert transformer.attention_head.head_dim == head_dim
    assert transformer.attention_head.hidden_dim == hidden_dim

    # Check if the MLP is a sequential model with the correct layers
    assert isinstance(transformer.mlp, torch.nn.Sequential)
    assert len(transformer.mlp) == 3
    assert isinstance(transformer.mlp[0], torch.nn.Linear)
    assert transformer.mlp[0].in_features == hidden_dim
    assert transformer.mlp[0].out_features == hidden_dim
    assert isinstance(transformer.mlp[1], torch.nn.ReLU)
    assert isinstance(transformer.mlp[2], torch.nn.Linear)
    assert transformer.mlp[2].in_features == hidden_dim
    assert transformer.mlp[2].out_features == hidden_dim

    # Check if the unembedding layer is initialized correctly
    assert isinstance(transformer.unembedding, torch.nn.Linear)
    assert transformer.unembedding.in_features == hidden_dim
    assert transformer.unembedding.out_features == hidden_dim
    assert torch.equal(transformer.unembedding.weight, torch.eye(hidden_dim))
    assert not transformer.unembedding.weight.requires_grad


def test_single_head_transformer_no_embedding_forward():
    vocab_size = 100
    head_dim = 16
    transformer = SingleHeadTransformerNoEmbedding(vocab_size, head_dim)

    # Create a random input tensor with the appropriate shape
    batch_size = 8
    seq_length = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Perform a forward pass
    y = transformer(x)

    # Check if the output tensor has the correct shape
    assert y.shape == (batch_size, vocab_size + 1)

    # Check if the output tensor is a torch.Tensor
    assert isinstance(y, torch.Tensor)


def test_single_head_transformer_one_hot_positional_no_mlp_initialization():
    vocab_size = 100
    head_dim = 16
    max_seq_len = 50
    transformer = SingleHeadTransformerOneHotPositionalNoMLP(
        vocab_size, head_dim, max_seq_len
    )

    hidden_dim = vocab_size + 1 + max_seq_len

    # Check if the embedding layer is initialized correctly
    assert isinstance(transformer.embedding, torch.nn.Embedding)
    assert transformer.embedding.num_embeddings == vocab_size + 1
    assert transformer.embedding.embedding_dim == vocab_size + 1
    assert torch.equal(transformer.embedding.weight, torch.eye(vocab_size + 1))
    assert not transformer.embedding.weight.requires_grad

    # Check if the positional embedding layer is initialized correctly
    assert isinstance(transformer.pos_embedding, torch.nn.Embedding)
    assert transformer.pos_embedding.num_embeddings == max_seq_len
    assert transformer.pos_embedding.embedding_dim == max_seq_len
    assert torch.equal(transformer.pos_embedding.weight, torch.eye(max_seq_len))
    assert not transformer.pos_embedding.weight.requires_grad

    # Check if the attention head is an instance of AttentionHead
    assert isinstance(transformer.attention_head, AttentionHead)
    assert transformer.attention_head.head_dim == head_dim
    assert transformer.attention_head.hidden_dim == hidden_dim

    # Check if the unembedding layer is initialized correctly
    assert isinstance(transformer.unembedding, torch.nn.Linear)
    assert transformer.unembedding.in_features == hidden_dim
    assert transformer.unembedding.out_features == vocab_size + 1


def test_single_head_transformer_one_hot_positional_no_mlp_forward():
    vocab_size = 100
    head_dim = 16
    max_seq_len = 50
    transformer = SingleHeadTransformerOneHotPositionalNoMLP(
        vocab_size, head_dim, max_seq_len
    )

    # Create a random input tensor with the appropriate shape
    batch_size = 8
    seq_length = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Perform a forward pass
    y = transformer(x)

    # Check if the output tensor has the correct shape
    assert y.shape == (batch_size, vocab_size + 1)

    # Check if the output tensor is a torch.Tensor
    assert isinstance(y, torch.Tensor)

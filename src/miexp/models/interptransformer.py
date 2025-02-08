import math

import torch
from torch import nn

# class InterpTransformer(nn.Module):
#     pass


class MultiHeadAttention(nn.Module):
    """MultiHeadAttention is a neural network module that applies multiple attention heads followed by a multi-layer perceptron (MLP)."""

    def __init__(self, n_heads: int, head_dim: int, hidden_dim: int) -> None:
        """Initializes the MultiHeadAttention.

        Args:
            n_heads (int): The number of attention heads.
            head_dim (int): The dimensionality of each attention head.
            hidden_dim (int): The dimensionality of the hidden layers.

        Attributes:
            attention_heads (nn.ModuleList): A list of attention heads.
            mlp (nn.Sequential): A multi-layer perceptron with two linear layers and a ReLU activation in between.
        """
        super().__init__()
        self.attention_heads = nn.ModuleList(
            [AttentionHead(head_dim, hidden_dim) for _ in range(n_heads)]
        )
        self.w_o = nn.Linear(n_heads * head_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the InterpTransformer model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention heads and MLP.
        """
        x = (
            self.w_o(torch.concat([head(x) for head in self.attention_heads], dim=-1))
            + x
        )
        return x


class AttentionHead(nn.Module):
    """AttentionHead is a neural network module that performs self-attention."""

    def __init__(self, head_dim: int, hidden_dim: int) -> None:
        """Initializes the AttentionHead.

        Args:
            head_dim (int): The dimensionality of the attention head.
            hidden_dim (int): The dimensionality of the hidden layers.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.w_q = nn.Linear(hidden_dim, head_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, head_dim, bias=False)
        self.w_v = nn.Linear(hidden_dim, head_dim, bias=False)
        # self.w_o = nn.Linear(head_dim, hidden_dim) --> Should not need this for a single head.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass of the transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, hidden_dim).
        """
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        A = torch.softmax(q @ k.transpose(-1, -2) / (self.head_dim**0.5), dim=-1)  # noqa: N806
        y = A @ v
        return y


class PositionalEncoding(nn.Module):
    """PositionalEncoding is a neural network module that adds positional encoding to the input."""

    def __init__(self, d_model: int, max_len: int = 20):
        """Initializes the PositionalEncoding.

        Args:
            d_model (int): The dimensionality of the input.
            max_len (int, optional): The maximum length of the input. Defaults to 20.
        """
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # Apply sin to even indices and cos to odd indices
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position[:, : d_model // 2] * div_term)

        # Register as buffer so it’s not a parameter
        self.register_buffer("positional_encoding", self.encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the positional encoding."""
        seq_len = x.size(1)
        return x + self.encoding[:seq_len, :]


class MultiHeadTransformerPosEmb(nn.Module):
    """MultiHeadTransformer is a neural network module that applies an embedding, then multiple attention heads followed by a multi-layer perceptron (MLP), followed by an unembedding."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        hidden_dim: int,
    ) -> None:
        """Initializes the MultiHeadTransformer.

        Args:
            vocab_size (int): The size of the vocabulary.
            max_seq_len (int): The maximum length of the input sequence.
            n_heads (int): The number of attention heads.
            head_dim (int): The dimensionality of each attention head.
            hidden_dim (int): The dimensionality of the hidden layer.
        """
        super().__init__()
        # hidden_dim = vocab_size + 1
        self.pos_embedding = PositionalEncoding(hidden_dim, max_seq_len)
        self.embedding = nn.Embedding(vocab_size + 1, hidden_dim)
        # self.embedding.weight = nn.Parameter(torch.eye(vocab_size + 1))
        # self.embedding.weight.requires_grad = False
        self.attention = MultiHeadAttention(n_heads, head_dim, hidden_dim)
        self.unembedding = nn.Linear(hidden_dim, vocab_size + 1)
        self.unembedding = nn.Linear(hidden_dim, vocab_size + 1, bias=False)
        # self.unembedding.weight = nn.Parameter(torch.eye(vocab_size + 1))
        # self.unembedding.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass of the transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, input_dim).
        """
        x = torch.cat(
            [torch.ones(x.shape[0], 1, dtype=torch.long, device=x.device) * 2, x], dim=1
        )
        y = self.embedding(x)
        y = self.pos_embedding(y)
        y = self.attention(y) + y
        y = self.unembedding(y[:, 0, :])
        return y


class SingleHeadTransformer(nn.Module):
    """SingleHeadTransformer is a neural network module that applies an embedding, then a single attention head followed by a multi-layer perceptron (MLP), followed by an unembedding."""

    def __init__(self, vocab_size: int, head_dim: int, hidden_dim: int) -> None:
        """Initializes the SingleHeadTransformer.

        Args:
            vocab_size (int): The size of the vocabulary.
            head_dim (int): The dimensionality of the attention head.
            hidden_dim (int): The dimensionality of the hidden layer.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, hidden_dim)
        self.attention_head = AttentionHead(head_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.unembedding = nn.Linear(hidden_dim, vocab_size + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass of the transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, input_dim).
        """
        x = torch.cat(
            [torch.ones(x.shape[0], 1, dtype=torch.long, device=x.device) * 2, x], dim=1
        )
        y = self.embedding(x)
        y = self.attention_head(y) + y
        y = self.mlp(y) + y
        y = self.unembedding(y[:, 0, :])
        return y


class SingleHeadTransformerNoEmbedding(nn.Module):
    """SingleHeadTransformer is a neural network module that applies an embedding, then a single attention head followed by a multi-layer perceptron (MLP), followed by an unembedding."""

    def __init__(self, vocab_size: int, head_dim: int) -> None:
        """Initializes the SingleHeadTransformer.

        Args:
            vocab_size (int): The size of the vocabulary.
            head_dim (int): The dimensionality of the attention head.
            hidden_dim (int): The dimensionality of the hidden layer.
        """
        super().__init__()
        hidden_dim = vocab_size + 1
        self.embedding = nn.Embedding(vocab_size + 1, hidden_dim)
        self.embedding.weight = nn.Parameter(torch.eye(vocab_size + 1))
        self.embedding.weight.requires_grad = False
        self.attention_head = AttentionHead(head_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.unembedding = nn.Linear(hidden_dim, vocab_size + 1, bias=False)
        self.unembedding.weight = nn.Parameter(torch.eye(vocab_size + 1))
        self.unembedding.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass of the transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, input_dim).
        """
        x = torch.cat(
            [torch.ones(x.shape[0], 1, dtype=torch.long, device=x.device) * 2, x], dim=1
        )
        y = self.embedding(x)
        y = self.attention_head(y) + y
        # y = self.mlp(y) + y
        y = self.unembedding(y[:, 0, :])
        return y


class SingleHeadTransformerNoEmbeddingNoMLP(nn.Module):
    """SingleHeadTransformer is a neural network module that applies an embedding, then a single attention head followed by a multi-layer perceptron (MLP), followed by an unembedding."""

    def __init__(self, vocab_size: int, head_dim: int) -> None:
        """Initializes the SingleHeadTransformer.

        Args:
            vocab_size (int): The size of the vocabulary.
            head_dim (int): The dimensionality of the attention head.
            hidden_dim (int): The dimensionality of the hidden layer.
        """
        super().__init__()
        hidden_dim = vocab_size + 1
        self.embedding = nn.Embedding(vocab_size + 1, hidden_dim)
        self.embedding.weight = nn.Parameter(torch.eye(vocab_size + 1))
        self.embedding.weight.requires_grad = False
        self.attention_head = AttentionHead(head_dim, hidden_dim)
        self.unembedding = nn.Linear(hidden_dim, vocab_size + 1, bias=False)
        self.unembedding.weight = nn.Parameter(torch.eye(vocab_size + 1))
        self.unembedding.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass of the transformer model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, input_dim).
        """
        x = torch.cat(
            [torch.ones(x.shape[0], 1, dtype=torch.long, device=x.device) * 2, x], dim=1
        )
        y = self.embedding(x)
        y = self.attention_head(y) + y
        y = self.unembedding(y[:, 0, :])
        return y


class AttentionLayer(nn.Module):
    """AttentionLayer is a neural network module that applies multiple attention heads followed by a multi-layer perceptron (MLP)."""

    def __init__(self, n_heads: int, head_dim: int, hidden_dim: int) -> None:
        """Initializes the AttentionLayer.

        Args:
            n_heads (int): The number of attention heads.
            head_dim (int): The dimensionality of each attention head.
            hidden_dim (int): The dimensionality of the hidden layers.

        Attributes:
            attention_heads (nn.ModuleList): A list of attention heads.
            mlp (nn.Sequential): A multi-layer perceptron with two linear layers and a ReLU activation in between.
        """
        super().__init__()
        self.attention_heads = nn.ModuleList(
            [AttentionHead(head_dim, hidden_dim) for _ in range(n_heads)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the InterpTransformer model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention heads and MLP.
        """
        y = sum([head(x) for head in self.attention_heads])
        y = self.mlp(y)
        return y

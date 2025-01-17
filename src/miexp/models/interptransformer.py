import torch
from torch import nn

# class InterpTransformer(nn.Module):
#     pass


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
        self.w_q = nn.Linear(hidden_dim, head_dim)
        self.w_k = nn.Linear(hidden_dim, head_dim)
        self.w_v = nn.Linear(hidden_dim, head_dim)
        self.w_o = nn.Linear(head_dim, hidden_dim)

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
        y = self.w_o(A @ v)
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


# class MLPLayer(nn.Module):
#     pass

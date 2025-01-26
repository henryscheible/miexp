# ruff : noqa
import math

import torch
from torch.nn import MultiheadAttention
from torch import nn

mps_avail = torch.backends.mps.is_available()
cuda_avail = torch.cuda.is_available()

if mps_avail:
    device = torch.device("mps")
elif cuda_avail:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class AttentionBlock(nn.Module):
    """Attention Block."""

    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiheadAttention(
            hidden_dim, num_heads, bias=False, batch_first=True
        )
        # self.attn = nn.MultiheadAttention(hidden_dim, num_heads, bias=False, batch_first=True)
        # self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        # self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for Attn Block.

        Args:
            x (torch.Tensor): Input

        Returns:
            torch.Tensor: Output
        """
        # x = self.norm1(x + self.attn(x, x, x)[0])
        # x = self.norm2(x + self.linear(x))
        x = x + self.attn(x, x, x)[0]
        x = x + self.linear(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        dropout: float,
        N: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        rank: str,
    ) -> None:
        super().__init__()
        self.N = N
        self.hidden_dim = hidden_dim
        self.h = num_heads
        self.l = num_layers
        self.ff_dim = ff_dim
        self.rank = rank
        self.dropout = dropout
        # Layers
        self.embeddings = torch.nn.Embedding(3, hidden_dim)
        # hidden_dim = hidden_dim + N

        self.transformer = nn.Sequential(
            *[
                AttentionBlock(
                    hidden_dim=hidden_dim,
                    ff_dim=ff_dim,
                    num_heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )

        # Layers/Networks
        self.mlp_head = torch.nn.Sequential(torch.nn.Linear(hidden_dim, 2))

    def makeBitTensor(self, x, N):  # noqa: ANN201, D102, N802
        y = format(x, "b")
        y = ("0" * (N - len(y))) + y
        return [int(z) for z in list(y)]

    def forward(self, x):  # noqa: ANN201, D102
        batch_size = x.shape[0]
        x = torch.cat(
            [
                2 * torch.ones((batch_size, 1), dtype=torch.int).to(x.device),
                x.type(torch.int),
            ],
            dim=1,
        )
        # dat = self.embeddings(x)
        # x = dat
        x = 
        x = self.transformer(x)
        x.to(self.rank)
        x = x[:, -1, :]
        return x

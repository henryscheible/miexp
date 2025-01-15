# ruff : noqa
import math

import torch
import torch.nn.functional as F  # noqa: N812
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

    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, N: int):
        super().__init__()
        self.attn = CustomMHA(hidden_dim, num_heads, bias=False, batch_first=True, N=N)
        # self.attn = nn.MultiheadAttention(hidden_dim, num_heads, bias=False, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-5)
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
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.linear(x))
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
        self.embeddings = torch.nn.Embedding(2, hidden_dim // 2)
        hidden_dim = N + hidden_dim // 2

        # self.positional_embeddings = torch.nn.Embedding(N, hidden_dim//2)
        # self.positional_embeddings = torch.eye(N, N)
        self.transformer = nn.Sequential(
            *[
                AttentionBlock(
                    hidden_dim=hidden_dim,
                    ff_dim=ff_dim,
                    num_heads=num_heads,
                    N=N,
                )
                for _ in range(num_layers)
            ]
        )
        # Layers/Networks
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, ff_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ff_dim, 2),
        )

    def makeBitTensor(self, x, N):  # noqa: ANN201, D102, N802
        y = format(x, "b")
        y = ("0" * (N - len(y))) + y
        return [int(z) for z in list(y)]

    def forward(self, x):  # noqa: ANN201, D102
        batch_size = x.shape[0]
        # inputNum = torch.LongTensor([self.makeBitTensor(num, self.N) for num in x]).to(
        #     self.rank
        # )
        # positional = torch.LongTensor(list(range(0, self.N))).unsqueeze(1).expand(-1, batch_size).T.to(device)
        # pos, dat = self.positional_embeddings(positional), self.embeddings(inputNum)
        pos = (
            torch.eye(self.N, self.N)
            .to(self.rank)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        dat = self.embeddings(x)
        x = torch.cat([pos, dat], dim=2)
        x = self.transformer(x)
        x.to(self.rank)
        x = x[:, 0, :]
        x = self.mlp_head(x)
        return x


class CustomMHA(torch.nn.MultiheadAttention):
    def __init__(self, embed_dim, num_heads, bias, batch_first, N):
        super().__init__(
            embed_dim=embed_dim, num_heads=num_heads, bias=bias, batch_first=batch_first
        )
        self.N = N

    def forward(self, query, key, value):
        is_batched = query.dim() == 3

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # Actual Calculation
        attn_output, attn_output_weights = multi_head_attention_forward(
            query,
            key,
            value,
            num_heads=self.num_heads,
            N=self.N,
            embed_dim_to_check=self.embed_dim,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            dropout_p=self.dropout,
            training=self.training,
            need_weights=True,
            average_attn_weights=True,
        )

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


def multi_head_attention_forward(
    query,
    key,
    value,
    num_heads,
    N,
    embed_dim_to_check,
    in_proj_weight,
    in_proj_bias,
    out_proj_weight,
    out_proj_bias,
    dropout_p=0,
    training=True,
    need_weights=True,
    average_attn_weights=True,
):
    r"""Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        num_heads: parallel attention heads.
        N: size of our boolean domain
        in_proj_weight, in_proj_bias: input projection weight and bias.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        dropout_p: probability of an element to be zeroed.
        training: apply dropout if is ``True``.
        need_weights: output attn_output_weights.

    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    assert embed_dim == embed_dim_to_check, (
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    )
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, (
        f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    )

    #
    # compute in-projection
    #

    E = query.size(-1)
    proj = F.linear(query, in_proj_weight, in_proj_bias)
    proj = (
        proj.unflatten(-1, (3, E))
        .unsqueeze(0)
        .transpose(0, -2)
        .squeeze(-2)
        .contiguous()
    )
    q, k, v = proj[0], proj[1], proj[2]

    #
    # reshape q, k, v for multihead attention and make them batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    # update source sequence length after adjustments
    src_len = k.size(1)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    if need_weights:
        B, Nt, E = q.shape
        q_scaled = q * math.sqrt(1.0 / float(E))

        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = F.dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        )
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights

    else:
        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=False
        )
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        )

        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        return attn_output, None

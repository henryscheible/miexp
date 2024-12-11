from torch import Tensor, nn
import torch


def get_collate_fn(P: int, embed_dim: int):
    def collate_fn(inputs: list[tuple[int, int]]):
        output_batch = torch.zeros((len(inputs), 3, embed_dim))
        for i, input in enumerate(inputs):
            output_batch[
                i,
                0,
            ]


class Transformer(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, P: int, device: torch.device | str, mlp_neurons: int, num_summands: int = 2
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.P = P
        self.embed_dim = embed_dim
        self.num_summands = num_summands
        self.embedding = nn.Embedding(
            num_embeddings=P + 1,
            embedding_dim=embed_dim,
            device=device
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings=num_summands + 1,
            embedding_dim=embed_dim,
            device=device
        )
        self.num_heads = num_heads   
        self.per_head_dim = embed_dim // num_heads     
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
        self.MLP_in = nn.Linear(embed_dim, mlp_neurons, bias=True)
        self.MLP_out = nn.Linear(mlp_neurons, embed_dim, bias=True)
        self.unembedding = nn.Linear(embed_dim, P, bias=False)

    def get_sample_input(self, batch_size: int):
        ex = torch.randint(
            low=0,
            high=self.P,
            size=(batch_size, self.num_summands + 1),
            device=self.device
        )
        ex[:, 0] = self.P
        return ex

    def forward(self, inputs: Tensor):
        assert inputs.shape[1] == self.num_summands + 1
        # inputs: (B, N) (input ids)
        embeds = self.embedding(inputs) + self.pos_embedding(torch.arange(self.num_summands + 1, device=self.device))
        # embeds shape: (B, N, E)
        Q, K, V = self.W_Q(embeds), self.W_K(embeds), self.W_V(embeds)
        attn_maps = [ 
            torch.softmax(
                torch.squeeze(torch.matmul(
                    K[:, :, j*self.per_head_dim:(j+1) *self.per_head_dim], 
                    torch.transpose(Q[:, 0:1, j*self.per_head_dim:(j+1) *self.per_head_dim], -1, -2)
                )), dim=-1
            ) for j in range(self.num_heads)
        ]
        # H: per_head_dim
        # V: (B, N, E), attn_map: [4 * (B, N)]
        attn_outs = self.W_O(torch.concat(
            [torch.matmul(attn_map.unsqueeze(-2), V[:, :, j*self.per_head_dim:(j+1) *self.per_head_dim]).view(-1, self.per_head_dim) for j, attn_map in enumerate(attn_maps)], axis=1
        ))
        h = embeds[:, 0, :] + attn_outs
        mlp_activations = nn.functional.relu(self.MLP_in(h))
        h = h + self.MLP_out(mlp_activations)
        logits = self.unembedding(h)
        return logits
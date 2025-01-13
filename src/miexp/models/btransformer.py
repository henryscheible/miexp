from typing import Self

import pydantic
import torch
from torch import FloatTensor, Tensor, nn


class BooleanTransformer(nn.Module):
    """Transformer which accepts a boolean sequence as an input and predicts a boolean output."""

    def __init__(
        self,
        max_seq_len: int,
        hidden_dim: int,
        n_heads: int,
        num_classifier_hidden_layers: int,
    ):
        """Inits Boolean Transformer.

        Args:
            max_seq_len (int): maximum length of boolean input to transformer.
            hidden_dim (int): hidden dimension of transformer.
            n_heads (int): number of attention heads.
            num_classifier_hidden_layers (int): Number of hidden layers in the classification MLP
        """
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.num_classifier_hidden_layers = num_classifier_hidden_layers
        self.n_heads = n_heads
        self.embedding = nn.Embedding(3, hidden_dim)  # Account for CLS token (2)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(
            hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim
        )
        classifier_layers = []
        for _ in range(num_classifier_hidden_layers):
            classifier_layers.append(nn.Linear(hidden_dim, hidden_dim))
            classifier_layers.append(nn.ReLU())
        classifier_layers.append(nn.Linear(hidden_dim, 2))
        self.classifier = nn.Sequential(*classifier_layers)

    def save_to_checkpoint(self, checkpoint_path: str) -> None:
        """Save the model (including parameters and architectural hyperparameters) to a .pt file.

        Args:
            checkpoint_path (str): Path to save model to
        """
        hyperparameters = {
            "max_seq_len": self.max_seq_len,
            "hidden_dim": self.hidden_dim,
            "num_classifier_hidden_layers": self.num_classifier_hidden_layers,
            "n_heads": self.n_heads,
        }
        state_dict = self.state_dict()
        torch.save(
            {"hyperparameters": hyperparameters, "state_dict": state_dict},
            checkpoint_path,
        )

    @classmethod
    def load_from_checkpoint(
        cls, checkpoint_path: str, map_device: torch.device = torch.device("cpu")
    ) -> Self:
        """Load the model from a checkpoint (including parameters and architectural hyperparameters).

        Args:
            checkpoint_path (str): Path to read the model from. Must unpickle to a dict with "hyperparameters" and "state_dict" keys
            map_device (torch.device | None): device (cpu/gpu) to place the model on. Defaults to cpu

        Returns:
            Self: Instantiated model with given parameters and hyperparameters
        """
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_device,
        )
        model = cls(**checkpoint["hyperparameters"])
        model.load_state_dict(checkpoint["state_dict"])
        if map_device is not None:
            model.to(map_device)
        return model

    @pydantic.validate_call(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def forward(self, input: Tensor) -> FloatTensor:
        """Forward method of Boolean Transformer.

        Args:
            input (BoolTensor): input boolean sequence

        Returns:
            FloatTensor: (B, 2) shaped tensor containing logit that output is 0 or 1
        """
        assert len(input.shape) == 2, "Input must be a 3D Tensor"
        B, N = input.shape  # noqa: N806

        int_input = torch.cat(
            [
                2 * torch.ones((B, 1), dtype=torch.int).to(input.device),
                input.type(torch.int),
            ],
            dim=1,
        )
        embedding = self.embedding(int_input)
        # pos_embedding = self.pos_embedding(int_input)
        output_sequence = self.transformer_layer(embedding)
        output_logits = self.classifier(output_sequence[:, 0, :])
        return output_logits

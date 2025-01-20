import torch
from torch.utils.data import Dataset


class MajDataset(Dataset):
    """Generate a dataset for the majority boolean function."""

    def __init__(self, N: int, num_samples: int = 10000):  # noqa: N803
        """Inits MajDataset.

        Args:
            N (int): Size of the input boolean string
            num_samples (int, optional): Size of the dataset. Defaults to 10000.
        """
        self.N = N
        self.num_samples = num_samples
        self.data = torch.randint(0, 2, (self.num_samples, self.N))
        self.labels = (torch.sum(self.data, dim=1) > (self.N // 2)).type(torch.int)

    def __len__(self) -> int:  # noqa: D105
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D105
        return (self.data[idx], self.labels[idx])

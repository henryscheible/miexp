import torch
from torch import Tensor
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


class DoubleMajDataset(Dataset):
    """Generate a dataset for the majority boolean function."""

    def __init__(
        self,
        N: int,  # noqa: N803
        low: float = 0.3,
        high: float = 0.7,
        num_samples: int = 10000,
    ):
        """Inits MajDataset.

        Args:
            N (int): Size of the input boolean string
            num_samples (int, optional): Size of the dataset. Defaults to 10000.
            low (float): Lower threshold for the function
            high (float): Upper threshold for the function
        """
        self.N = N
        self.num_samples = num_samples
        self.high = high
        self.low = low
        self.data = torch.randint(0, 2, (self.num_samples, self.N)).type(torch.int)
        self.labels = (
            (torch.sum(self.data, dim=1) < (self.N * self.high))
            & (torch.sum(self.data, dim=1) >= (self.N * self.low))
        ).type(torch.int)

    def __len__(self) -> int:  # noqa: D105
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D105
        return (self.data[idx], self.labels[idx])


class RandMajDataset(Dataset):
    """Generate a dataset for the majority boolean function."""

    def __init__(
        self,
        N: int,  # noqa: N803
        frac: float = 0.5,
        num_samples: int = 10000,
    ):
        """Inits MajDataset.

        Args:
            N (int): Size of the input boolean string
            num_samples (int, optional): Size of the dataset. Defaults to 10000.
            frac (float): Fraction of the input to consider
        """
        self.N = N
        self.num_samples = num_samples
        self.frac = frac
        self.data = torch.randint(0, 2, (self.num_samples, self.N)).type(torch.int)
        self.choice = torch.randperm(N)[: int(frac * N)]
        self.labels = (
            torch.sum(self.data[:, self.choice], dim=1) >= int(self.N * frac * 0.5)
        ).type(torch.int)

    def __len__(self) -> int:  # noqa: D105
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D105
        return (self.data[idx], self.labels[idx])


class MultiComponentSpectrumDataset(Dataset):
    """Generate a dataset for the multi-component spectrum boolean function."""

    def __init__(self, N: int, coeffs: Tensor, comps: Tensor, num_samples: int = 10000):  # noqa: N803
        """Inits MultiComponentSpectrumDataset.

        Args:
            N (int): Size of the input boolean string.
            coeffs (Tensor): Coefficients for the components (between -1 and 1) (Shape: (1, num_components))
            comps (Tensor): The fourier components of the function (boolean valued) (Shape: (num_components, N))
            num_samples (int, optional): Size of the dataset. Defaults to 10000.
        """
        assert len(coeffs) == len(comps), (
            "The number of coefficients and components should be the same"
        )
        assert torch.max(torch.abs(coeffs)) <= 1, (
            "The coefficients should be between -1 and 1"
        )
        assert comps.shape[1] == N, (
            "The components should have the same size as the input"
        )
        assert len(comps.shape) == 2, "The components should be a 2D tensor"
        assert len(coeffs.shape) == 1, "The coefficients should be a 2D tensor"

        self.N = N
        self.num_samples = num_samples
        self.data = torch.randint(0, 2, (self.num_samples, self.N)).type(torch.int)
        self.labels = self.generate_labels(self.data, coeffs, comps).type(torch.int)

    def __len__(self) -> int:  # noqa: D105
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D105
        return (self.data[idx], self.labels[idx])

    def generate_labels(self, data: Tensor, coeffs: Tensor, comps: Tensor) -> Tensor:
        """Generate the labels for the data.

        All tensors should have dtype float.

        Args:
            data (Tensor): Data to generate the labels for. (Shape: (num_samples, N))
            coeffs (Tensor): Coeffients for the components. (Shape: (num_components))
            comps (Tensor): The fourier components of the function. (Shape: (num_components, N))

        Returns:
            Tensor: Labels for the data.
        """
        labels = (1 + torch.sign(data.to(torch.float) @ comps.T @ coeffs)) // 2

        return labels

    def get_percent_positive(self) -> float:
        """Calculate the percentage of positive labels.

        Returns:
            float: The mean value of the labels, representing the percentage of positive labels.
        """
        return self.labels.type(torch.float).mean().item()


class MultiComponentDNFDataset(Dataset):
    """Generate a dataset for the multi-component DNF boolean function."""

    def __init__(self, N: int, comps: Tensor, num_samples: int = 10000) -> None:  # noqa: N803
        """Initialize dataset for the multi-component DNF boolean function.

        Args:
            N (int): Size of the input boolean string
            comps (Tensor): The DNF components (ternary valued, 0 indicating not included) (Shape: (num_components, N))
            num_samples (int, optional): Size of the dataset. Defaults to 10000.
        """
        assert comps.shape[1] == N, (
            "The components should have the same size as the input"
        )
        assert len(comps.shape) == 2, "The components should be a 2D tensor"

        self.N = N
        self.num_samples = num_samples
        self.data = torch.randint(0, 2, (self.num_samples, self.N)).type(torch.int)
        comps = comps.to(torch.int)
        self.labels = self.generate_labels(self.data, comps).type(torch.int)

    def __len__(self) -> int:  # noqa: D105
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: D105
        return (self.data[idx], self.labels[idx])

    def generate_labels(self, data: Tensor, comps: Tensor) -> Tensor:
        """Generate the labels for the data.

        All tensors should have dtype int.

        Args:
            data (Tensor): Data to generate the labels for. (Shape: (num_samples, N))
            comps (Tensor): The DNF components of the function. (Shape: (num_components, N))

        Returns:
            Tensor: Labels for the data.
        """
        labels = torch.any(
            (2 * data - 1) @ comps.T == comps.count_nonzero(dim=1), dim=1
        )

        return labels

    def get_percent_positive(self) -> float:
        """Calculate the percentage of positive labels.

        Returns:
            float: The mean value of the labels, representing the percentage of positive labels.
        """
        return self.labels.type(torch.float).mean().item()

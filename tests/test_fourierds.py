import torch

from miexp.bfuncs import MultiComponentSpectrumDataset


def test_fourierds():
    N, num_samples = 20, 1000  # noqa: N806
    # coeffs = torch.tensor([-12 / 13, 1 / 13])
    coeffs = (torch.rand((2,)) * 2) - 1
    comps = torch.randint(0, 2, (2, N)).type(torch.float)
    ds = MultiComponentSpectrumDataset(
        N=N, coeffs=coeffs, comps=comps, num_samples=num_samples
    )

    print(coeffs, torch.sum(ds.labels))
    assert len(ds) == num_samples
    assert ds.data.shape == (num_samples, N)
    assert ds.labels.shape == (num_samples,)
    assert torch.max(ds.labels) in [0, 1] and torch.min(ds.labels) in [0, 1]
    assert torch.max(ds.data) in [0, 1] and torch.min(ds.data) in [0, 1]

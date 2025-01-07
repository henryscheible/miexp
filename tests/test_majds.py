import torch

from miexp.bfuncs import MajDataset


def test_majds():
    N, num_samples = 20, 1000  # noqa: N806
    ds = MajDataset(N, num_samples)

    assert len(ds) == num_samples
    assert ds.data.shape == (num_samples, N)

    row, label = ds[0]

    assert row.shape == (N,)
    assert torch.max(row) in [0, 1] and torch.min(row) in [0, 1]
    assert label == (torch.sum(row) > (N // 2)).type(torch.int)

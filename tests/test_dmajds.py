import torch

from miexp.bfuncs import DoubleMajDataset


def test_majds():
    N, num_samples = 20, 1000  # noqa: N806
    low, high = 0.3, 0.7
    ds = DoubleMajDataset(N, low=low, high=high, num_samples=num_samples)

    assert len(ds) == num_samples
    assert ds.data.shape == (num_samples, N)

    row = ds.data[0]
    assert row.shape == (N,)
    assert torch.max(row) in [0, 1] and torch.min(row) in [0, 1]

    for row, label in zip(ds.data, ds.labels):
        if torch.sum(row) < (N * high) and torch.sum(row) >= (N * low):
            print(N * high, N * low, label)
            print(row, row.sum())
            assert label == 1
        else:
            assert label == 0

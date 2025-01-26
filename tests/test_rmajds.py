import torch

from miexp.bfuncs import RandMajDataset


def test_rmajds():
    N, num_samples = 20, 1000  # noqa: N806
    frac = 0.5
    ds = RandMajDataset(N, frac=frac, num_samples=num_samples)

    assert len(ds) == num_samples
    assert ds.data.shape == (num_samples, N)

    for row, label in ds:
        assert row.shape == (N,)
        selected_row = row[ds.choice]
        assert selected_row.shape == (int(frac * N),)
        assert label == int(torch.sum(selected_row).item() >= (ds.N * 0.5 * ds.frac))

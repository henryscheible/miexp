import torch

from miexp.bfuncs import MajDataset


def test_majdataset_init():
    N = 5
    num_samples = 1000
    dataset = MajDataset(N, num_samples)

    assert dataset.N == N
    assert dataset.num_samples == num_samples
    assert dataset.data.shape == (num_samples, N)
    assert dataset.labels.shape == (num_samples,)
    assert torch.all(
        (dataset.data == 0) | (dataset.data == 1)
    ).item()  # Ensure data is binary
    assert torch.all(
        (dataset.labels == 0) | (dataset.labels == 1)
    ).item()  # Ensure labels are binary
    assert torch.equal(
        dataset.labels, (torch.sum(dataset.data, dim=1) > (N // 2)).type(torch.int)
    )


def test_majdataset_default_num_samples():
    N = 5
    dataset = MajDataset(N)

    assert dataset.num_samples == 10000
    assert dataset.data.shape == (10000, N)
    assert dataset.labels.shape == (10000,)


def test_majdataset_len():
    N = 5
    num_samples = 1000
    dataset = MajDataset(N, num_samples)

    assert len(dataset) == num_samples

    default_dataset = MajDataset(N)
    assert len(default_dataset) == 10000


def test_majdataset_getitem():
    N = 5
    num_samples = 1000
    dataset = MajDataset(N, num_samples)

    for idx in range(10):  # Test the first 10 items
        data, label = dataset[idx]
        assert torch.equal(data, dataset.data[idx])
        assert torch.equal(label, dataset.labels[idx])

    # Test the last item
    data, label = dataset[num_samples - 1]
    assert torch.equal(data, dataset.data[num_samples - 1])
    assert torch.equal(label, dataset.labels[num_samples - 1])

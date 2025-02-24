import torch

from miexp.bfuncs import MultiComponentDNFDataset


def test_dnf():
    N = 10
    num_comps = 5
    comps = torch.randint(0, 2, (num_comps, N)).to(torch.int)
    num_samples = 10000

    dataset = MultiComponentDNFDataset(N, comps, num_samples=num_samples)

    audit_idx = torch.randint(0, num_samples, (20,))

    for idx in audit_idx:
        idx = int(idx)
        data, label = dataset[idx]

        comp_parities = [
            torch.sum(data[comp.to(torch.bool)]).item() % 2 for comp in comps
        ]
        true_label = min(1, sum(comp_parities))

        assert label.item() == true_label


if __name__ == "__main__":
    test_dnf()

import torch

from miexp.bfuncs import MultiComponentDNFDataset


def test_dnf():
    N = 10
    num_comps = 5
    comps = torch.randint(-1, 2, (num_comps, N)).to(torch.int)
    num_samples = 10000

    dataset = MultiComponentDNFDataset(N, comps, num_samples=num_samples)

    audit_idx = torch.randint(0, num_samples, (20,))

    for idx in audit_idx:
        idx = int(idx)
        data, label = dataset[idx]

        true_label = 0
        for comp in comps:
            clause = all(
                [elem == 0 or 2 * data[i] - 1 == elem for i, elem in enumerate(comp)]
            )
            print(clause, 2 * data - 1, comp)

            if clause:
                true_label = 1
                break

        assert label.item() == true_label


if __name__ == "__main__":
    test_dnf()

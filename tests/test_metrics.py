import torch

from miexp.util.metrics import binary_accuracy


def test_binary_accuracy_all_correct():
    logits = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
    labels = torch.tensor([1, 1, 1])
    assert binary_accuracy(logits, labels) == 1.0


def test_binary_accuracy_all_incorrect():
    logits = torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    labels = torch.tensor([1, 1, 1])
    assert binary_accuracy(logits, labels) == 0.0


def test_binary_accuracy_half_correct():
    logits = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])
    labels = torch.tensor([0, 1, 1, 0])
    assert binary_accuracy(logits, labels) == 0.5


def test_binary_accuracy_mixed():
    logits = torch.tensor([[0.6, 0.4], [0.4, 0.6], [0.5, 0.5], [0.7, 0.3]])
    labels = torch.tensor([0, 1, 0, 1])
    assert binary_accuracy(logits, labels) == 0.75

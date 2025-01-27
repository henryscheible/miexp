import pytest
import torch
from torch import Tensor, nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from miexp.train.train_util import eval_epoch, train_epoch


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    return SGD(model.parameters(), lr=0.01)


@pytest.fixture
def criterion():
    return nn.CrossEntropyLoss()


@pytest.fixture
def dataloader():
    inputs = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=10)


def test_train_epoch(model, optimizer, dataloader, device, criterion):
    result = train_epoch(model, optimizer, dataloader, device, criterion)

    assert "acc" in result
    assert "loss" in result
    assert isinstance(result["acc"], float)
    assert isinstance(result["loss"], float)
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert f"norm/{name}" in result
            assert isinstance(result[f"norm/{name}"], float)


def test_eval_epoch(model, dataloader, device):
    result = eval_epoch(model, dataloader, device)

    assert "inputs" in result
    assert "correct_outputs" in result
    assert "probabilities" in result
    assert isinstance(result["inputs"], Tensor)
    assert isinstance(result["correct_outputs"], Tensor)
    assert isinstance(result["probabilities"], Tensor)
    assert (
        len(result["inputs"])
        == len(result["correct_outputs"])
        == len(result["probabilities"])
    )

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float | None]:
    """Trains a binary classification model for one epoch.

    Args:
        model (nn.Module): The neural network model to be trained.
        optimizer (Optimizer): The optimizer used for training.
        dataloader (DataLoader): The DataLoader providing the training data.
        device (torch.device): The device (CPU or GPU) to perform training on.
        criterion (nn.Module): The loss function used to compute the loss.

    Returns:
        dict[str, float | None]: A dictionary containing the training accuracy ('acc'),
                                 the training loss ('loss'), and the gradient norms
                                 for each parameter ('norm/{param_name}').
    """
    model = model.to(device)
    total_train_loss = 0
    total_train_acc = 0
    total_items = 0
    for input, labels in dataloader:
        input = input.to(device)
        labels = labels.to(device)
        output = model(input)
        loss = criterion(output, labels.type(torch.long))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += (loss.item()) * len(input)
        total_train_acc += torch.sum(torch.argmax(output, dim=1) == labels).item()
        total_items += len(input)
    return {
        "acc": total_train_acc / total_items,
        "loss": total_train_loss / total_items,
        **{
            f"norm/{name}": torch.norm(param.grad).item()
            for name, param in model.named_parameters()
            if param.grad is not None
        },
    }


def eval_epoch(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> dict[str, Tensor]:
    """Evaluates the binary classification model for one epoch on the provided dataloader.

    Args:
        model (nn.Module): The neural network model to evaluate.
        dataloader (DataLoader): The DataLoader providing the dataset.
        device (torch.device): The device (CPU or GPU) to perform the evaluation on.

    Returns:
        dict[str, Tensor]: A dictionary containing:
            - "inputs": List of input data.
            - "correct_outputs": List of correct labels.
            - "probabilities": List of predicted probabilities for each class.
    """
    model = model.to(device)
    inputs = []
    correct_outputs = []
    probabilities = []
    for input, labels in dataloader:
        input = input.to(device)
        labels = labels.to(device)
        output = model(input)
        inputs.append(input)
        correct_outputs.append(labels)
        probabilities.append(torch.softmax(output, dim=1))
    return {
        "inputs": torch.stack(inputs),
        "correct_outputs": torch.stack(correct_outputs),
        "probabilities": torch.stack(probabilities),
    }

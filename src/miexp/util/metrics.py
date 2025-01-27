from torch import Tensor


def binary_accuracy(logits: Tensor, labels: Tensor) -> float:
    """Computes the accuracy of a binary classification model.

    Args:
        logits (Tensor): The model's predictions (shape (batch_size, 2)).
        labels (Tensor): The true labels (shape (batch_size,))

    Returns:
        float: The accuracy of the model.
    """
    return (logits.argmax(dim=1) == labels).float().mean().item()

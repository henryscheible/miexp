import pandas as pd
import torch
from pydantic import BaseModel
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from miexp.bfuncs import MajDataset
from miexp.models.btransformer import BooleanTransformerNoEmbedding
from miexp.script_util import parse_args_from_conf


class Configuration(BaseModel):
    device: str
    lr: float
    n_heads: int
    dataset_size: int
    func_width: int
    checkpoint_save_path: str
    logs_save_path: str
    num_classifier_hidden_layers: int


def train_epoch(
    model: nn.Module,
    optimizer: Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float | None]:
    model = model.to(device)
    total_train_loss = 0
    total_train_acc = 0
    total_items = 0
    for input, labels in dataloader:
        input = input.to(device)
        labels = labels.to(device)
        output = model(input)
        loss = criterion(output, labels)
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
) -> dict[str, list[float]]:
    model = model.to(device)
    inputs = []
    correct_outputs = []
    probabilities = []
    for input, labels in dataloader:
        input = input.to(device)
        labels = labels.to(device)
        output = model(input)
        inputs += input.tolist()
        correct_outputs += labels.tolist()
        probabilities += torch.softmax(output, dim=1)[:, 1].tolist()
    return {
        "inputs": inputs,
        "correct_outputs": correct_outputs,
        "probabilities": probabilities,
    }


def main(args: Configuration) -> None:
    dataset = MajDataset(args.func_width, num_samples=args.dataset_size)

    model = BooleanTransformerNoEmbedding(
        max_seq_len=args.func_width,
        n_heads=args.n_heads,
        num_classifier_hidden_layers=args.num_classifier_hidden_layers,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    results = []
    for epoch in tqdm(range(500)):
        results.append(
            train_epoch(
                model, optimizer, dataloader, torch.device(args.device), criterion
            )
        )

    model.save_to_checkpoint(args.checkpoint_save_path)
    pd.DataFrame.from_records(results).to_csv(args.logs_save_path)


if __name__ == "__main__":
    args = parse_args_from_conf(Configuration)
    main(args)

import pandas as pd
import torch
from pydantic import BaseModel
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from miexp.bfuncs import MajDataset
from miexp.models.interptransformer import (
    SingleHeadTransformerNoEmbeddingNoMLP,
)
from miexp.script_util import parse_args_from_conf


class Configuration(BaseModel):
    device: str
    lr: float
    dataset_size: int
    func_width: int
    head_dim: int
    csv_save_path: str | None = None
    num_epochs: int
    model_save_path: str | None = None
    train_frac: float
    dataset_save_path: str | None = None


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
        probabilities += torch.softmax(output, dim=1).tolist()
    return {
        "inputs": inputs,
        "correct_outputs": correct_outputs,
        "probabilities": probabilities,
    }


def main(args: Configuration) -> None:
    torch.manual_seed(42)
    dataset = MajDataset(args.func_width, num_samples=args.dataset_size)

    train_data, eval_data = torch.utils.data.random_split(
        dataset,
        [args.train_frac, 1 - args.train_frac],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"train: {len(train_data)}, eval: {len(eval_data)}")

    model = SingleHeadTransformerNoEmbeddingNoMLP(
        vocab_size=2,
        head_dim=args.head_dim,
    ).to(torch.device(args.device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True)
    eval_dataloader = DataLoader(eval_data, batch_size=256, shuffle=False)

    results = []
    for epoch in tqdm(range(args.num_epochs)):
        cur_results = train_epoch(
            model, optimizer, train_dataloader, torch.device(args.device), criterion
        )

        eval_results = eval_epoch(model, eval_dataloader, torch.device(args.device))
        eval_acc = torch.argmax(
            torch.tensor(eval_results["probabilities"]), dim=1
        ) == torch.tensor(eval_results["correct_outputs"])
        cur_results["eval_acc"] = eval_acc.float().mean().item()
        results.append(cur_results)

    if args.dataset_save_path is not None:
        torch.save({"train": train_data, "eval": eval_data}, args.dataset_save_path)
    pd.DataFrame.from_records(results).to_csv(args.csv_save_path)
    if args.model_save_path:
        torch.save(model.state_dict(), args.model_save_path)


if __name__ == "__main__":
    args = parse_args_from_conf(Configuration)
    main(args)

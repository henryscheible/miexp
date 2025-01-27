import pandas as pd
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from miexp.bfuncs import DoubleMajDataset
from miexp.models.interptransformer import (
    SingleHeadTransformerNoEmbeddingNoMLP,
)
from miexp.script_util import parse_args_from_conf
from miexp.train.train_util import eval_epoch, train_epoch
from miexp.util.metrics import binary_accuracy


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
    low_threshold: float
    high_threshold: float


def main(args: Configuration) -> None:
    torch.manual_seed(42)
    dataset = DoubleMajDataset(
        args.func_width,
        low=args.low_threshold,
        high=args.high_threshold,
        num_samples=args.dataset_size,
    )

    low_threshold_eval_data = DoubleMajDataset(
        args.func_width, low=args.low_threshold, high=1.0, num_samples=args.dataset_size
    )
    high_threshold_eval_data = DoubleMajDataset(
        args.func_width,
        low=0.0,
        high=args.high_threshold,
        num_samples=args.dataset_size,
    )

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

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True)
    eval_dataloader = DataLoader(eval_data, batch_size=256, shuffle=False)
    low_eval_dataloader = DataLoader(
        low_threshold_eval_data,
        batch_size=256,
        shuffle=False,
    )
    high_eval_dataloader = DataLoader(
        high_threshold_eval_data,
        batch_size=256,
        shuffle=False,
    )

    results = []
    for epoch in tqdm(range(args.num_epochs)):
        cur_results = train_epoch(
            model, optimizer, train_dataloader, torch.device(args.device), criterion
        )

        eval_results = eval_epoch(model, eval_dataloader, torch.device(args.device))
        cur_results["eval_acc"] = binary_accuracy(
            eval_results["probabilities"], eval_results["correct_outputs"]
        )
        cur_results["low_eval_acc"] = binary_accuracy(
            eval_epoch(model, low_eval_dataloader, torch.device(args.device))[
                "probabilities"
            ],
            low_threshold_eval_data.labels,
        )
        cur_results["high_eval_acc"] = binary_accuracy(
            eval_epoch(model, high_eval_dataloader, torch.device(args.device))[
                "probabilities"
            ],
            high_threshold_eval_data.labels,
        )
        results.append(cur_results)

    if args.dataset_save_path is not None:
        torch.save({"train": train_data, "eval": eval_data}, args.dataset_save_path)
    pd.DataFrame.from_records(results).to_csv(args.csv_save_path)
    if args.model_save_path:
        torch.save(model.state_dict(), args.model_save_path)


if __name__ == "__main__":
    args = parse_args_from_conf(Configuration)
    main(args)

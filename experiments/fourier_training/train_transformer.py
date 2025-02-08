import pandas as pd
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from miexp.bfuncs import MultiComponentSpectrumDataset, RandMajDataset
from miexp.models.interptransformer import (
    MultiHeadTransformerPosEmb,
)
from miexp.script_util import parse_args_from_conf
from miexp.train.train_util import eval_epoch, train_epoch


class Configuration(BaseModel):
    device: str
    lr: float
    wd: float
    dataset_size: int
    func_width: int
    head_dim: int
    num_heads: int
    hidden_dim: int
    visualization_save_path: str
    csv_save_path: str
    num_epochs: int
    model_save_path: str
    train_frac: float
    dataset_save_path: str


def main(args: Configuration) -> None:
    torch.manual_seed(42)
    coeffs = torch.tensor([-0.7277, 0.5924])
    comps = torch.zeros((2, args.func_width))
    for row in comps:
        row[torch.randperm(args.func_width)[:4]] = 1
    dataset = MultiComponentSpectrumDataset(
        N=args.func_width, coeffs=coeffs, comps=comps, num_samples=args.dataset_size
    )
    dataset = RandMajDataset(N=args.func_width, frac=0.3, num_samples=args.dataset_size)
    train_data, eval_data = torch.utils.data.random_split(
        dataset,
        [args.train_frac, 1 - args.train_frac],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"train: {len(train_data)}, eval: {len(eval_data)}")

    model = MultiHeadTransformerPosEmb(
        vocab_size=2,
        max_seq_len=args.func_width + 1,
        n_heads=args.num_heads,
        head_dim=args.head_dim,
        hidden_dim=args.hidden_dim,
    ).to(torch.device(args.device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
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

    torch.save({"train": train_data, "eval": eval_data}, args.dataset_save_path)
    pd.DataFrame.from_records(results).to_csv(args.csv_save_path)
    torch.save(model.state_dict(), args.model_save_path)


if __name__ == "__main__":
    args = parse_args_from_conf(Configuration)
    main(args)

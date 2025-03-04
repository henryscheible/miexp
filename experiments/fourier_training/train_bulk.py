import uuid
from itertools import product
from typing import Any

import pandas as pd
import torch
import yaml
from pydantic import BaseModel
from tqdm import tqdm

from miexp.bfuncs import MultiComponentSpectrumDataset
from miexp.script_util import parse_args_from_conf
from miexp.train.fourier import FourierTrainingConfiguration, train_transformer_fourier


class BulkConfiguration(BaseModel):
    training_type: str = "BulkFourierTraining"
    device: str = "cpu"
    lr: float = 0.01
    wd: float = 0
    dataset_size: int = 50
    func_width: int = 10
    head_dim: int = 3
    num_heads: int = 1
    event_csv_save_path: str | None = None
    metadata_csv_save_path: str | None = None
    bulk_conf_save_path: str | None = None
    num_epochs: int = 1
    model_save_path: str | None = None
    train_frac: float = 0.5
    num_components: int = 4
    num_functions: int = 2
    comp_p: float = 0.4
    num_trials_per_function: int = 2
    init_random_seed: int = 0
    low_reject_threshold: float = 0.4
    high_reject_threshold: float = 0.6


if __name__ == "__main__":
    args = parse_args_from_conf(BulkConfiguration)
    if args.bulk_conf_save_path is not None:
        with open(args.bulk_conf_save_path, "w") as f:
            yaml.dump(args.model_dump(), f)
    torch.manual_seed(args.init_random_seed)
    events_table = pd.DataFrame(
        columns=pd.Series(
            [
                "uuid",
                "epoch",
                "loss",
                "train_acc",
                "eval_acc",
                *[f"eval_acc_{i}" for i in range(args.num_components)],
                *[f"eval_acc/head_{j}" for j in range(args.head_dim)],
                *[
                    f"eval_acc_{i}/head_{j}"
                    for i, j in product(
                        range(args.num_components), range(args.head_dim)
                    )
                ],
            ]
        )
    )
    metadata_table = pd.DataFrame(
        columns=pd.Series(
            [
                "run_uuid",
                "function_uuid",
                "coeffs",
                "comps",
                "random_seed",
                "min_loss",
                "max_eval_acc",
                "ds_fraction_positive",
            ]
        )
    )

    model_objs: dict[str, list[dict[str, Any]]] = {}

    for func_index in range(args.num_functions):
        frac = 0
        while not (args.low_reject_threshold <= frac <= args.high_reject_threshold):
            tensor_coeffs = torch.rand(args.num_components) * 2 - 1
            tensor_comps = (
                torch.rand(args.num_components, args.func_width) > args.comp_p
            ).type(torch.float)
            test_ds = MultiComponentSpectrumDataset(
                args.func_width, tensor_coeffs, tensor_comps, num_samples=100
            )
            frac = test_ds.get_percent_positive()

        func_uuid = uuid.uuid4()
        for i in range(args.num_trials_per_function):
            random_seed = torch.randint(0, 1000000, (1,)).item()
            random_seed = random_seed
            coeffs = tensor_coeffs.tolist()  # type: ignore
            comps = tensor_comps.tolist()  # type: ignore
            metadata_table.loc[len(metadata_table)] = pd.Series(
                {
                    "run_uuid": str(uuid.uuid4()),
                    "function_uuid": func_uuid,
                    "coeffs": coeffs,
                    "comps": comps,
                    "random_seed": random_seed,
                }
            )

    metadata_table = metadata_table.set_index("run_uuid")

    for i in tqdm(range(len(metadata_table))):
        metadata = metadata_table.iloc[i]
        run_uuid: str = str(metadata_table.index[i])  # type: ignore
        output = train_transformer_fourier(
            FourierTrainingConfiguration(
                device=args.device,
                lr=args.lr,
                wd=args.wd,
                dataset_size=args.dataset_size,
                func_width=args.func_width,
                head_dim=args.head_dim,
                num_heads=args.num_heads,
                num_epochs=args.num_epochs,
                train_frac=args.train_frac,
                random_seed=metadata["random_seed"],  # type: ignore
                coeffs=metadata["coeffs"],  # type: ignore
                comps=metadata["comps"],  # type: ignore
            )
        )
        small_events_table = output.events_log
        model_obj = output.params_dict
        min_loss = small_events_table["loss"].min()
        max_eval_acc = small_events_table["eval_acc"].max()
        metadata_table.loc[run_uuid, "min_loss"] = min_loss
        metadata_table.loc[run_uuid, "max_eval_acc"] = max_eval_acc
        metadata_table.loc[run_uuid, "ds_fraction_positive"] = output.ds_positive_frac
        small_events_table["uuid"] = run_uuid

        small_events_table = small_events_table.reindex(
            columns=events_table.columns, fill_value=None
        )
        if len(events_table) > 0:
            events_table = pd.concat(
                [events_table, small_events_table], ignore_index=True
            )
        else:
            events_table = small_events_table
        model_objs[run_uuid] = model_obj
        if args.model_save_path is not None:
            torch.save(model_objs, args.model_save_path)
        if args.event_csv_save_path is not None:
            events_table.to_csv(args.event_csv_save_path)
        if args.metadata_csv_save_path is not None:
            metadata_table.to_csv(args.metadata_csv_save_path)

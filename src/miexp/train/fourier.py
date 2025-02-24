from typing import Any

import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from miexp.bfuncs import MultiComponentSpectrumDataset
from miexp.models.interptransformer import SingleHeadTransformerOneHotPositionalNoMLP
from miexp.train.train_util import eval_epoch, train_epoch
from miexp.util.lambda_model import LambdaModel
from miexp.util.metrics import binary_accuracy


class FourierTrainingConfiguration(BaseModel):
    """FourierTrainingConfiguration is a configuration class for training a model using Fourier transformations.

    Attributes:
        device (str): The device to be used for training (e.g., 'cpu' or 'cuda').
        lr (float): Learning rate for the optimizer.
        wd (float): Weight decay for the optimizer.
        dataset_size (int): The size of the dataset to be used for training.
        func_width (int): The width of the function to be used in the model (length of bitstring).
        head_dim (int): The dimension of each attention head.
        num_heads (int): The number of attention heads.
        num_epochs (int): The number of epochs for training.
        dataset_save_path (str | None): The path to save the dataset, if any.
        csv_save_path (str | None): The path to save the CSV file, if any.
        model_save_path (str | None): The path to save the trained model, if any.
        train_frac (float): The fraction of the dataset to be used for training.
        random_seed (int): The random seed for reproducibility.
        coeffs (list[float] | None): The list of coefficients for the Fourier components, if any.
        comps (list[list[int]] | None): The list of component masks for each fourier component, if any.
    """

    device: str = "cpu"
    lr: float = 0.01
    wd: float = 0
    dataset_size: int = 1000
    func_width: int = 10
    head_dim: int = 3
    num_heads: int = 1
    num_epochs: int = 10
    dataset_save_path: str | None = None
    csv_save_path: str | None = None
    model_save_path: str | None = None
    train_frac: float = 0.5
    random_seed: int = 0
    coeffs: list[float] | None = None
    comps: list[list[int]] | None = None


class OutputData(BaseModel):
    """OutputData contains the ouputs to train_transformer_fourier.

    Attributes:
        model_config (ConfigDict): Configuration dictionary allowing arbitrary types.
        events_log (pd.DataFrame): A pandas DataFrame containing the events log.
        model_dict (dict): A dictionary containing model-related data.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    events_log: pd.DataFrame
    params_dict: list[dict[str, Any]]
    ds_positive_frac: float


def train_transformer_fourier(args: FourierTrainingConfiguration) -> OutputData:
    """Trains a transformer model on a boolean function with multiple fourier components.

    Args:
        args (FourierTrainingConfiguration): Configuration parameters for training.

    Returns:
        OutputData: Contains the events log and model dictionary.

    The function performs the following steps:
    1. Sets the random seed for reproducibility.
    2. Initializes coefficients and components based on provided arguments or defaults.
    3. Creates a dataset of multi-component spectra.
    4. Splits the dataset into training and evaluation sets.
    5. Initializes the transformer model and optimizer.
    6. Sets up the loss criterion and data loaders.
    7. Creates datasets for single component evaluations.
    8. Trains the model for a specified number of epochs, evaluating and logging results.
    9. Saves the dataset, results, and model if specified in the arguments.

    The training process includes:
    - Training the model on the training dataset.
    - Evaluating the model on the evaluation dataset.
    - Evaluating the model on single component datasets.
    - Adjusting the learning rate based on evaluation loss.

    The results are saved to CSV and model checkpoints if paths are provided.
    """
    torch.manual_seed(args.random_seed)
    if args.coeffs is not None:
        coeffs = torch.tensor(args.coeffs, dtype=torch.float)
    else:
        coeffs = torch.rand(3)
    if args.comps is not None:
        comps = torch.tensor(args.comps, dtype=torch.float)
    else:
        comps = torch.zeros(3, args.func_width)
        for i in range(3):
            comps[i, torch.randperm(args.func_width)[:4]] = 1
    dataset = MultiComponentSpectrumDataset(
        N=args.func_width, coeffs=coeffs, comps=comps, num_samples=args.dataset_size
    )

    train_data, eval_data = torch.utils.data.random_split(
        dataset,
        [args.train_frac, 1 - args.train_frac],
        generator=torch.Generator().manual_seed(42),
    )

    model = SingleHeadTransformerOneHotPositionalNoMLP(
        vocab_size=2,
        max_seq_len=args.func_width + 1,
        head_dim=args.head_dim,
    ).to(torch.device(args.device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()
    train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True)
    eval_dataloader = DataLoader(eval_data, batch_size=256, shuffle=False)

    single_comp_datasets = {
        coef.item(): MultiComponentSpectrumDataset(
            N=args.func_width,
            coeffs=coef.unsqueeze(0),
            comps=comp.unsqueeze(0),
            num_samples=args.dataset_size,
        )
        for comp, coef in zip(comps, coeffs)
    }

    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=10)

    results = []
    model_dicts = []
    for epoch in range(args.num_epochs):
        cur_results: dict[str, float | None] = {}
        cur_results.update(
            train_epoch(
                model, optimizer, train_dataloader, torch.device(args.device), criterion
            )
        )
        eval_results = eval_epoch(model, eval_dataloader, torch.device(args.device))
        eval_acc = (
            torch.argmax(eval_results["probabilities"], dim=1)
            == eval_results["correct_outputs"]
        )
        cur_results["eval_acc"] = eval_acc.float().mean().item()
        for i, (coef, ds) in enumerate(single_comp_datasets.items()):
            single_eval_results = eval_epoch(
                model,
                DataLoader(ds, batch_size=256, shuffle=False),
                torch.device(args.device),
            )
            cur_results[f"eval_acc_{i}"] = binary_accuracy(
                single_eval_results["probabilities"],
                single_eval_results["correct_outputs"],
            )
        head_total_mask = torch.eye(
            args.head_dim, dtype=torch.int, device=torch.device(args.device)
        )
        for j in range(args.head_dim):
            single_head_model = LambdaModel(
                lambda input: model(input, head_dim_mask=head_total_mask[j, :])
            )
            eval_results = eval_epoch(
                single_head_model, eval_dataloader, torch.device(args.device)
            )
            eval_acc = (
                torch.argmax(eval_results["probabilities"], dim=1)
                == eval_results["correct_outputs"]
            )
            cur_results[f"eval_acc/head_{j}"] = eval_acc.float().mean().item()
            for i, (coef, ds) in enumerate(single_comp_datasets.items()):
                single_eval_results = eval_epoch(
                    single_head_model,
                    DataLoader(ds, batch_size=256, shuffle=False),
                    torch.device(args.device),
                )
                cur_results[f"eval_acc_{i}/head_{j}"] = binary_accuracy(
                    single_eval_results["probabilities"],
                    single_eval_results["correct_outputs"],
                )
        cur_results["lr"] = scheduler.get_last_lr()[0]
        results.append(cur_results)
        scheduler.step(cur_results["loss"])  # type: ignore
        model_dicts.append(model.get_save_dict())

    if args.dataset_save_path is not None:
        torch.save({"train": train_data, "eval": eval_data}, args.dataset_save_path)
    if args.csv_save_path is not None:
        pd.DataFrame.from_records(results).to_csv(args.csv_save_path)
    if args.model_save_path is not None:
        model.save_to_checkpoint(args.model_save_path)

    return OutputData(
        events_log=pd.DataFrame.from_records(results).reset_index(names="epoch"),
        params_dict=model_dicts,
        ds_positive_frac=dataset.get_percent_positive(),
    )

import itertools
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import yaml

from src.constants import RESULTS_DIR, SUMMARY_COLUMNS, SUMMARY_PATH
from src.data_sampling.data_sampler import DataSampler
from src.logging import logger

# from src.models.bnn import BNN
from src.models.der import DER
from src.models.mcdropout_bnn import MCDropoutBNN
from src.models.sder import SDER


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def save_results(
    preds_all,
    epistemic_all,
    aleatoric_all,
    aleatoric_true,
    X_test,
    X_train,
    y_test,
    y_train,
    train_times,
    infer_times,
    job,
    base_dir=RESULTS_DIR,
):
    model_name = job["model_name"]
    results_dir = generate_result_path(base_dir, model_name, job)
    os.makedirs(results_dir, exist_ok=True)

    # Save predictions and uncertainties
    np.save(results_dir / "y_pred_all.npy", preds_all)
    np.save(results_dir / "epistemic_all.npy", epistemic_all)
    np.save(results_dir / "aleatoric_all.npy", aleatoric_all)
    np.save(results_dir / "aleatoric_true.npy", aleatoric_true)

    # Save train and test data
    np.save(results_dir / "X_test.npy", X_test)
    np.save(results_dir / "X_train.npy", X_train)
    np.save(results_dir / "y_true.npy", y_test)
    np.save(results_dir / "y_train.npy", y_train)

    # Save times
    np.save(results_dir / "train_times.npy", np.array(train_times))
    np.save(results_dir / "infer_times.npy", np.array(infer_times))

    # Save config
    with open(results_dir / "config.json", "w") as f:
        json.dump(job, f, indent=2)

    logger.info(f"Saved results to: {results_dir}")


def build_model(model_name, model_params):
    # if model_name == "bnn":
    #     return BNN(
    #         **model_params,
    #     )
    if model_name == "mcdropout_bnn":
        return MCDropoutBNN(
            **model_params,
        )
    elif model_name == "der":
        return DER(
            **model_params,
        )
    elif model_name == "sder":
        return SDER(
            **model_params,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# def sample_data(config):
#     data_cfg = config["data"]
#     train_data_cfg = sampling_cfg["train"]
#     test_data_cfg = sampling_cfg["test"]

#     # Choose sampler
#     sampler_type = sampling_cfg["sampler"]

#     if sampler_type == "single":
#         sampler = CubicSingleFeatureSampler(seed=sampling_cfg["seed"])
#         train_df = sampler.sample_train_data(
#             n_unique=train_data_cfg["n_unique"],
#             n_repeats=train_data_cfg["n_repeats"],
#             min_val=train_data_cfg["min_val"],
#             max_val=train_data_cfg["max_val"],
#             n_sparse_center=train_data_cfg["n_sparse_center"],
#             use_sparse_center=train_data_cfg["use_sparse_center"],
#         )
#         test_df = sampler.sample_test_data(
#             n_points=test_data_cfg["n_points"],
#             min_val=test_data_cfg["min_val"],
#             max_val=test_data_cfg["max_val"],
#         )

#     elif sampler_type == "x3":
#         sampler = x3Sampler(
#             n_samples=sampling_cfg["n_samples"], seed=sampling_cfg["seed"]
#         )
#         x_min = sampling_cfg["x_min"]
#         x_max = sampling_cfg["x_max"]
#         train_df = sampler.sample_train_data(x_min=x_min, x_max=x_max, train=True)
#         test_df = sampler.sample_test_data(x_min=-7, x_max=7, train=False)
#         test_df["aleatoric_true"] = 3.0

#     elif sampler_type == "linear":
#         sampler = LinearMultiFeatureSampler(
#             n_features=sampling_cfg["n_features"], seed=sampling_cfg["seed"]
#         )
#         train_df = sampler.sample_train_data(
#             n_unique=train_data_cfg["n_unique"],
#             n_repeats=train_data_cfg["n_repeats"],
#             min_val=train_data_cfg["min_val"],
#             max_val=train_data_cfg["max_val"],
#             sparse_center_frac=train_data_cfg["sparse_center_frac"],
#         )
#         test_df = sampler.sample_test_grid(
#             grid_length=test_data_cfg["grid_length"],
#             min_val=test_data_cfg["min_val"],
#             max_val=test_data_cfg["max_val"],
#         )
#     else:
#         raise ValueError(f"Unknown sampler type: {sampler_type}")

#     return train_df, test_df


def create_train_test_data(
    job: Dict[str, Union[str, int, float, list]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Baut Trainings- und Test-DataFrames aus dem gegebenen Konfig-Dict.

    Erwartete Keys im config (entspricht deiner Klasse):
      - seed: int
      - function: str  (Key in FUNCTIONS)
      - noise: str     (Key in NOISES)
      - train_interval: [a,b] oder [[a,b], [c,d], ...]
      - train_n_instances: int
      - train_n_repeats: int
      - test_interval: [a,b]
      - test_grid_length: int

    Rückgabe:
      (df_train, df_test) : Tuple[pd.DataFrame, pd.DataFrame]
        Spalten: x1..xd, y, y_clean, noise
    """
    sampler = DataSampler(job)

    # --- Train ---
    train = sampler.sample_train_data()
    d_train = train["X"].shape[1]
    x_cols_train = [f"x{i + 1}" for i in range(d_train)]
    df_train = pd.DataFrame(train["X"], columns=x_cols_train)
    df_train["y"] = train["y"]
    df_train["y_clean"] = train["y_clean"]
    df_train["noise"] = train["noise"]

    # --- Test ---
    test = sampler.sample_test_data()
    d_test = test["X"].shape[1]
    x_cols_test = [f"x{i + 1}" for i in range(d_test)]
    df_test = pd.DataFrame(test["X"], columns=x_cols_test)
    df_test["y"] = test["y"]
    df_test["y_clean"] = test["y_clean"]
    df_test["noise"] = test["noise"]

    return df_train, df_test


def load_yaml(path: str) -> dict:
    """Load a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_benchmark_jobs(config_path: str) -> List[Dict]:
    """
    Generate all benchmark jobs (Cartesian product) for the given config.
    """
    # Experiment configs
    experiment_cfg = load_yaml(config_path)["experiment"]
    experiment_name = experiment_cfg["name"]
    model_runs = experiment_cfg["model_runs"]
    seed = experiment_cfg["seed"]

    # Data configs
    data_cfg = experiment_cfg["data"]
    functions = data_cfg["function"]

    # Train data configs
    train_cfg = data_cfg["train_data"]
    train_intervals = train_cfg["interval"]
    train_instances = train_cfg["n_instances"]
    train_repeats = train_cfg["n_repeats"]

    # Test data configs
    test_cfg = data_cfg["test_data"]
    test_intervals = test_cfg["interval"]
    test_grid_lengths = test_cfg["grid_length"]

    # Models config: list of {model_name: {params}}
    model_cfg = experiment_cfg["models"]
    models_expanded = []
    for item in model_cfg:
        ((name, params),) = item.items()
        models_expanded.append((name, dict(params)))

    all_jobs = []

    for fn, noise, tr_int, tr_n, tr_r, te_int, te_grid, (
        model_name,
        model_params,
    ) in itertools.product(
        functions,
        train_intervals,
        train_instances,
        train_repeats,
        test_intervals,
        test_grid_lengths,
        models_expanded,
    ):
        job = {
            "experiment_name": experiment_name,
            "model_runs": model_runs,
            "random_seed": seed,
            "function": fn,
            "train_interval": tr_int,
            "train_instances": tr_n,
            "train_repeats": tr_r,
            "test_interval": te_int,
            "test_grid_length": te_grid,
            "model_name": model_name,
            "model_params": model_params,
        }
        all_jobs.append(job)

    # Deduplicate
    seen = set()
    unique_jobs: List[Dict] = []
    for job in all_jobs:
        key = json.dumps(job, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_jobs.append(job)

    logger.info(
        f"Generated {len(unique_jobs)} benchmark jobs for experiment {experiment_name}."
    )
    return unique_jobs


def append_summary(job: Dict, result_dir: str) -> None:
    """
    Append a single row to the global benchmark_summary.csv.

    Notes:
        - The function writes without header (append mode).
        - result_folder is stored relative to the results/ root.
        - A completion timestamp is added.

    Args:
        job (Dict): Full job dict (all parameters).
        result_dir (str): Absolute path to the result folder (will be converted to relative).
    """
    row = {col: job.get(col, None) for col in SUMMARY_COLUMNS}
    relative_result_dir = result_dir.split("results/")[1]

    row["result_folder"] = relative_result_dir
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_row = pd.DataFrame([row])
    df_row.to_csv(SUMMARY_PATH, mode="a", header=False, index=False)


def _as_str(v):
    # stringify complex types so comparison is scalar-vs-scalar
    if isinstance(v, (list, dict, tuple)):
        return json.dumps(v, sort_keys=True)
    if v is None:
        return "None"
    return str(v)


def job_already_done(job: Dict[str, Any]) -> bool:
    """
    A job is 'done' if any CSV row matches ALL job columns exactly.
    Complex fields are compared as JSON strings.
    """
    if not os.path.exists(SUMMARY_PATH):
        return False
    if os.path.getsize(SUMMARY_PATH) == 0:
        return False

    # read as strings to avoid pandas turning lists into objects/arrays
    df = pd.read_csv(SUMMARY_PATH, dtype=str)

    if df.empty:
        return False

    # ensure schema matches expected columns
    for col in SUMMARY_COLUMNS:
        if col not in df.columns:
            return False

    # build row-wise mask
    mask = pd.Series(True, index=df.index)
    for col in SUMMARY_COLUMNS:
        val = _as_str(job[col])
        # df is dtype=str, NaNs appear as "nan"; we treat None == "None" only
        mask &= df[col] == val

    return bool(mask.any())


def create_full_job_name(job: Dict) -> str:
    """
    Compact string for job description in logs/UIs
    """
    return (
        f"Func:{job['function']} | "
        f"Noise:{job['noise']} | "
        f"Train: interval: {job['train_interval']}, instances: {job['train_instances']}, repeats: {job['train_repeats']} | "
        f"Test: interval: {job['test_interval']}, grid_length: {job['test_grid_length']} | "
        f"Seed:{job['random_seed']} | "
        f"Model:{job['model_name']}"
    )


def generate_result_path(base_dir: str, model_name: str, job: Dict) -> str:
    """
    Build a readable, short path. Example:
      <base>/<model_name>/seed42_fn-nl2_nz-nl2_tr-m2_2_te-m2_2_trn1000x1_ten10000x1_model-bnn/250920_1342_data
    """
    parts: List[str] = []

    parts.append(f"seed-{job['random_seed']}")
    parts.append(f"fn-{job['function']}")
    parts.append(f"nz-{job['noise']}")
    parts.append(f"tri-{job['train_interval']}")
    parts.append(f"tei-{job['test_interval']}")
    parts.append(f"trn-{job['train_instances']}x{job['train_repeats']}")
    parts.append(f"grid-{job['test_grid_length']}")
    parts.append(f"model-{job['model_name']}")

    setting = "_".join(parts)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    return Path(base_dir) / model_name / setting / f"{timestamp}"

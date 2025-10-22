import itertools
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from copy import deepcopy

from src.constants import RESULTS_DIR, SUMMARY_COLUMNS, SUMMARY_PATH
from src.utils.utils_logging import logger


def set_global_seed(seed: int):
    """Set seed for all used libraries to get reproducible results."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def load_yaml(path: str) -> dict:
    """Load a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_benchmark_jobs(config_path: str) -> List[Dict]:
    """Generate all benchmark jobs (Cartesian product) for the given config."""
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
    test_add_points = test_cfg["add_points"]

    # Models config: list of {model_name: {params}}
    model_cfg = experiment_cfg["models"]
    # --- Base-Params extrahieren ---
    base_params = {}
    for model in model_cfg:
        if "base_model" in model:
            base_params = dict(model["base_model"])
            break

    models_expanded = []
    for model in model_cfg:
        if "base_model" in model:
            continue
        name, params = next(iter(model.items()))
        merged = deepcopy(base_params)
        merged.update(params)
        models_expanded.append((name, merged))

    all_jobs = []

    for fn, tr_int, tr_n, tr_r, te_int, te_grid, te_points, (
        model_name,
        model_params,
    ) in itertools.product(
        functions,
        train_intervals,
        train_instances,
        train_repeats,
        test_intervals,
        test_grid_lengths,
        test_add_points,
        models_expanded,
    ):
        job = {
            "experiment_name": experiment_name,
            "model_runs": model_runs,
            "seed": seed,
            "function": fn,
            "train_interval": tr_int,
            "train_instances": tr_n,
            "train_repeats": tr_r,
            "test_interval": te_int,
            "test_grid_length": te_grid,
            "test_add_points": te_points,
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
    row = {}
    for col in SUMMARY_COLUMNS:
        if col == "function":
            row[col] = job["function"][0] if "function" in job else None
        elif col == "noise":
            row[col] = job["function"][1] if "function" in job else None
        else:
            row[col] = job.get(col, None)

    relative_result_dir = result_dir.relative_to(RESULTS_DIR)

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
    Special case:
      - SUMMARY_COLUMNS enthält 'function' und 'noise'
      - job['function'] ist eine Liste [function_value, noise_value]
    Complex fields are compared as JSON strings via _as_str.
    """
    if not SUMMARY_PATH.exists():
        return False
    if SUMMARY_PATH.stat().st_size == 0:
        return False

    df = pd.read_csv(SUMMARY_PATH, dtype=str)

    if df.empty:
        return False

    for col in SUMMARY_COLUMNS:
        if col not in df.columns:
            return False

    def _job_value_for(col: str):
        if col == "function":
            return (job.get("function") or [None, None])[0]
        if col == "noise":
            return (job.get("function") or [None, None])[1]
        return job.get(col, None)

    wanted = {col: _as_str(_job_value_for(col)) for col in SUMMARY_COLUMNS}

    mask = pd.Series(True, index=df.index)
    for col, val in wanted.items():
        mask &= df[col] == val

    return bool(mask.any())


def create_full_job_name(job: Dict) -> str:
    """Compact string for job description in logs/UIs."""
    return (
        f"Func:{job['function'][0]} | "
        f"Noise:{job['function'][1]} | "
        f"Train: interval: {job['train_interval']}, instances: {job['train_instances']}, repeats: {job['train_repeats']} | "
        f"Test: interval: {job['test_interval']}, grid_length: {job['test_grid_length']} | "
        f"Seed:{job['seed']} | "
        f"Model:{job['model_name']}"
    )


def generate_result_path(base_dir: str, job: Dict) -> str:
    """
    Build a readable, short path. Example:
      <base>/<model_name>/seed42_fn-nl2_nz-nl2_tr-m2_2_te-m2_2_trn1000x1_ten10000x1_model-bnn/250920_1342_data
    """

    def _extract_number_from_function(job, prefix="f"):
        """Extract the *last* number after a certain prefix (e.g. "f", "sigma")."""
        fn_field = job.get("function", None)
        if fn_field is None:
            return "0"

        def flatten(x):
            if isinstance(x, (list, tuple)):
                for xi in x:
                    yield from flatten(xi)
            else:
                yield str(x)

        pattern = rf"{prefix}[_a-z]*?(\d+)"
        last_match = None

        for item in flatten(fn_field):
            matches = list(re.finditer(pattern, item))
            if matches:
                # Take the *last* match in this item
                last_match = matches[-1].group(1)

        return last_match if last_match is not None else "0"

    parts = []

    fn_match = _extract_number_from_function(job, prefix="f")
    nz_match = _extract_number_from_function(job, prefix="sigma")

    parts.append(f"seed-{job['seed']}")
    parts.append(f"fn-{fn_match}")
    parts.append(f"nz-{nz_match}")
    parts.append(f"tri-{job['train_interval']}")
    parts.append(f"tei-{job['test_interval']}")
    parts.append(f"trn-{job['train_instances']}x{job['train_repeats']}")
    parts.append(f"grid-{job['test_grid_length']}")
    parts.append(f"model-{job['model_name']}")

    setting = "_".join(parts)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    return (
        Path(base_dir)
        / job["experiment_name"]
        / job["model_name"]
        / setting
        / f"{timestamp}"
    )


def save_results(
    epoch_losses_all,
    preds_all,
    epistemic_all,
    aleatoric_all,
    nll_all,
    cal_score_all,
    aleatoric_true,
    X_test,
    X_train,
    y_test,
    y_train,
    train_time_all,
    infer_time_all,
    job,
    results_dir,
):
    """Save all results, data, times, config to results_dir."""
    os.makedirs(results_dir, exist_ok=True)

    # Save losses
    np.save(results_dir / "epoch_losses_all.npy", epoch_losses_all)

    # Save predictions and uncertainties
    np.save(results_dir / "y_pred_all.npy", preds_all)
    np.save(results_dir / "epistemic_all.npy", epistemic_all)
    np.save(results_dir / "aleatoric_all.npy", aleatoric_all)
    np.save(results_dir / "aleatoric_true.npy", aleatoric_true)

    # Save eval metrics
    np.save(results_dir / "nll_all.npy", np.array(nll_all))
    np.save(results_dir / "cal_score_all.npy", np.array(cal_score_all))

    # Save train and test data
    np.save(results_dir / "X_test.npy", X_test)
    np.save(results_dir / "X_train.npy", X_train)
    np.save(results_dir / "y_true.npy", y_test)
    np.save(results_dir / "y_train.npy", y_train)

    # Save times
    np.save(results_dir / "train_times.npy", np.array(train_time_all))
    np.save(results_dir / "infer_times.npy", np.array(infer_time_all))

    # Save config
    with open(results_dir / "config.json", "w") as f:
        json.dump(job, f, indent=2)

    logger.info(f"Saved results to: {results_dir}")

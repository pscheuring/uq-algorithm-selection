import itertools
import json
import os
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.constants import RESULTS_DIR, SUMMARY_COLUMNS, SUMMARY_PATH
from src.utils.utils_logging import logger

Job = dict[str, Any]


def set_global_seed(seed: int) -> None:
    """Set seed for all used libraries to get reproducible results."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file."""
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)


def generate_benchmark_jobs(config_path: str | Path) -> list[Job]:
    """Generate all benchmark jobs (Cartesian product) for the given config."""
    # Experiment configs
    experiment_cfg = load_yaml(config_path)["experiment"]
    experiment_name: str = experiment_cfg["name"]
    model_runs: int = experiment_cfg["model_runs"]
    seed: int = experiment_cfg["seed"]

    # Data configs
    data_cfg = experiment_cfg["data"]
    functions: Sequence[Any] = data_cfg["function"]

    # Train data configs
    train_cfg = data_cfg["train_data"]
    train_intervals: Sequence[Any] = train_cfg["interval"]
    train_instances: Sequence[int] = train_cfg["n_instances"]
    train_repeats: Sequence[int] = train_cfg["n_repeats"]

    # Test data configs
    test_cfg = data_cfg["test_data"]
    test_intervals: Sequence[Any] = test_cfg["interval"]
    test_grid_lengths: Sequence[int] = test_cfg["grid_length"]
    test_add_points: Sequence[Any] = test_cfg["add_points"]

    # Models config: list of {model_name: {params}}
    model_cfg: list[dict[str, Any]] = experiment_cfg["models"]

    # Extract base params
    base_params: dict[str, Any] = {}
    for model in model_cfg:
        if "base_model" in model:
            base_params = dict(model["base_model"])
            break

    models_expanded: list[tuple[str, dict[str, Any]]] = []
    for model in model_cfg:
        if "base_model" in model:
            continue
        name, params = next(iter(model.items()))
        merged = deepcopy(base_params)
        merged.update(params)
        models_expanded.append((name, merged))

    all_jobs: list[Job] = []

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
        job: Job = {
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
    seen: set[str] = set()
    unique_jobs: list[Job] = []
    for job in all_jobs:
        key = json.dumps(job, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique_jobs.append(job)

    logger.info(
        "Generated %d benchmark jobs for experiment %s.",
        len(unique_jobs),
        experiment_name,
    )
    return unique_jobs


def _split_functions_and_noise(fn_spec: Any) -> tuple[list[Any], list[Any]]:
    """
    Split job['function'] into two lists:
      - list of f_* names
      - list of sigma_* names

    Supported shapes:
      - [f, sigma]
      - [[f1, s1], [f2, s2], ...]
    """
    if fn_spec is None:
        return [], []

    # Multi-target: list of [f, sigma] pairs
    if (
        isinstance(fn_spec, (list, tuple))
        and len(fn_spec) > 0
        and isinstance(fn_spec[0], (list, tuple))
    ):
        funcs = [pair[0] for pair in fn_spec]
        noises = [pair[1] for pair in fn_spec]
        return funcs, noises

    # Single-target: [f, sigma]
    if isinstance(fn_spec, (list, tuple)) and len(fn_spec) == 2:
        return [fn_spec[0]], [fn_spec[1]]

    # Fallback: just one function without noise
    return [fn_spec], [None]


def append_summary(job: Mapping[str, Any], result_dir: Path) -> None:
    """
    Append a single row to the global benchmark_summary.csv.

    Notes:
        - The function writes without header (append mode).
        - result_folder is stored relative to the results/ root.
        - A completion timestamp is added.

    Args:
        job: Full job dict (all parameters).
        result_dir: Absolute path to the result folder (will be converted to relative).
    """
    fn_funcs, fn_noises = _split_functions_and_noise(job.get("function"))

    row: dict[str, Any] = {}
    for col in SUMMARY_COLUMNS:
        if col == "function":
            # Single-target: store single function name as before
            if len(fn_funcs) == 1:
                row[col] = fn_funcs[0]
            else:
                # Multi-target: store all f_* names as JSON list
                row[col] = json.dumps(fn_funcs)
        elif col == "noise":
            if len(fn_noises) == 1:
                row[col] = fn_noises[0]
            else:
                # Multi-target: store all sigma_* names as JSON list
                row[col] = json.dumps(fn_noises)
        else:
            row[col] = job.get(col, None)

    relative_result_dir = result_dir.relative_to(RESULTS_DIR)

    row["result_folder"] = relative_result_dir
    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df_row = pd.DataFrame([row])
    df_row.to_csv(SUMMARY_PATH, mode="a", header=False, index=False)


def _as_str(v: Any) -> str:
    """Stringify complex types so comparison is scalar-vs-scalar."""
    if isinstance(v, (list, dict, tuple)):
        return json.dumps(v, sort_keys=True)
    if v is None:
        return "None"
    return str(v)


def job_already_done(job: Mapping[str, Any]) -> bool:
    """
    A job is 'done' if any CSV row matches ALL job columns exactly.

    Special case:
      - SUMMARY_COLUMNS contains 'function' and 'noise'
      - job['function'] can be either
            [f, sigma] or [[f1, s1], [f2, s2], ...]
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

    fn_funcs, fn_noises = _split_functions_and_noise(job.get("function"))

    def _job_value_for(col: str) -> Any:
        if col == "function":
            # Single-target: single name, multi-target: list of f_* names
            return fn_funcs[0] if len(fn_funcs) == 1 else fn_funcs
        if col == "noise":
            return fn_noises[0] if len(fn_noises) == 1 else fn_noises
        return job.get(col, None)

    wanted: dict[str, str] = {
        col: _as_str(_job_value_for(col)) for col in SUMMARY_COLUMNS
    }

    mask = pd.Series(True, index=df.index)
    for col, val in wanted.items():
        mask &= df[col] == val

    return bool(mask.any())


def create_full_job_name(job: Mapping[str, Any]) -> str:
    """Compact string for job description in logs/UIs."""
    fn_funcs, fn_noises = _split_functions_and_noise(job.get("function"))

    if len(fn_funcs) == 1:
        func_str = fn_funcs[0]
    else:
        func_str = f"{fn_funcs[0]}+{len(fn_funcs) - 1} more"

    if len(fn_noises) == 1:
        noise_str = fn_noises[0]
    else:
        noise_str = f"{fn_noises[0]}+{len(fn_noises) - 1} more"

    return (
        f"Func:{func_str} | "
        f"Noise:{noise_str} | "
        f"Train: interval: {job['train_interval']}, "
        f"instances: {job['train_instances']}, "
        f"repeats: {job['train_repeats']} | "
        f"Test: interval: {job['test_interval']}, "
        f"grid_length: {job['test_grid_length']} | "
        f"Seed:{job['seed']} | "
        f"Model:{job['model_name']}"
    )


def generate_result_path(base_dir: str | Path, job: Mapping[str, Any]) -> Path:
    """
    Build a readable, short path. Example:
      <base>/<model_name>/seed42_fn-nl2_nz-nl2_tr-m2_2_te-m2_2_trn1000x1_ten10000x1_model-bnn/250920_1342
    """

    def _extract_number_from_function(job_mapping: Mapping[str, Any]) -> str:
        """Extract the number directly before `_feat` from any string in `function`."""
        fn_field = job_mapping.get("function", None)
        if fn_field is None:
            return "0"

        def flatten(x: Any) -> Iterable[str]:
            if isinstance(x, (list, tuple)):
                for xi in x:
                    yield from flatten(xi)
            else:
                yield str(x)

        pattern = r"(\d+)_feat"
        last_match: str | None = None

        for item in flatten(fn_field):
            matches = list(re.finditer(pattern, item))
            if matches:
                last_match = matches[-1].group(1)

        return last_match if last_match is not None else "0"

    base_dir = Path(base_dir)

    parts: list[str] = []

    fn_match = _extract_number_from_function(job)
    nz_match = _extract_number_from_function(job)

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
        base_dir / job["experiment_name"] / job["model_name"] / setting / f"{timestamp}"
    )


def save_results(
    epoch_losses_all: np.ndarray,
    preds_all: np.ndarray,
    epistemic_all: np.ndarray,
    aleatoric_all: np.ndarray,
    nll_ood_all: Sequence[float],
    nll_id_all: Sequence[float],
    aleatoric_true: np.ndarray,
    X_test: np.ndarray,
    X_train: np.ndarray,
    y_test: np.ndarray,
    y_clean: np.ndarray,
    y_train: np.ndarray,
    train_time_all: Sequence[float],
    infer_time_all: Sequence[float],
    job: Mapping[str, Any],
    results_dir: Path,
) -> None:
    """Save all results, data, times, and config to results_dir."""
    os.makedirs(results_dir, exist_ok=True)

    # Save losses
    np.save(results_dir / "epoch_losses_all.npy", epoch_losses_all)

    # Save predictions and uncertainties
    np.save(results_dir / "y_pred_all.npy", preds_all)
    np.save(results_dir / "epistemic_all.npy", epistemic_all)
    np.save(results_dir / "aleatoric_all.npy", aleatoric_all)
    np.save(results_dir / "aleatoric_true.npy", aleatoric_true)

    # Save eval metrics
    np.save(results_dir / "nll_ood_all.npy", np.array(nll_ood_all))
    np.save(results_dir / "nll_id_all.npy", np.array(nll_id_all))

    # Save train and test data
    np.save(results_dir / "X_test.npy", X_test)
    np.save(results_dir / "X_train.npy", X_train)
    np.save(results_dir / "y_test.npy", y_test)
    np.save(results_dir / "y_clean.npy", y_clean)
    np.save(results_dir / "y_train.npy", y_train)

    # Save times
    np.save(results_dir / "train_times.npy", np.array(train_time_all))
    np.save(results_dir / "infer_times.npy", np.array(infer_time_all))

    # Save config
    with (results_dir / "config.json").open("w") as f:
        json.dump(dict(job), f, indent=2)

    logger.info("Saved results to: %s", results_dir)

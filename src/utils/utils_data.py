import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.data_sampler import DataSampler
from src.utils.utils_logging import logger


def create_train_test_data(
    job: dict[str, str | int | float | list[Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Create training and test dataframes, auto-detecting single- vs multi-output.

    Detection rules:
      - Single-output: job["function"] is a pair [f_name, sigma_name].
      - Multi-output:  job["function"] is a list/tuple of pairs:
                       [[f1, s1], [f2, s2], ...].

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, int]:
            - df_train: columns x1..xD, y (or tuple of y's), sigma
            - df_test:  columns x1..xD, y (or tuple of y's), sigma
            - n_features: input dimensionality D
    """
    func = job.get("function")
    is_multi = (
        isinstance(func, (list, tuple))
        and len(func) > 0
        and isinstance(func[0], (list, tuple))
    )

    if is_multi:
        logger.debug("Using multi-output function.")
        return create_train_test_data_multi_output(job)
    else:
        logger.debug("Using single-output function.")
        return create_train_test_data_single_output(job)


def create_train_test_data_single_output(
    job: dict[str, str | int | float | list[Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Build single-output training and test DataFrames from a job config.

    Expected job keys (forwarded to DataSampler):
        seed (int): Random seed.
        function (list[str, str]): [function_name, noise_name].
        noise (str): Noise key.
        train_interval (list): [a, b] or [[a, b], [c, d], ...].
        train_n_instances (int): Number of training instances.
        train_n_repeats (int): Repeats per instance.
        test_interval (list): [a, b].
        test_grid_length (int): Number of grid points per axis for the test set.
        (optional) add_test_points (list[list[float]]): Specific test points.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, int]:
            df_train, df_test, n_features
    """
    sampler = DataSampler(job)

    # Train
    train = sampler.sample_train_data()
    d_train = train["X"].shape[1]
    x_cols_train = [f"x{i + 1}" for i in range(d_train)]
    df_train = pd.DataFrame(train["X"], columns=x_cols_train)
    df_train["y"] = train["y"]
    df_train["sigma"] = train["sigma"]

    # Test
    test = sampler.sample_test_data()
    d_test = test["X"].shape[1]
    x_cols_test = [f"x{i + 1}" for i in range(d_test)]
    df_test = pd.DataFrame(test["X"], columns=x_cols_test)
    df_test["y"] = test["y"]
    df_test["sigma"] = test["sigma"]
    df_test["y_clean"] = test["y_clean"]

    # Input dimensionality
    n_features = int(train["n_features"])

    return df_train, df_test, n_features


def create_train_test_data_multi_output(
    job: dict[str, str | int | float | list[Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Build multi-output train/test DataFrames by composing multiple single-output samplers.

    This reproduces exactly what you would obtain if you ran a separate
    DataSampler with the same seed for each [f, sigma] pair and then stacked
    the outputs into a multi-target dataset.

    job["function"] may be:
      - ["f_name", "sigma_name"]              (single-output)
      - [[f1, s1], [f2, s2], ...]             (multi-output)

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, int]:
            df_train, df_test, n_features

    Notes:
        - df_train columns: x1..xD, y, sigma
          where y/sigma are per-row tuples: (y1, y2, ...).
        - df_test columns: x1..xD, y, sigma (y in test is “clean” -> without noise).
    """
    # Normalize to a list of pairs
    pairs: Sequence[Sequence[Any]] = job["function"]

    train_runs: list[dict[str, Any]] = []
    test_runs: list[dict[str, Any]] = []

    # For each pair, call a separate sampler with the same seed
    for pair in pairs:
        job_k = copy.deepcopy(job)
        job_k["function"] = pair  # keep identical seed and other params
        sampler_k = DataSampler(job_k)

        train_k = sampler_k.sample_train_data()
        test_k = sampler_k.sample_test_data()

        train_runs.append(train_k)
        test_runs.append(test_k)

    # Ensure all X are identical across outputs
    X_train = train_runs[0]["X"]
    X_test = test_runs[0]["X"]
    for k, tr in enumerate(train_runs[1:], start=2):
        if not np.array_equal(X_train, tr["X"]):
            raise RuntimeError(
                f"Train X differs for output #{k}. "
                "Seed and intervals must be identical for bitwise reproduction."
            )
    for k, te in enumerate(test_runs[1:], start=2):
        if not np.array_equal(X_test, te["X"]):
            raise RuntimeError(
                f"Test X differs for output #{k}. Please use identical test parameters."
            )

    n_features = int(train_runs[0]["n_features"])
    x_cols = [f"x{i + 1}" for i in range(n_features)]
    df_train = pd.DataFrame(X_train, columns=x_cols)
    df_test = pd.DataFrame(X_test, columns=x_cols)

    # Stack per-output targets and noises: shape (N, K)
    y_train = np.stack([tr["y"] for tr in train_runs], axis=1)
    sigma_train = np.stack([tr["sigma"] for tr in train_runs], axis=1)

    y_test = np.stack([te["y"] for te in test_runs], axis=1)
    sigma_test = np.stack([te["sigma"] for te in test_runs], axis=1)
    y_clean = np.stack([te["y_clean"] for te in test_runs], axis=1)

    # Store per-row tuples
    df_train["y"] = list(map(tuple, y_train))
    df_train["sigma"] = list(map(tuple, sigma_train))

    df_test["y"] = list(map(tuple, y_test))
    df_test["sigma"] = list(map(tuple, sigma_test))
    df_test["y_clean"] = list(map(tuple, y_clean))

    return df_train, df_test, n_features

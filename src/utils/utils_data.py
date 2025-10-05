import pandas as pd
from typing import Dict, Tuple, Union
from src.data_sampler import DataSampler


def create_train_test_data(
    job: Dict[str, Union[str, int, float, list]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build training and test DataFrames from a given config dictionary.

    Expected keys in the config (matching the DataSampler class):
        seed (int): Random seed.
        function (str): Key in FUNCTIONS.
        noise (str): Key in NOISES.
        train_interval (list): [a, b] or [[a, b], [c, d], ...].
        train_n_instances (int): Number of training instances.
        train_n_repeats (int): Number of repeats per instance.
        test_interval (list): [a, b].
        test_grid_length (int): Number of grid points per axis for the test set.
        (optional) test_points (list of lists): Specific test points.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, int]: df_train, df_test, n_features
    """
    sampler = DataSampler(job)

    # Train
    train = sampler.sample_train_data()
    d_train = train["X"].shape[1]
    x_cols_train = [f"x{i + 1}" for i in range(d_train)]
    df_train = pd.DataFrame(train["X"], columns=x_cols_train)
    df_train["y"] = train["y"]
    df_train["y_clean"] = train["y_clean"]
    df_train["sigma"] = train["sigma"]

    # Test
    test = sampler.sample_test_data()
    d_test = test["X"].shape[1]
    x_cols_test = [f"x{i + 1}" for i in range(d_test)]
    df_test = pd.DataFrame(test["X"], columns=x_cols_test)
    df_test["y"] = test["y"]
    df_test["sigma"] = test["sigma"]

    # Dimension
    n_features = train["n_features"]

    return df_train, df_test, n_features

import pandas as pd
from typing import Dict, Tuple, Union
from src.data_sampler import DataSampler


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
    df_train["sigma"] = train["sigma"]

    # --- Test ---
    test = sampler.sample_test_data()
    d_test = test["X"].shape[1]
    x_cols_test = [f"x{i + 1}" for i in range(d_test)]
    df_test = pd.DataFrame(test["X"], columns=x_cols_test)
    df_test["y"] = test["y"]
    df_test["sigma"] = test["sigma"]

    # --- Dimensionen ---
    n_features = train["n_features"]

    return df_train, df_test, n_features

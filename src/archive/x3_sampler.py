import numpy as np
import pandas as pd
import torch


class x3Sampler:
    def __init__(self, n_samples=1000, seed=None):
        self.seed = seed
        self.n_samples = n_samples

        if seed is not None:
            np.random.seed(seed)

    def sample_train_data(self, x_min=-4, x_max=4, rng=None, train=True, dtype=None):
        if rng is None:
            rng = np.random.default_rng(seed=42)

        if dtype is None:
            dtype = np.float32
        n = self.n_samples

        x = np.linspace(x_min, x_max, n)
        x = np.expand_dims(x, -1).astype(dtype)

        sigma = 3.0 * np.ones_like(x) if train else np.zeros_like(x)
        y = x**3 + rng.normal(0.0, sigma).astype(dtype)

        df = pd.DataFrame(np.concatenate([x, y], axis=1), columns=["x0", "y"])
        return df

    def sample_test_data(self, x_min=-7, x_max=7, rng=None, train=False, dtype=None):
        if rng is None:
            rng = np.random.default_rng(seed=42)

        if dtype is None:
            dtype = np.float32
        n = 300

        # x = np.linspace(x_min, x_max, n)
        # x = np.expand_dims(x, -1).astype(dtype)
        x = torch.linspace(x_min, x_max, n).unsqueeze(1).to("cpu")

        sigma = 3.0 * np.ones_like(x) if train else np.zeros_like(x)
        y = x**3 + rng.normal(0.0, sigma).astype(dtype)

        df = pd.DataFrame(np.concatenate([x, y], axis=1), columns=["x0", "y"])
        return df

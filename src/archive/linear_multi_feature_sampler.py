import numpy as np
import pandas as pd


class LinearMultiFeatureSampler:
    def __init__(self, n_features, seed):
        self.n_features = n_features
        self.seed = seed
        self.weights = np.linspace(0.1, 1.0, n_features).astype(np.float32)
        self.weights_noise = np.linspace(0.1, 1.0, n_features).astype(np.float32)

        # Set seed
        np.random.seed(seed)

    def sample_train_data(
        self,
        n_unique,
        n_repeats,
        min_val,
        max_val,
        sparse_center_frac,
    ) -> pd.DataFrame:
        n_sparse = int(n_unique * sparse_center_frac)
        n_dense = n_unique - n_sparse

        # Dense points away from center
        X_dense = np.random.uniform(
            min_val, max_val, size=(n_dense, self.n_features)
        ).astype(np.float32)
        X_dense = X_dense[np.any(np.abs(X_dense) > 0.5, axis=1)]

        # Sparse points in center region
        X_sparse = np.random.uniform(
            -0.5, 0.5, size=(n_sparse, self.n_features)
        ).astype(np.float32)

        # Combine and repeat
        X_unique = np.vstack([X_dense, X_sparse])
        X = np.repeat(X_unique, repeats=n_repeats, axis=0)

        y_true = np.sum(X * self.weights, axis=1)
        noise_std = np.sum(np.abs(X) * self.weights_noise, axis=1)
        y = y_true + np.random.normal(0, noise_std)

        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(self.n_features)])
        df["y"] = y
        df["aleatoric_true"] = noise_std
        return df

    def sample_test_grid(self, grid_length, min_val, max_val) -> pd.DataFrame:
        axes = [
            np.linspace(min_val, max_val, grid_length) for _ in range(self.n_features)
        ]
        mesh = np.meshgrid(*axes)
        X = np.stack(mesh, axis=-1).reshape(-1, self.n_features).astype(np.float32)

        y_true = np.sum(X * self.weights, axis=1)
        noise_std = np.sum(np.abs(X) * self.weights_noise, axis=1)

        df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(self.n_features)])
        df["y"] = y_true
        df["aleatoric_true"] = noise_std
        return df

import numpy as np
import pandas as pd


class CubicSingleFeatureSampler:
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def sample_train_data(
        self,
        n_unique,
        n_repeats,
        min_val,
        max_val,
        use_sparse_center=False,
        n_sparse_center=0,
    ) -> pd.DataFrame:
        """
        Generates 1D training data with repeated samples and heteroskedastic noise.

        If use_sparse_center=True, samples are sparse in center and dense outside.
        If False, samples are uniform across the full range.
        """
        rng = np.random.default_rng(seed=42)
        if use_sparse_center:
            if n_sparse_center > n_unique:
                raise ValueError("n_sparse_center cannot exceed n_unique.")
            n_dense = n_unique - n_sparse_center

            # Dense sampling outside the center
            x_dense = []
            while len(x_dense) < n_dense:
                candidates = np.random.uniform(min_val, max_val, size=(n_dense * 2,))
                x_dense = candidates[np.abs(candidates) > 0.5][:n_dense]

            # Sparse sampling in the center
            x_sparse = np.random.uniform(-0.5, 0.5, size=(n_sparse_center,))

            x_unique = np.concatenate([x_dense, x_sparse]).astype(np.float32)

        else:
            # Uniform sampling across full range
            x_unique = np.random.uniform(min_val, max_val, size=(n_unique,)).astype(
                np.float32
            )

        x = np.repeat(x_unique, repeats=n_repeats)
        y_true = x**3
        noise_std = 0.1 * (x**2)
        y = y_true + rng.normal(0, noise_std)

        df = pd.DataFrame({"x": x, "y": y, "aleatoric_true": noise_std})
        return df

    def sample_test_data(self, n_points, min_val, max_val) -> pd.DataFrame:
        """
        Generates evenly spaced 1D test data across a wider range.
        """
        x = np.linspace(min_val, max_val, n_points).astype(np.float32)
        y_true = x**3
        noise_std = 0.1 * (x**2)

        df = pd.DataFrame({"x": x, "y": y_true, "aleatoric_true": noise_std})
        return df

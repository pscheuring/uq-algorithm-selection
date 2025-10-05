from typing import Callable, Dict, Union
import numpy as np
from scipy.stats import qmc


def f_linear_1_feature(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return x1**3


def f_linear_1_feature_complex(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return 1.5 + 0.1 * x1**2 + 0.3 * x1 + np.sin(2 * x1 - 2)


def f_linear_1_feature_complex_v2(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return 1 + 0.1 * x1**3 + np.sin(x1 - 2)


def sigma_linear_1_feature_complex(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return 1.3 + np.sin(2 * x1) + 0.3 * x1


def sigma_linear_1_feature_complex_v2(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return 0.5 - 0.1 * (x1**2)


def f_non_linear_2_features(X: np.ndarray) -> np.ndarray:
    x1, x2 = X[:, 0], X[:, 1]
    return np.sin(x1) + 0.5 * (x2**2)


def f_non_linear_3_features(X: np.ndarray) -> np.ndarray:
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(x1) + x2 * x3


def sigma_constant_1_feature(X: np.ndarray) -> np.ndarray:
    return 3.0


def sigma_linear_1_feature(X: np.ndarray) -> np.ndarray:
    x1 = X[:, 0]
    return 0.1 * (x1**2)


def sigma_linear_2_features(X: np.ndarray) -> np.ndarray:
    x1, x2 = X[:, 0], X[:, 1]
    return 0.8 * np.abs(x1) + 0.2 * np.abs(x2)


FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "f_linear_1_feature": f_linear_1_feature,
    "f_linear_1_feature_complex": f_linear_1_feature_complex,
    "f_linear_1_feature_complex_v2": f_linear_1_feature_complex_v2,
    "f_non_linear_2_features": f_non_linear_2_features,
    "f_non_linear_3_features": f_non_linear_3_features,
}

SIGMAS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigma_linear_1_feature": sigma_linear_1_feature,
    "sigma_linear_1_feature_complex": sigma_linear_1_feature_complex,
    "sigma_linear_1_feature_complex_v2": sigma_linear_1_feature_complex_v2,
    "sigma_linear_2_features": sigma_linear_2_features,
    "sigma_constant_1_feature": sigma_constant_1_feature,
}


class DataSampler:
    """Minimalistic data sampler.

    Expects a flat config dictionary (`job`) with keys:
      - seed, function, sigma
      - train_interval, train_instances, train_repeats
      - test_interval, test_grid_length
      - (optional) test_points
    """

    def __init__(
        self,
        job: Dict[str, Union[str, int, float, list]],
        functions: Dict[str, Callable] = FUNCTIONS,
        sigmas: Dict[str, Callable] = SIGMAS,
    ) -> None:
        """Initialize the sampler.

        Args:
            job: Flat config with data generation settings.
            functions: Mapping from function name to callable f(X)->y.
            sigmas: Mapping from noise function name to callable sigma(X)->σ.
        """
        self.job = job
        self.functions = functions
        self.sigmas = sigmas
        self.rng = np.random.default_rng(seed=job["seed"])

    def sample_train_data(self) -> dict[str, np.ndarray]:
        """Sample training data with heteroscedastic noise using Latin Hypercube Sampling.

        Returns:
            Dict with:
                - X: (N, D) inputs
                - y: (N,) or (N, 1) noisy targets
                - y_clean: (N,) or (N, 1) noise-free targets
                - sigma: (N,) or (N, 1) noise std per sample
                - n_features: scalar number of input features D
        """
        fn = self.functions[self.job["function"][0]]
        sigma_fn = self.sigmas[self.job["function"][1]]

        interval_spec = self.job["train_interval"]
        n_instances = self.job["train_instances"]
        n_repeats = self.job["train_repeats"]

        n_features = self._probe_dim(fn)

        is_single_interval = not isinstance(interval_spec[0], (list, tuple, np.ndarray))

        if is_single_interval:
            a, b = map(float, interval_spec)
            sampler = qmc.LatinHypercube(d=n_features, seed=self.rng)
            X_unique = qmc.scale(
                sampler.random(n_instances),
                [a] * n_features,
                [b] * n_features,
            )
        else:  # Two intervals
            (a, b), (c, d) = interval_spec
            a, b, c, d = map(float, (a, b, c, d))
            n1 = n_instances // 2
            n2 = n_instances - n1

            if n_features == 1:
                s1 = qmc.LatinHypercube(d=1, rng=self.rng)
                s2 = qmc.LatinHypercube(d=1, rng=self.rng)
                axis = np.concatenate(
                    [
                        qmc.scale(s1.random(n1), [a], [b]).ravel(),
                        qmc.scale(s2.random(n2), [c], [d]).ravel(),
                    ]
                )
                X_unique = axis.reshape(-1, 1)
            else:
                # For each feature: draw LHS samples from both intervals and permute independently
                # Avoids grid artifacts and create Latin-hypercube-like combinations across dimensions
                axes = []
                for _ in range(n_features):
                    parts = []

                    s1 = qmc.LatinHypercube(d=1, rng=self.rng)
                    parts.append(qmc.scale(s1.random(n1), [a], [b]).ravel())
                    s2 = qmc.LatinHypercube(d=1, rng=self.rng)
                    parts.append(qmc.scale(s2.random(n2), [c], [d]).ravel())

                    base_axis = np.concatenate(parts)
                    axes.append(base_axis[self.rng.permutation(n_instances)])

                X_unique = np.stack(axes, axis=1)

        # Repeats (identical X multiple times)
        X = (
            X_unique
            if n_repeats <= 1
            else np.repeat(X_unique, repeats=n_repeats, axis=0)
        )

        # Targets + heteroscedastic noise
        y_clean = fn(X)
        sigma = sigma_fn(X)
        noise = self.rng.normal(loc=0.0, scale=sigma, size=X.shape[0])
        y = y_clean + noise

        return {
            "X": X,
            "y": y,
            "y_clean": y_clean,
            "sigma": sigma,
            "n_features": np.int64(n_features),
        }

    def sample_test_data(self) -> dict[str, np.ndarray]:
        """Sample test data on a uniform meshgrid spanning test_interval.

        Builds a Cartesian grid with test_grid_length points per dimension.
        Optionally appends 1D test_points (broadcast to all dims when D>1).
        Duplicates are removed.

        Returns:
            Dict with:
                - X: (M, D) test inputs
                - y: (M,  ) or (M, 1) clean targets
                - sigma: (M,)  noise std per input
        """
        fn = self.functions[self.job["function"][0]]
        sigma_fn = self.sigmas[self.job["function"][1]]

        n_features = self._probe_dim(fn)

        min_val, max_val = map(float, self.job["test_interval"])
        grid_length = int(self.job["test_grid_length"])

        X_list: list[np.ndarray] = []

        axis = np.linspace(min_val, max_val, grid_length, dtype=np.float64)
        mesh = np.meshgrid(*([axis] * n_features), indexing="xy")
        X_grid = np.stack(mesh, axis=-1).reshape(-1, n_features)
        X_list.append(X_grid)

        test_points = self.job.get("test_points", None)
        if test_points is not None:
            # flat list of scalars → use as-is for D==1, else broadcast across dims
            if all(not isinstance(t, (list, tuple, np.ndarray)) for t in test_points):
                vals = np.asarray(test_points, dtype=np.float64).reshape(-1, 1)
                X_extra = vals if n_features == 1 else np.tile(vals, (1, n_features))
                X_list.append(X_extra)

        X = np.vstack(X_list)
        X = np.unique(X, axis=0)

        y = fn(X)
        sigma = sigma_fn(X)

        return {"X": X, "y": y, "sigma": sigma}

    def _probe_dim(self, f: Callable[[np.ndarray], np.ndarray]) -> int:
        """Infer input dimensionality of a function by probing.

        Tries zero-input batches with dimensions 1..1000 until f accepts it.

        Args:
            f: Callable mapping (N, D) arrays to targets.

        Returns:
            Detected input dimension D.
        """
        for d in range(1, 1000):
            try:
                _ = f(np.zeros((1, d)))
                return d
            except Exception:
                pass

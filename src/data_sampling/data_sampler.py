from typing import Callable, Dict, List, Union, Tuple
import numpy as np


def f_non_linear_two_features(X: np.ndarray) -> np.ndarray:
    x1, x2 = X[:, 0], X[:, 1]
    return np.sin(x1) + 0.5 * (x2**2)


def f_non_linear_three_features(X: np.ndarray) -> np.ndarray:
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(x1) + x2 * x3


def noise_non_linear_two_features(
    X: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    x1 = X[:, 0]
    sigma = 0.1 * (1.0 + np.abs(x1))
    return rng.normal(loc=0.0, scale=sigma, size=X.shape[0])


FUNCTIONS = {
    "non_linear_two_features": f_non_linear_two_features,
    "non_linear_three_features": f_non_linear_three_features,
}

NOISES = {
    "non_linear_two_features": noise_non_linear_two_features,
}


class DataSampler:
    """
    Minimalistic data sampler. Expects a flat config dictionary with keys:
      seed, function, noise,
      train_interval, train_n_instances, train_n_repeats,
      test_interval, test_grid_length (or test_n_instances/test_n_repeats)
    """

    def __init__(
        self,
        job: Dict[str, Union[str, int, float, list]],
        functions: Dict[str, Callable] = FUNCTIONS,
        noises: Dict[str, Callable] = NOISES,
    ) -> None:
        self.job = job
        self.functions = functions
        self.noises = noises

    def sample_train_data(self) -> Dict[str, np.ndarray]:
        job = self.job
        fn = self.functions[job["function"][0]]
        noise_fn = self.noises[job["function"][1]]
        interval_spec = job["train_interval"]
        n_instances = int(job["train_instances"])
        n_repeats = int(job["train_repeats"])

        # Infer feature dimension
        dim = self._probe_dim(fn)

        # Initialize random number generator
        rng = np.random.default_rng(job["random_seed"])

        # Inline interval normalization (single [a,b] vs [[a,b],[c,d],...])
        if len(interval_spec) == 2 and not isinstance(interval_spec[0], (list, tuple)):
            intervals: List[Tuple[float, float]] = [
                (float(interval_spec[0]), float(interval_spec[1]))
            ]
        else:
            intervals = [(float(a), float(b)) for a, b in interval_spec]

        start, end = self._choose_interval_bounds(rng, intervals, n_instances, dim)

        # Sample uniformly from chosen intervals (vectorized over instances and features)
        X_unique = rng.uniform(start, end)

        X = (
            X_unique
            if n_repeats <= 1
            else np.repeat(X_unique, repeats=n_repeats, axis=0)
        )

        y_clean = fn(X)
        noise = noise_fn(X, rng)
        y = y_clean + noise
        return {"X": X, "y": y, "y_clean": y_clean, "noise": noise}

    def sample_test_data(self) -> Dict[str, np.ndarray]:
        job = self.job
        fn = self.functions[job["function"]]
        noise_fn = self.noises[job["noise"]]

        # infer feature dimension
        d = self._probe_dim(fn)

        # RNG
        rng = np.random.default_rng(job["random_seed"])

        # single interval [min_val, max_val] for test
        min_val, max_val = map(float, job["test_interval"])
        grid_length = job["test_grid_length"]

        # build Cartesian grid with equal spacing per feature
        axis = np.linspace(min_val, max_val, grid_length, dtype=np.float64)
        mesh = np.meshgrid(*([axis] * d), indexing="xy")
        X = np.stack(mesh, axis=-1).reshape(-1, d)

        # targets
        y_clean = fn(X)
        noise = noise_fn(X, rng)
        y = y_clean + noise

        return {"X": X, "y": y, "y_clean": y_clean, "noise": noise}

    def _probe_dim(self, f: Callable) -> int:
        for d in range(1, 11):
            try:
                _ = f(np.zeros((1, d)))
                return d
            except Exception:
                pass
        raise ValueError(
            "Cannot infer input dimension; please register a known function."
        )

    def _choose_interval_bounds(
        self,
        rng: np.random.Generator,
        intervals: List[Tuple[float, float]],
        n_instances: int,
        dim: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Choose start/end bounds for random draws from a union of intervals.

        For each element in a matrix of shape (n_instances, dim),
        an interval is selected with probability proportional to its length.
        The function returns two arrays `start` and `end` with the chosen lower
        and upper bounds.

        Args:
            rng: NumPy random number generator.
            intervals: List of (start, end) tuples defining allowed ranges.
            n_instances: Number of rows (instances).
            dim: Number of features (columns).

        Returns:
            Tuple (start, end):
                start: np.ndarray of shape (n_instances, dim), lower bounds.
                end:   np.ndarray of shape (n_instances, dim), upper bounds.

        Example:
            >>> rng = np.random.default_rng(0)
            >>> intervals = [(-2, 2), (4, 6)]
            >>> start, end = choose_interval_bounds(rng, intervals, n_instances=3, dim=2)
            >>> start
            array([[-2., -2.],
                [-2.,  4.],
                [ 4.,  4.]])
            >>> end
            array([[ 2.,  2.],
                [ 2.,  6.],
                [ 6.,  6.]])
        """
        intervals_arr = np.asarray(intervals, dtype=float)

        # interval lengths -> probabilities
        interval_lengths = intervals_arr[:, 1] - intervals_arr[:, 0]
        interval_probs = interval_lengths / interval_lengths.sum()

        # pick interval index for each sample
        seg_idx = rng.choice(
            len(intervals_arr), size=(n_instances, dim), p=interval_probs
        )

        # directly extract start and end bounds
        start = intervals_arr[seg_idx, 0]
        end = intervals_arr[seg_idx, 1]

        return start, end

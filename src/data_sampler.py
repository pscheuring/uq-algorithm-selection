from typing import Callable, Dict, List, Union, Tuple
import numpy as np


def f_non_linear_2_features(X: np.ndarray) -> np.ndarray:
    x1, x2 = X[:, 0], X[:, 1]
    return np.sin(x1) + 0.5 * (x2**2)


def f_non_linear_3_features(X: np.ndarray) -> np.ndarray:
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(x1) + x2 * x3


def sigma_non_linear_2_features(X: np.ndarray) -> np.ndarray:
    """Deterministische Rausch-Skala σ(X). Kein Ziehen von Noise hier!"""
    x1, x2 = X[:, 0], X[:, 1]
    return np.sin(x1) + (1.0 + np.abs(x2))


FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "f_non_linear_2_features": f_non_linear_2_features,
    "f_non_linear_3_features": f_non_linear_3_features,
}

SIGMAS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sigma_non_linear_2_features": sigma_non_linear_2_features,
}


class DataSampler:
    """
    Minimalistic data sampler. Expects a flat config dictionary with keys:
      seed, function, sigma,
      train_interval, train_instances, train_repeats,
      test_interval, test_grid_length,
      (optional) test_points  # flache Liste -> wird bei d>1 zu [v,...,v] gekachelt
    """

    def __init__(
        self,
        job: Dict[str, Union[str, int, float, list]],
        functions: Dict[str, Callable] = FUNCTIONS,
        sigmas: Dict[str, Callable] = SIGMAS,
    ) -> None:
        self.job = job
        self.functions = functions
        self.sigmas = sigmas

    # --------------------------
    # Public API
    # --------------------------

    def sample_train_data(self) -> Dict[str, np.ndarray]:
        job = self.job
        fn = self.functions[job["function"][0]]
        sigma_fn = self.sigmas[job["function"][1]]

        interval_spec = job["train_interval"]
        n_instances = int(job["train_instances"])
        n_repeats = int(job["train_repeats"])

        # Feature-Dimension ermitteln
        n_features = self._probe_dim(fn)

        # RNG
        rng = np.random.default_rng(job["seed"])

        # Intervalle normalisieren: [a,b] oder [[a,b], [c,d], ...]
        if len(interval_spec) == 2 and not isinstance(interval_spec[0], (list, tuple)):
            intervals: List[Tuple[float, float]] = [
                (float(interval_spec[0]), float(interval_spec[1]))
            ]
        else:
            intervals = [(float(a), float(b)) for a, b in interval_spec]

        start, end = self._choose_interval_bounds(
            rng, intervals, n_instances, n_features
        )

        # Uniform aus gewählten Intervallen ziehen
        X_unique = rng.uniform(start, end)

        X = (
            X_unique
            if n_repeats <= 1
            else np.repeat(X_unique, repeats=n_repeats, axis=0)
        )

        # Targets
        y_clean = fn(X)
        sigma = sigma_fn(X)
        noise = rng.normal(loc=0.0, scale=sigma, size=X.shape[0])
        y = y_clean + noise

        return {
            "X": X,
            "y": y,
            "y_clean": y_clean,
            "sigma": sigma,
            "n_features": n_features,
        }

    def sample_test_data(self) -> Dict[str, np.ndarray]:
        job = self.job
        fn = self.functions[job["function"][0]]
        sigma_fn = self.sigmas[job["function"][1]]

        n_features = self._probe_dim(fn)

        # --- Grid bauen (robust gegenüber Liste/Skalar) ---
        min_val, max_val = map(float, job["test_interval"])

        # test_grid_length: int oder [int]
        grid_length_val = job["test_grid_length"]
        grid_length = grid_length_val

        X_list = []

        axis = np.linspace(min_val, max_val, grid_length, dtype=np.float64)
        mesh = np.meshgrid(*([axis] * n_features), indexing="xy")
        X_grid = np.stack(mesh, axis=-1).reshape(-1, n_features)
        X_list.append(X_grid)

        # --- Zusätzliche Testpunkte integrieren ---
        test_points = job.get("test_points", None)
        if test_points is not None:
            # Fall 1: flache Liste von Skalaren -> bei d==1 normal,
            # bei d>1 zu [v, v, ..., v] (gleicher Wert in allen Dimensionen) erweitern
            if all(not isinstance(t, (list, tuple, np.ndarray)) for t in test_points):
                vals = np.asarray(test_points, dtype=np.float64).reshape(-1, 1)
                if n_features == 1:
                    X_extra = vals
                else:
                    X_extra = np.tile(vals, (1, n_features))
                X_list.append(X_extra)

        # Kombinieren + Deduplizieren
        X = np.vstack(X_list)
        # np.unique für Reihen
        X = np.unique(X, axis=0)

        # Targets
        y = fn(X)
        sigma = sigma_fn(X)

        return {"X": X, "y": y, "sigma": sigma}

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

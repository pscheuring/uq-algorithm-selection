import numpy as np
from pathlib import Path
from src.constants import RESULTS_DIR


def load_meta_model_benchmarking_results(folder_name, base_path=RESULTS_DIR):
    """Load uncertainty benchmarking arrays and return a dict of numpy arrays."""
    folder = Path(base_path) / folder_name

    # Load arrays
    aleatoric_all = np.load(folder / "aleatoric_all.npy")
    aleatoric_true = np.load(folder / "aleatoric_true.npy")

    epistemic_all = np.load(folder / "epistemic_all.npy")

    y_pred_all = np.load(folder / "y_pred_all.npy")
    y_test = np.load(folder / "y_test.npy")
    y_clean = np.load(folder / "y_clean.npy")

    y_train = np.load(folder / "y_train.npy")
    X_test = np.load(folder / "X_test.npy")
    X_train = np.load(folder / "X_train.npy")

    # Determine number of features
    n_features = X_test.shape[1] if X_test.ndim > 1 else 1
    feature_names = [f"x{i + 1}" for i in range(n_features)]

    return {
        "X_test": X_test,
        "y_test": y_test,
        "y_clean": y_clean,
        "X_train": X_train,
        "y_train": y_train,
        "y_pred_all": y_pred_all,
        "aleatoric_all": aleatoric_all,
        "aleatoric_true": aleatoric_true,
        "epistemic_all": epistemic_all,
        "feature_names": feature_names,
    }

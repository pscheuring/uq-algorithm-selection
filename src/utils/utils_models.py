from src.models.bbb import BayesByBackprop
from src.models.ensemble import DeepEnsemble
from src.models.evidential import DeepEvidentialRegression
from src.models.mcdropout import MCDropout


def build_model(model_name, model_params, seed, n_features):
    """Factory function to build a model based on its name."""
    if model_name == "mcdropout":
        return MCDropout(
            **model_params,
            seed=seed,
            in_features=n_features,
        )
    elif model_name == "ensemble":
        return DeepEnsemble(
            **model_params,
            seed=seed,
            in_features=n_features,
        )
    elif model_name == "evidential":
        return DeepEvidentialRegression(
            **model_params,
            seed=seed,
            in_features=n_features,
        )
    elif model_name == "bbb":
        return BayesByBackprop(
            **model_params,
            seed=seed,
            in_features=n_features,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def choose_hparams_by_bins(n_samples: int):
    """
    Returns a dictionary with batch_size, learning_rate (lr), and epochs
    based on fixed dataset size bins.

    Bins:
      - n < 1,000:         batch=16,  lr=5e-4,  epochs=600
      - 1,000 <= n < 5,000: batch=32,  lr=1e-3,  epochs=300
      - 5,000 <= n < 25,000: batch=64, lr=1e-3,  epochs=150
      - n >= 25,000:       batch=128, lr=2e-3,  epochs=75
    """
    if n_samples < 1000:
        return {"batch_size": 16, "lr": 5e-4, "epochs": 600}
    elif n_samples < 5000:
        return {"batch_size": 32, "lr": 1e-3, "epochs": 300}
    elif n_samples < 25000:
        return {"batch_size": 64, "lr": 1e-3, "epochs": 150}
    else:
        return {"batch_size": 128, "lr": 2e-3, "epochs": 75}

from src.models.bbb import BayesByBackprop
from src.models.ensemble import DeepEnsemble
from src.models.evidential import DeepEvidentialRegression
from src.models.mcdropout import MCDropout


def build_model(model_name, model_params, seed, n_features):
    if model_name == "mcdropout":
        return MCDropout(
            **model_params,
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
            in_features=n_features,
        )
    elif model_name == "bbb":
        return BayesByBackprop(
            **model_params,
            in_features=n_features,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

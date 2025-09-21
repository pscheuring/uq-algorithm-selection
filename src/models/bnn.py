from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras

from src.models.base_meta_model import BaseModel
from utils import logger

tfpl = tfp.layers
tfd = tfp.distributions


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return keras.Sequential(
        [
            tfpl.DistributionLambda(
                lambda t: tfd.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return keras.Sequential(
        [
            tfpl.VariableLayer(tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype),
            tfpl.MultivariateNormalTriL(n),
        ]
    )


class BNN(BaseModel):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        learning_rate: float,
        mc_samples: int,
        epochs: int,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mc_samples = mc_samples
        self.model = None
        self.n_samples = None

    def _build_model(self, n_samples: int) -> tf.keras.Model:
        model = keras.Sequential(
            [
                tfpl.DenseVariational(
                    units=self.hidden_dim,
                    input_shape=(self.input_dim,),
                    make_prior_fn=prior,
                    make_posterior_fn=posterior,
                    kl_weight=1 / n_samples,
                    kl_use_exact=False,
                    activation="sigmoid",
                ),
                tfpl.DenseVariational(
                    units=2,
                    make_prior_fn=prior,
                    make_posterior_fn=posterior,
                    kl_weight=1 / n_samples,
                    kl_use_exact=False,
                ),
                tfpl.IndependentNormal(1),
            ]
        )

        def neg_loglik(y_true, y_pred):
            return -y_pred.log_prob(y_true)

        model.compile(
            loss=neg_loglik,
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
        )
        return model

    def fit(self, train_df: pd.DataFrame):
        X_train = train_df[
            [col for col in train_df.columns if col.startswith("x")]
        ].values
        y_train = train_df["y"].values.reshape(-1, 1)
        self.n_samples = X_train.shape[0]

        self.model = self._build_model(self.n_samples)
        logger.info(f"Training BNN with {self.n_samples} samples")
        self.model.fit(X_train, y_train, epochs=self.epochs, verbose=1)

    def predict_with_uncertainties(
        self, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
        - mean prediction (MC averaged)
        - epistemic uncertainty (stddev of means)
        - aleatoric uncertainty (mean of stddevs)
        """
        logger.info(f"Predicting with uncertainties using {self.mc_samples} MC samples")

        all_means = []
        all_stddevs = []

        for _ in range(self.n_mc_samples):
            preds = self.model(X_test)
            all_means.append(preds.mean().numpy().flatten())
            all_stddevs.append(preds.stddev().numpy().flatten())

        all_means = np.stack(all_means)  # shape: [n_mc_samples, n_points]
        all_stddevs = np.stack(all_stddevs)  # shape: [n_mc_samples, n_points]

        y_pred = np.mean(all_means, axis=0)
        epistemic = np.std(all_means, axis=0)
        aleatoric = np.mean(all_stddevs, axis=0)

        return y_pred, epistemic, aleatoric

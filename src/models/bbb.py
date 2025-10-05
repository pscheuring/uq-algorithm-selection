import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.base_model import BaseModel
from src.models.layers import VariationalDense, VariationalDenseNormal
from src.models.losses import elbo_loss
from src.utils.utils_logging import logger


class BayesByBackprop(BaseModel):
    """Bayesian NN with mean-field variational inference (Bayes by Backprop).

    Based on Blundell et al. (2015). Trains with ELBO and a front-loaded beta
    schedule per batch. Head outputs [mu, log_var].
    """

    def __init__(
        self,
        kl_beta: float,
        prior_pi: float,
        prior_sigma1: float,
        prior_sigma2: float,
        mu_init_mean: float,
        mu_init_std: float,
        rho_init: float,
        n_mc_samples: int,
        **kwargs,
    ) -> None:
        """Init model and store prior/posterior hyperparameters.

        Args:
            kl_beta: Global KL weight.
            prior_pi: Mixture weight of prior.
            prior_sigma1: Std of first prior Gaussian.
            prior_sigma2: Std of second prior Gaussian.
            mu_init_mean: Mean for init of posterior mus.
            mu_init_std: Std for init of posterior mus.
            rho_init: Init value for posterior rhos.
            n_mc_samples: Number of MC samples at test time.
            **kwargs: Forwarded to BaseModel.
        """
        self._prior: dict[str, float] = {
            "prior_pi": prior_pi,
            "prior_sigma1": prior_sigma1,
            "prior_sigma2": prior_sigma2,
        }
        self._post_init: dict[str, float] = {
            "mu_init_mean": mu_init_mean,
            "mu_init_std": mu_init_std,
            "rho_init": rho_init,
        }
        self.kl_beta: float = kl_beta
        self.n_mc_samples: int = int(n_mc_samples)
        self.dataset_size: int | None = None

        # call after setting _prior/_post_init, because BaseModel.__init__ uses them
        super().__init__(**kwargs)

    def make_hidden_layer(self, in_features: int, hidden_features: int) -> nn.Module:
        """Return a variational dense layer (in_dim -> out_dim)."""
        return VariationalDense(
            in_features=in_features,
            out_features=hidden_features,
            **self._prior,
            **self._post_init,
        )

    def make_head(self, backbone_out_features: int, target_dim: int) -> nn.Module:
        """Return variational head projecting to [mu, log_var]."""
        return VariationalDenseNormal(
            in_features=backbone_out_features,
            target_dim=target_dim,
            **self._prior,
            **self._post_init,
        )

    def kl_total_loss(self) -> torch.Tensor:
        """Sum KL divergences from all variational submodules."""
        kl_total = torch.zeros((), device=self.device)
        for m in self.modules():
            if hasattr(m, "kl_loss") and callable(getattr(m, "kl_loss")):
                kl_total = kl_total + m.kl_loss()
        return kl_total

    def loss(
        self, y_true: torch.Tensor, head_out: torch.Tensor, beta: float
    ) -> torch.Tensor:
        """Compute negative ELBO (NLL_mean + beta*KL/N)."""
        kl = self.kl_total_loss()
        assert self.dataset_size is not None, "dataset_size not set, call fit() first"
        return elbo_loss(y_true, head_out, kl, N=self.dataset_size, beta=beta)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        clip_grad_norm: float | None = None,
    ) -> list[float]:
        """Train model on (X_train, y_train) by minimizing ELBO.

        Shapes:
            X_train: (N, input_dim)
            y_train: (N,) or (N, D)

        Returns:
            List of average epoch losses.
        """
        self.train()
        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(1)
        if X_train.ndim == 1:
            X_train = X_train.unsqueeze(1)

        self.dataset_size = int(X_train.shape[0])

        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            generator=generator,
        )
        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        losses: list[float] = []

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            num_batches = len(loader)

            # front-loaded beta schedule
            M = num_batches
            den = 2.0 - 2.0 ** (-(M - 1))

            for b_idx, (batch_x, batch_y) in enumerate(loader, start=1):
                pi_i = 2.0 ** (-(b_idx - 1)) / den
                beta = self.kl_beta * pi_i

                optimizer.zero_grad(set_to_none=True)
                pred = self(batch_x)
                loss = self.loss(batch_y, pred, beta=beta)
                loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            avg = epoch_loss / n_batches
            losses.append(avg)
            logger.info(f"Epoch {epoch:3d}/{self.epochs}  elbo={avg:.6f}")

        return losses

    @torch.no_grad()
    def predict_with_uncertainties(
        self, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict mean, epistemic variance, aleatoric variance.

        Runs n_mc_samples stochastic forward passes.

        Args:
            X_test: Test inputs shaped (N, input_dim).

        Returns:
            Tuple (mu_mean, epistemic_var, aleatoric_var), each (N, D).
        """
        self.eval()
        device = next(self.parameters()).device
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)

        mus: list[torch.Tensor] = []
        alea: list[torch.Tensor] = []
        for _ in range(self.n_mc_samples):
            out = self(X_test)
            mu, log_var = torch.chunk(out, 2, dim=-1)
            mus.append(mu)
            alea.append(torch.exp(log_var))

        mu_stack = torch.stack(mus, dim=0)
        alea_stack = torch.stack(alea, dim=0)

        mu_mean = mu_stack.mean(dim=0)
        epistemic = mu_stack.var(dim=0, unbiased=False)
        aleatoric = alea_stack.mean(dim=0)

        return (
            mu_mean.cpu().numpy().astype(np.float32),
            epistemic.cpu().numpy().astype(np.float32),
            aleatoric.cpu().numpy().astype(np.float32),
        )

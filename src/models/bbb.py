import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.base_model import BaseModel
from src.models.layers import VariationalDenseNormal, VariationalLinear
from src.models.losses import elbo_loss

from src.logging import logger


class BayesByBackdrop(BaseModel):
    """BNN mit MFVI (Bayes by Backprop). Based on Blundell et al. 2015."""

    def __init__(
        self,
        *args,
        kl_beta: float,
        prior_pi: float,
        prior_sigma1: float,
        prior_sigma2: float,
        n_mc_samples: int,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._prior = dict(
            prior_pi=prior_pi, prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2
        )
        self.kl_beta = kl_beta
        self.n_mc_samples = n_mc_samples

    def make_hidden_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        return VariationalLinear(in_dim, out_dim, **self._prior)

    def make_head(self, in_dim: int, out_dim: int) -> nn.Module:
        # Erwartet: concat([mu, logvar])
        return VariationalDenseNormal(in_dim, out_dim)

    def kl_loss(self) -> torch.Tensor:
        kl_total = torch.zeros((), device=self.device)
        for m in self.modules():
            if hasattr(m, "kl_loss") and callable(getattr(m, "kl_loss")):
                kl_total = kl_total + m.kl_loss()
        return kl_total

    def loss(self, y_true, head_out, beta: float):
        kl = self.kl_loss()
        return elbo_loss(y_true, head_out, kl, N=self.dataset_size, beta=beta)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        shuffle: bool = True,
        clip_grad_norm: float | None = None,
    ):
        """
        Trainiert das BNN (ELBO-Minimierung).
        - X: (N, input_dim), y: (N,) oder (N, D)
        """
        self.train()
        X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.as_tensor(y_train, dtype=torch.float32, device=self.device)
        if y_train.ndim == 1:
            y_train = y_train[:, None]  # (N,1)

        self.dataset_size = int(X_train.shape[0])

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle
        )

        optimizer = optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        losses = []

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            num_batches = len(loader)

            M = num_batches
            den = 2.0 - 2.0 ** (-(M - 1))  # denominator for pi_i

            for b_idx, (batch_x, batch_y) in enumerate(loader, start=1):
                # front-loaded β using numerically more stable version of (2.0 ** (M - i)) / (2.0 ** M - 1.0)
                pi_i = 2.0 ** (-(b_idx - 1)) / den
                beta = self.kl_beta * pi_i

                optimizer.zero_grad(set_to_none=True)
                pred = self(batch_x)
                loss = self.loss(batch_y, pred, beta=beta)  # NLL_mean + (beta * KL) / N
                loss.backward()
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), clip_grad_norm)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / n_batches
            losses.append(avg)
            logger.debug(f"Epoch {epoch:3d}/{self.epochs}  elbo={avg:.6f}")

        return losses

    @torch.no_grad()
    def predict_with_uncertainties(self, X_test: np.ndarray):
        """
        Gibt (mu_mean, epistemic_var, aleatoric_var) als np.ndarrays zurück. Shapes: (N, D)
        """
        self.eval()
        device = next(self.parameters()).device
        X_test = torch.as_tensor(X_test, dtype=torch.float32, device=device)

        mus = []
        alea = []
        for _ in range(self.n_mc_samples):
            out = self(X_test)  # (N, 2*D), VI sampelt Gewichte auch in eval()
            mu, log_var = torch.chunk(out, 2, dim=-1)
            mus.append(mu)
            alea.append(torch.exp(log_var))  # aleatorische Varianz

        mu_stack = torch.stack(mus, dim=0)  # (S,N,D)
        alea_stack = torch.stack(alea, dim=0)  # (S,N,D)

        mu_mean = mu_stack.mean(dim=0)  # (N,D)
        epistemic = mu_stack.var(dim=0, unbiased=False)  # (N,D)
        aleatoric = alea_stack.mean(dim=0)  # (N,D)

        return (
            mu_mean.cpu().numpy(),
            epistemic.cpu().numpy(),
            aleatoric.cpu().numpy(),
        )

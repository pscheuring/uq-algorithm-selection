# pip install torch numpy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ---------------- Bayesian Bausteine ----------------
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).normal_(0, 0.1)
        )
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

        self.register_buffer(
            "prior_log_var", torch.tensor(math.log(prior_std**2), dtype=torch.float32)
        )

    def _sigma(self, rho):  # softplus für Stabilität
        return F.softplus(rho)

    def _sample(self):
        eps_w = torch.randn_like(self.weight_mu)
        eps_b = torch.randn_like(self.bias_mu)
        w = self.weight_mu + self._sigma(self.weight_rho) * eps_w
        b = self.bias_mu + self._sigma(self.bias_rho) * eps_b
        return w, b

    def kl_divergence(self):
        prior_log_var = self.prior_log_var
        var_p_inv = torch.exp(-prior_log_var)

        w_var = self._sigma(self.weight_rho) ** 2
        b_var = self._sigma(self.bias_rho) ** 2

        kl_w = 0.5 * torch.sum(
            -prior_log_var
            + torch.log(w_var)
            + var_p_inv * (w_var + self.weight_mu**2)
            - 1.0
        )
        kl_b = 0.5 * torch.sum(
            -prior_log_var
            + torch.log(b_var)
            + var_p_inv * (b_var + self.bias_mu**2)
            - 1.0
        )
        return kl_w + kl_b

    def forward(self, x, sample=True):
        if sample:
            w, b = self._sample()
        else:
            w, b = self.weight_mu, self.bias_mu
        out = F.linear(x, w, b)
        return out, self.kl_divergence()


class MeanVarLinearBNN(nn.Module):
    """
    Ein lineares BNN, das pro Sample zwei Werte ausgibt:
      - mu_y (mean)
      - log_var_y (logarithmische Varianz; wird exponentiert für Varianz)
    """

    def __init__(self, in_features, prior_std=1.0):
        super().__init__()
        self.bayes = BayesianLinear(in_features, out_features=2, prior_std=prior_std)

    def forward(self, x, sample=True):
        y, kl = self.bayes(x, sample=sample)
        mu = y[..., 0]
        log_var = y[..., 1]
        return mu, log_var, kl

    @torch.no_grad()
    def predict_mc(self, x, num_samples=100):
        mus, vars_ = [], []
        for _ in range(num_samples):
            mu_s, log_var_s, _ = self.forward(x, sample=True)
            mus.append(mu_s)
            vars_.append(torch.exp(log_var_s).clamp_min(1e-12))
        mus = torch.stack(mus, 0)  # [S,N]
        alea = torch.stack(vars_, 0)  # [S,N]

        mean = mus.mean(0)
        epistemic = mus.var(0, unbiased=False)
        aleatoric = alea.mean(0)
        total = epistemic + aleatoric
        return mean, total, aleatoric, epistemic


# ---------------- Loss-Helfer ----------------
def gaussian_nll(y, mu, log_var):
    inv_var = torch.exp(-log_var).clamp_max(1e12)
    return (0.5 * (math.log(2 * math.pi) + log_var + (y - mu) ** 2 * inv_var)).mean()


# ---------------- High-level Wrapper mit fit/predict ----------------
class BNNRegressor:
    """
    Sklearn-ähnliche API:
      - fit(X, y)
      - predict(X, return_uncertainties=True) -> mean, (total_var, aleatoric_var, epistemic_var)
    Heteroskedastische Gauß-Likelihood, Mean-Field-VI, lineares Modell.
    """

    def __init__(
        self,
        prior_std: float = 1.0,
        lr: float = 2e-3,
        epochs: int = 50,
        batch_size: int = 128,
        mc_train_samples: int = 3,
        mc_pred_samples: int = 200,
        device: str = None,
        seed: int = 42,
        kl_beta: float = 1.0,  # optionales β für KL (Annealing/Temperierung)
        clip_log_var: tuple | None = (
            -10.0,
            8.0,
        ),  # optionales Clamping der log-Varianz
    ):
        self.prior_std = prior_std
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.mc_train_samples = mc_train_samples
        self.mc_pred_samples = mc_pred_samples
        self.kl_beta = kl_beta
        self.clip_log_var = clip_log_var

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.model = None
        self._n_features = None
        self._dataset_size = None

    # --- utils ---
    def _to_tensor(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if y is not None:
            y = np.asarray(y, dtype=np.float32).reshape(-1)
            return torch.from_numpy(X), torch.from_numpy(y)
        return torch.from_numpy(X)

    # --- API ---
    def fit(self, X, y, verbose: bool = True):
        X_t, y_t = self._to_tensor(X, y)
        self._n_features = X_t.shape[1]
        self._dataset_size = len(X_t)

        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model = MeanVarLinearBNN(self._n_features, prior_std=self.prior_std).to(
            self.device
        )
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            loss_sum = nll_sum = kl_sum = 0.0
            n_total = 0

            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                # MC-Approx des erwarteten NLL und KL
                nll_mc = 0.0
                kl_mc = 0.0
                for _ in range(self.mc_train_samples):
                    mu, log_var, kl = self.model(xb, sample=True)
                    if self.clip_log_var is not None:
                        log_var = log_var.clamp(*self.clip_log_var)
                    nll_mc = nll_mc + gaussian_nll(yb, mu, log_var)
                    kl_mc = kl_mc + kl
                nll_mc /= self.mc_train_samples
                kl_mc /= self.mc_train_samples

                # ELBO-Minimierung: NLL + β * KL/N
                loss = nll_mc + self.kl_beta * (kl_mc / self._dataset_size)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                bsz = xb.size(0)
                n_total += bsz
                loss_sum += loss.item() * bsz
                nll_sum += nll_mc.item() * bsz
                kl_sum += (kl_mc.item() / self._dataset_size) * bsz

            if verbose:
                print(
                    f"Epoch {epoch:02d} | Loss {loss_sum / n_total:.4f} | NLL {nll_sum / n_total:.4f} | KL/N {kl_sum / n_total:.4f}"
                )

        return self

    @torch.no_grad()
    def predict(self, X, return_uncertainties: bool = True):
        if self.model is None:
            raise RuntimeError("Model is not trained. Call fit(X, y) first.")
        self.model.eval()

        X_t = self._to_tensor(X).to(self.device)
        # Bei sehr großen X ggf. in Batches vorhersagen:
        if X_t.dim() == 1:
            X_t = X_t.unsqueeze(0)

        # MC-Samples für prädiktive Zerlegung
        mean, total_var, alea_var, epis_var = self.model.predict_mc(
            X_t, num_samples=self.mc_pred_samples
        )

        mean = mean.cpu().numpy()
        if not return_uncertainties:
            return mean

        return (
            mean,
            {
                "total_variance": total_var.cpu().numpy(),
                "aleatoric_variance": alea_var.cpu().numpy(),
                "epistemic_variance": epis_var.cpu().numpy(),
            },
        )


# ---------------- Minimalbeispiel ----------------
if __name__ == "__main__":
    # synthetische Tabulardaten
    rng = np.random.default_rng(0)
    N, D = 3000, 8
    X = rng.normal(size=(N, D)).astype(np.float32)
    w_true = rng.normal(size=D)
    # heteroskedastische Varianz abhängig von Feature 0
    log_var_y = np.log(
        np.clip(0.2 + 0.3 * (X[:, 0] > 0).astype(np.float32), 0.05, None)
    )
    y = X @ w_true + rng.normal(scale=np.exp(0.5 * log_var_y))

    # Train/Test
    idx = rng.permutation(N)
    tr, te = idx[:2400], idx[2400:]
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    model = BNNRegressor(epochs=40, mc_pred_samples=300, kl_beta=1.0).fit(Xtr, ytr)

    mean, unc = model.predict(Xte, return_uncertainties=True)
    print("Pred mean (first 5):", mean[:5])
    print("Aleatoric var (first 5):", unc["aleatoric_variance"][:5])
    print("Epistemic var (first 5):", unc["epistemic_variance"][:5])

from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class BNNModel(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, p_drop: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
        )
        # 2 outputs -> mu, log_var
        self.out_layer = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        h = self.backbone(x)
        out = self.out_layer(h)
        mu = out[:, 0:1]
        log_var = out[
            :, 1:2
        ]  # .clamp(min=-10.0, max=5.0)  # clamp or softplus for numerical stability
        return mu, log_var


def nll_loss(pred: Tuple[torch.Tensor, torch.Tensor], y: torch.Tensor) -> torch.Tensor:
    mu, log_var = pred
    return torch.mean(0.5 * (torch.exp(-log_var) * (y - mu) ** 2 + log_var))


class MCDropoutBNN:
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        p_drop: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        batch_size: int = 128,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.p_drop = p_drop
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = BNNModel(
            input_dim=input_dim, hidden_dim=hidden_dim, p_drop=p_drop
        ).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = nll_loss(pred, batch_y)
                loss.backward()
                optimizer.step()

    def predict_with_uncertainties(
        self, X_test: np.ndarray, n_mc_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        mus, vars = [], []
        self.model.train()  # enables mc dropout to sample epistemic uncertainty
        with torch.no_grad():
            for i in range(n_mc_samples):
                mu_i, log_var_i = self.model(X_test)
                mus.append(mu_i)
                vars.append(torch.exp(log_var_i))

        mus = torch.stack(mus, dim=0)  # [T, N, 1]
        vars = torch.stack(vars, dim=0)  # [T, N, 1]

        mean = mus.mean(dim=0)
        epistemic = mus.var(dim=0, unbiased=False)
        aleatoric = vars.mean(dim=0)

        return (
            mean.squeeze(-1).cpu().numpy(),
            epistemic.squeeze(-1).cpu().numpy(),
            aleatoric.squeeze(-1).cpu().numpy(),
        )

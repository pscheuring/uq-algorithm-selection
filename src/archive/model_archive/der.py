from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.base_meta_model import BaseModel


class DERModel(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.der_layer = DERLayer()

    def forward(self, x):
        raw_out = self.backbone(x)
        return self.der_layer(raw_out)


class DERLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        gamma = x[:, 0]
        nu = nn.functional.softplus(x[:, 1])
        alpha = nn.functional.softplus(x[:, 2]) + 1.0
        beta = nn.functional.softplus(x[:, 3])
        return torch.stack((gamma, nu, alpha, beta), dim=1)


def der_loss(pred, y, coeff=1.0):
    gamma, nu, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    error = gamma - y
    omega = 2.0 * beta * (1.0 + nu)

    return torch.mean(
        0.5 * torch.log(torch.tensor(np.pi) / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(error**2 * nu + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
        + coeff * torch.abs(error) * (2.0 * nu + alpha)
    )


# def scan_model(model, x_min=-4, x_max=4, n_points=300, device="cpu"):
#     x = torch.linspace(x_min, x_max, n_points).unsqueeze(1).to(device)
#     model.eval()
#     with torch.no_grad():
#         y = model(x).cpu().numpy()
#     return x.cpu().numpy(), y


class DER(BaseModel):
    def __init__(
        self,
        hidden_dim,
        lr,
        epochs,
        coeff,
        batch_size,
        input_dim: int = 2,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.coeff = coeff
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DERModel(input_dim=input_dim, hidden_dim=hidden_dim).to(
            self.device
        )

    def fit(self, X_train, y_train):
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = der_loss(pred, batch_y, coeff=self.coeff)
                loss.backward()
                optimizer.step()
        # self.model.eval()
        # x_scan, y_scan = scan_model(
        #                         self.model,
        #                         x_min=-7,
        #                         x_max=7,
        #                         device=self.device
        #                                         )
        # self.x_scan = {'x': x_scan, 'y': y_scan}

    def predict_with_uncertainties(
        self, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
        - mean prediction (gamma)
        - epistemic uncertainty
        - aleatoric uncertainty
        """
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_test).cpu().numpy()

        gamma, nu, alpha, beta = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]

        # Proposed method by Meinert et al. (2023)
        aleatoric = np.sqrt(beta * (nu + 1) / nu / alpha)
        epistemic = 1.0 / np.sqrt(nu)

        # SOTA (State of the art) according to Meinert et al. (2023)
        # aleatoric = np.sqrt(beta / (alpha - 1))
        # epistemic = aleatoric / np.sqrt(nu)

        # Method by Amiri et al. (2020)
        # aleatoric = beta / (alpha - 1)
        # epistemic = beta / (nu * (alpha - 1))

        return gamma, epistemic, aleatoric

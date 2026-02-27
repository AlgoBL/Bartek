"""
autoencoder_factors.py — Autoencoder Latent Factor Model dla odkrywania
nieliniowych czynników ryzyka portfela.

Architektura:
  Encoder: N → 256 → 128 → k (k czynników ukrytych, k ≈ 8–16)
  Decoder: k → 128 → 256 → N
  Loss: MSE + λ||z||₁ (sparse autoencoder)

Zastosowania:
  1. Feature extraction — k czynników zamiast FF5
  2. Anomaly detection — wysoki błąd rekonstrukcji = anomalia ryzyka
  3. Compression — redukcja wymiarowości portfela

Referencje:
  Gu, Kelly & Xiu (2021) — "Autoencoder Asset Pricing Models"
  Rußwurm et al. (2020) — "Self-Attention for Raw Optical Satellite Time Series"
  Bai et al. (2019) — "Deep learning for financial time series prediction"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Soft dep: torch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch nie jest zainstalowany. "
        "Uruchom: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )


# ─── ARCHITEKTURA SIECI ───────────────────────────────────────────────────────

if _TORCH_AVAILABLE:

    class _Encoder(nn.Module):
        def __init__(self, n_assets: int, hidden_dim: int, n_factors: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_assets, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim // 2, n_factors),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)

    class _Decoder(nn.Module):
        def __init__(self, n_assets: int, hidden_dim: int, n_factors: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_factors, hidden_dim // 2),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, n_assets),
            )

        def forward(self, z: "torch.Tensor") -> "torch.Tensor":
            return self.net(z)

    class PortfolioAutoencoder(nn.Module):
        """
        Sparse Autoencoder dla odkrywania latentnych czynników ryzyka.

        Gu, Kelly & Xiu (2021): czynniki latentne AE przewyższają
        czynniki Fama-French w przewidywaniu przekrojowych zwrotów.
        """

        def __init__(
            self,
            n_assets: int,
            n_factors: int = 8,
            hidden_dim: int = 128,
            sparsity_lambda: float = 0.001,
        ):
            """
            Parameters
            ----------
            n_assets         : liczba aktywów (wymiar wejściowy)
            n_factors        : liczba czynników ukrytych (zazw. 4–16)
            hidden_dim       : rozmiar warstwy ukrytej
            sparsity_lambda  : waga regularyzacji L1 (sparse AE)
            """
            super().__init__()
            self.n_assets       = n_assets
            self.n_factors      = n_factors
            self.sparsity_lambda = sparsity_lambda
            self.encoder = _Encoder(n_assets, hidden_dim, n_factors)
            self.decoder = _Decoder(n_assets, hidden_dim, n_factors)

        def forward(
            self, x: "torch.Tensor"
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            z    = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat, z

        def loss(
            self, x: "torch.Tensor", x_hat: "torch.Tensor", z: "torch.Tensor"
        ) -> "torch.Tensor":
            """MSE + L1 sparsity regularization."""
            mse = nn.functional.mse_loss(x_hat, x)
            l1  = self.sparsity_lambda * torch.mean(torch.abs(z))
            return mse + l1

        def anomaly_score(self, x: "torch.Tensor") -> np.ndarray:
            """Reconstruction error per sample jako miara anomalii ryzyka."""
            self.eval()
            with torch.no_grad():
                x_hat, _ = self.forward(x)
                scores = torch.mean((x - x_hat) ** 2, dim=-1)
            return scores.numpy()


# ─── TRAINING WRAPPER ─────────────────────────────────────────────────────────

class AutoencoderFactorModel:
    """
    High-level wrapper do treningu i inferencji autoenkodera.

    Przykład użycia:
        ae = AutoencoderFactorModel(n_factors=8)
        ae.fit(returns_df, epochs=100)
        factors_df = ae.get_factor_loadings(returns_df)
        anomalies  = ae.detect_anomalies(returns_df)
    """

    def __init__(
        self,
        n_factors:       int   = 8,
        hidden_dim:      int   = 128,
        sparsity_lambda: float = 0.001,
        lr:              float = 1e-3,
        batch_size:      int   = 64,
        device:          str   = "cpu",
    ):
        self.n_factors       = n_factors
        self.hidden_dim      = hidden_dim
        self.sparsity_lambda = sparsity_lambda
        self.lr              = lr
        self.batch_size      = batch_size
        self.device          = device
        self._model          = None
        self._n_assets       = None
        self._mean: np.ndarray | None = None
        self._std:  np.ndarray | None = None
        self._trained        = False

    def _prepare_data(self, returns_df: pd.DataFrame) -> "torch.Tensor":
        """Normalizuje dane i konwertuje do Tensor."""
        X = returns_df.values.astype(np.float32)
        if self._mean is None:
            self._mean = X.mean(axis=0)
            self._std  = X.std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std
        return torch.tensor(X_norm, dtype=torch.float32)

    def fit(
        self,
        returns_df: pd.DataFrame,
        epochs: int = 100,
        verbose: bool = False,
    ) -> list[float]:
        """
        Trenuje autoenkoder na danych historycznych zwrotów.

        Parameters
        ----------
        returns_df : pd.DataFrame (T, n_assets)
        epochs     : liczba epok treningowych
        verbose    : czy wypisywać loss co 10 epok

        Returns
        -------
        lista wartości loss per epoka
        """
        if not _TORCH_AVAILABLE:
            logger.error("PyTorch niedostępny — trening niemożliwy")
            return []

        self._n_assets = returns_df.shape[1]
        self._model    = PortfolioAutoencoder(
            n_assets=self._n_assets,
            n_factors=self.n_factors,
            hidden_dim=self.hidden_dim,
            sparsity_lambda=self.sparsity_lambda,
        ).to(self.device)

        X_tensor = self._prepare_data(returns_df).to(self.device)
        dataset  = TensorDataset(X_tensor)
        loader   = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self._model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        loss_history = []
        self._model.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                x_hat, z = self._model(batch)
                loss = self._model.loss(batch, x_hat, z)
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            avg_loss = epoch_loss / max(len(loader), 1)
            loss_history.append(avg_loss)

            if verbose and epoch % 10 == 0:
                logger.info(f"AE Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.6f}")

        self._trained = True
        logger.info(
            f"Autoencoder fit: {self._n_assets} aktywów → "
            f"{self.n_factors} czynników | final loss={loss_history[-1]:.6f}"
        )
        return loss_history

    def get_factor_loadings(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Zwraca latentne czynniki ryzyka dla każdego okresu.

        Returns
        -------
        pd.DataFrame (T, n_factors) — wartości czynników ukrytych
        """
        if not self._trained or not _TORCH_AVAILABLE:
            return pd.DataFrame()

        self._model.eval()
        X_tensor = self._prepare_data(returns_df).to(self.device)
        with torch.no_grad():
            z = self._model.encoder(X_tensor)
        factor_cols = [f"Factor_{i+1}" for i in range(self.n_factors)]
        return pd.DataFrame(
            z.cpu().numpy(),
            index=returns_df.index,
            columns=factor_cols,
        )

    def detect_anomalies(
        self,
        returns_df: pd.DataFrame,
        threshold_pct: float = 0.95,
    ) -> pd.DataFrame:
        """
        Wykrywa anomalie ryzyka jako dni z wysokim błędem rekonstrukcji.

        Anomaly score > 95. percentyl → anomalia (potencjalny sygnał kryzysu).

        Returns
        -------
        pd.DataFrame z kolumnami: anomaly_score, is_anomaly, threshold
        """
        if not self._trained or not _TORCH_AVAILABLE:
            return pd.DataFrame()

        self._model.eval()
        X_tensor = self._prepare_data(returns_df).to(self.device)
        scores   = self._model.anomaly_score(X_tensor)

        threshold = float(np.percentile(scores, threshold_pct * 100))
        result = pd.DataFrame({
            "anomaly_score": scores,
            "is_anomaly":    scores > threshold,
            "threshold":     threshold,
        }, index=returns_df.index)
        return result

    def get_reconstruction_quality(self, returns_df: pd.DataFrame) -> dict:
        """
        Ocena jakości rekonstrukcji (R² per aktywo i total).

        Returns
        -------
        dict z: r2_per_asset, r2_total, mse_total, n_factors
        """
        if not self._trained or not _TORCH_AVAILABLE:
            return {"error": "Model niezainicjalizowany."}

        self._model.eval()
        X_tensor = self._prepare_data(returns_df).to(self.device)
        with torch.no_grad():
            x_hat, z = self._model(X_tensor)

        X_np     = X_tensor.cpu().numpy()
        X_hat_np = x_hat.cpu().numpy()

        ss_res = np.sum((X_np - X_hat_np)**2, axis=0)
        ss_tot = np.sum((X_np - X_np.mean(axis=0))**2, axis=0) + 1e-10
        r2_per = 1 - ss_res / ss_tot

        return {
            "r2_per_asset": dict(zip(returns_df.columns, r2_per.tolist())),
            "r2_total":     float(np.mean(r2_per)),
            "mse_total":    float(np.mean((X_np - X_hat_np)**2)),
            "n_factors":    self.n_factors,
            "n_assets":     self._n_assets,
        }

    def compare_with_ff5(
        self,
        returns_df: pd.DataFrame,
        factor_returns_df: pd.DataFrame,
    ) -> dict:
        """
        Porównuje R² autoenkodera vs klasycznej regresji FF5.

        Returns
        -------
        dict z: ae_r2, ff5_r2, ae_vs_ff5_improvement
        """
        from scipy.linalg import lstsq

        ae_quality = self.get_reconstruction_quality(returns_df)
        ae_r2      = ae_quality.get("r2_total", 0.0)

        # FF5 OLS R² (uproszczone)
        common_idx = returns_df.index.intersection(factor_returns_df.index)
        if len(common_idx) < 30:
            return {"error": "Za mało wspólnych obserwacji FF5 vs zwroty."}

        Y = returns_df.loc[common_idx].values        # (T, n)
        X = factor_returns_df.loc[common_idx].values
        X = np.column_stack([np.ones(len(X)), X])    # dodaj stałą

        r2_list = []
        for col in range(Y.shape[1]):
            b, _, _, _ = lstsq(X, Y[:, col])
            resid = Y[:, col] - X @ b
            r2    = 1 - np.var(resid) / (np.var(Y[:, col]) + 1e-10)
            r2_list.append(float(r2))

        ff5_r2 = float(np.mean(r2_list))

        return {
            "ae_r2":               ae_r2,
            "ff5_r2":              ff5_r2,
            "ae_advantage_pct":    (ae_r2 - ff5_r2) * 100,
            "n_ae_factors":        self.n_factors,
            "n_ff5_factors":       X.shape[1] - 1,
        }

    def save(self, path: str) -> None:
        """Zapisuje model PyTorch."""
        if not _TORCH_AVAILABLE or not self._trained:
            return
        import torch
        torch.save({
            "model_state": self._model.state_dict(),
            "n_assets":    self._n_assets,
            "n_factors":   self.n_factors,
            "hidden_dim":  self.hidden_dim,
            "mean":        self._mean,
            "std":         self._std,
        }, path)

    def load(self, path: str) -> None:
        """Wczytuje model PyTorch."""
        if not _TORCH_AVAILABLE:
            return
        import torch
        checkpoint     = torch.load(path, map_location=self.device)
        self._n_assets = checkpoint["n_assets"]
        self.n_factors = checkpoint["n_factors"]
        self._mean     = checkpoint["mean"]
        self._std      = checkpoint["std"]
        self._model    = PortfolioAutoencoder(
            n_assets=self._n_assets,
            n_factors=self.n_factors,
            hidden_dim=checkpoint["hidden_dim"],
        ).to(self.device)
        self._model.load_state_dict(checkpoint["model_state"])
        self._trained = True

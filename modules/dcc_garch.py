"""
dcc_garch.py — Dynamic Conditional Correlation GARCH(1,1).

Implementuje model DCC-GARCH (Engle 2002) dla dynamicznych korelacji
warunkowych między aktywami portfela. Zastępuje statyczny parametr θ
kopuły archimedejskiej zmienną macierzą korelacji R_t.

Kluczowe założenia:
  - Univariate GARCH(1,1) dla każdego aktywa: h_{i,t} następnie DCC
  - DCC: Q_t = (1-a-b)Q̄ + a z_{t-1}z'_{t-1} + b Q_{t-1}
  - Korelacja warunkowa: R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}

Referencje:
  Engle (2002) — "Dynamic Conditional Correlations"
  Engle & Sheppard (2001) — "Theoretical and Empirical Properties of DCC"
  Tse & Tsui (2002) — alternatywna parametryzacja
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from modules.logger import setup_logger

logger = setup_logger(__name__)


class DCCGARCHModel:
    """
    Dynamic Conditional Correlation GARCH(1,1) model.

    Parametry GARCH (alpha_g, beta_g) i DCC (a_dcc, b_dcc) mogą być
    podane ręcznie lub estymowane z danych przez fit().

    Warunek stacjonarności: alpha_g + beta_g < 1 (GARCH), a_dcc + b_dcc < 1 (DCC).
    """

    def __init__(
        self,
        alpha_g: float = 0.10,
        beta_g:  float = 0.85,
        a_dcc:   float = 0.04,
        b_dcc:   float = 0.92,
    ):
        self.alpha_g = alpha_g
        self.beta_g  = beta_g
        self.a_dcc   = a_dcc
        self.b_dcc   = b_dcc

        # Estymowane przez fit()
        self._Q_bar: np.ndarray | None = None  # długookresowa macierz korelacji
        self._n_assets: int = 0
        self._garch_omegas: np.ndarray | None = None   # GARCH omega per asset
        self._unconditional_vars: np.ndarray | None = None
        self._fitted = False

    # ─── 1. UNIVARIATE GARCH ─────────────────────────────────────────────────

    def _fit_garch_series(
        self, returns: np.ndarray
    ) -> tuple[float, float, float, float]:
        """
        Dopasowuje GARCH(1,1) do pojedynczego szeregu zwrotów.

        Returns: (omega, alpha, beta, uncond_var)
        """
        T = len(returns)
        sigma2_0 = np.var(returns)

        def garch_log_likelihood(params):
            omega_p, alpha_p, beta_p = params
            if omega_p <= 0 or alpha_p < 0 or beta_p < 0 or alpha_p + beta_p >= 1:
                return 1e10
            h = np.zeros(T)
            h[0] = sigma2_0
            for t in range(1, T):
                h[t] = omega_p + alpha_p * returns[t-1]**2 + beta_p * h[t-1]
            h = np.maximum(h, 1e-10)
            ll = -0.5 * np.sum(np.log(h) + returns**2 / h)
            return -ll  # minimize → negative log-likelihood

        res = minimize(
            garch_log_likelihood,
            x0=[sigma2_0 * 0.05, 0.10, 0.80],
            method="L-BFGS-B",
            bounds=[(1e-8, None), (1e-6, 0.5), (1e-6, 0.99)],
        )
        omega_hat, alpha_hat, beta_hat = res.x
        uncond_var = omega_hat / max(1 - alpha_hat - beta_hat, 1e-6)
        return float(omega_hat), float(alpha_hat), float(beta_hat), float(uncond_var)

    def _compute_garch_variance(
        self,
        returns: np.ndarray,
        omega: float,
        alpha: float,
        beta: float,
    ) -> np.ndarray:
        """Oblicza warunkową wariancję GARCH(1,1) dla szeregu."""
        T = len(returns)
        h = np.zeros(T)
        h[0] = np.var(returns)
        for t in range(1, T):
            h[t] = omega + alpha * returns[t-1]**2 + beta * h[t-1]
        return np.maximum(h, 1e-10)

    # ─── 2. FIT ───────────────────────────────────────────────────────────────

    def fit(self, returns_df: pd.DataFrame) -> "DCCGARCHModel":
        """
        Estymacja parametrów univariate GARCH i macierzy Q̄ z danych historycznych.

        Parameters
        ----------
        returns_df : pd.DataFrame (T, n_assets) — dzienne zwroty
        """
        n = returns_df.shape[1]
        self._n_assets = n
        rets = returns_df.values  # (T, n)

        # 1. Univariate GARCH dla każdego aktywa
        garch_params = []
        std_residuals = np.zeros_like(rets)  # standaryzowane residua z_t

        for i in range(n):
            r_i = rets[:, i]
            omega_i, alpha_i, beta_i, _ = self._fit_garch_series(r_i)
            h_i = self._compute_garch_variance(r_i, omega_i, alpha_i, beta_i)
            std_residuals[:, i] = r_i / np.sqrt(h_i)
            garch_params.append((omega_i, alpha_i, beta_i))

        self._garch_params = garch_params

        # 2. Q̄ = długookresowa korelacja standaryzowanych residuów
        self._Q_bar = np.corrcoef(std_residuals.T)
        np.fill_diagonal(self._Q_bar, 1.0)
        self._Q_bar += np.eye(n) * 1e-6  # regularyzacja

        # 3. Unconditional variances per asset
        self._unconditional_vars = np.var(rets, axis=0)
        self._std_residuals_hist = std_residuals
        self._fitted = True

        logger.info(
            f"DCC-GARCH fit: {n} aktywów, "
            f"a_dcc={self.a_dcc}, b_dcc={self.b_dcc}"
        )
        return self

    # ─── 3. SIMULATE ──────────────────────────────────────────────────────────

    def simulate_paths(
        self,
        n_sims: int,
        n_days: int,
        crash_regime_days: list[int] | None = None,
        crash_dcc_multiplier: float = 3.0,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Symuluje wielowymiarowe zwroty z dynamiczną macierzą korelacji DCC.

        Parameters
        ----------
        n_sims                : liczba symulacji
        n_days                : liczba dni
        crash_regime_days     : lista dni, gdzie a_dcc jest mnożona × crash_multiplier
                                (symuluje kontagion w krachu)
        crash_dcc_multiplier  : jak silnie korelacje eksplodują w krachu

        Returns
        -------
        (n_sims, n_days, n_assets) — macierz zwrotów
        """
        if not self._fitted:
            logger.warning("DCC-GARCH: model nie był fittowany — używam domyślnych parametrów")
            n = 2
            self._n_assets = n
            self._Q_bar = np.eye(n)
            self._garch_params = [(1e-4, 0.10, 0.85)] * n
            self._unconditional_vars = np.array([0.0001, 0.0001])

        if seed is not None:
            np.random.seed(seed)

        n = self._n_assets
        Q_bar = self._Q_bar
        crash_set = set(crash_regime_days or [])

        all_paths = np.zeros((n_sims, n_days, n))  # output

        for sim in range(n_sims):
            # Initialize GARCH variances at unconditional level
            h_t   = self._unconditional_vars.copy()      # (n,) current GARCH var
            Q_t   = Q_bar.copy()                          # (n, n) DCC quasi-corr
            prev_z = np.zeros(n)                          # z_{t-1}

            for day in range(n_days):
                # ── Aktualna a_dcc (crisis amplification) ─
                a_eff = self.a_dcc
                if day in crash_set:
                    a_eff = min(self.a_dcc * crash_dcc_multiplier, 1 - self.b_dcc - 1e-4)

                # ── DCC update: Q_t ────────────────────────────────────────────
                Q_t = (
                    (1 - a_eff - self.b_dcc) * Q_bar
                    + a_eff * np.outer(prev_z, prev_z)
                    + self.b_dcc * Q_t
                )
                # Normalize → correlation matrix R_t
                q_diag_sqrt = np.sqrt(np.maximum(np.diag(Q_t), 1e-8))
                R_t = Q_t / np.outer(q_diag_sqrt, q_diag_sqrt)
                np.fill_diagonal(R_t, 1.0)
                R_t = np.clip(R_t, -0.999, 0.999)

                # ── Cholesky → correlated standard normals ────────────────────
                try:
                    L_t = np.linalg.cholesky(R_t + np.eye(n) * 1e-7)
                except np.linalg.LinAlgError:
                    L_t = np.eye(n)

                epsilon = L_t @ np.random.standard_normal(n)   # correlated z

                # ── GARCH variance update ────────────────────────────────────
                for i, (omega_i, alpha_i, beta_i) in enumerate(self._garch_params):
                    h_t[i] = (
                        omega_i
                        + alpha_i * (epsilon[i] * np.sqrt(h_t[i])) ** 2
                        + beta_i * h_t[i]
                    )
                h_t = np.maximum(h_t, 1e-8)

                # ── Returns = epsilon * sqrt(h_t) ─────────────────────────────
                returns_t = epsilon * np.sqrt(h_t)
                all_paths[sim, day, :] = returns_t

                # Standaryzowane residua dla następnego DCC kroku
                prev_z = epsilon  # z_t = epsilon / sqrt(h_t) zanualizowane

        return all_paths

    # ─── 4. CORRELATION SERIES ────────────────────────────────────────────────

    def get_conditional_correlations(
        self,
        returns_df: pd.DataFrame,
    ) -> list[np.ndarray]:
        """
        Oblicza DCC macierze korelacji R_t dla każdego okresu w historii.
        Do wizualizacji ewolucji korelacji.

        Returns
        -------
        list[np.ndarray] — lista macierzy R_t (T, n, n)
        """
        if not self._fitted:
            self.fit(returns_df)

        rets = returns_df.values
        T, n = rets.shape
        Q_bar = self._Q_bar

        # Standaryzowane residua z historycznego GARCH
        z = self._std_residuals_hist
        R_series: list[np.ndarray] = []
        Q_t = Q_bar.copy()

        for t in range(T):
            q_diag_sqrt = np.sqrt(np.maximum(np.diag(Q_t), 1e-8))
            R_t = Q_t / np.outer(q_diag_sqrt, q_diag_sqrt)
            np.fill_diagonal(R_t, 1.0)
            R_series.append(R_t.copy())

            if t < T - 1:
                prev_z = z[t]
                Q_t = (
                    (1 - self.a_dcc - self.b_dcc) * Q_bar
                    + self.a_dcc * np.outer(prev_z, prev_z)
                    + self.b_dcc * Q_t
                )

        return R_series

    def summary(self) -> dict:
        """Zwraca słownik z parametrami modelu."""
        return {
            "alpha_g_list": [p[1] for p in (self._garch_params or [])],
            "beta_g_list":  [p[2] for p in (self._garch_params or [])],
            "a_dcc":        self.a_dcc,
            "b_dcc":        self.b_dcc,
            "n_assets":     self._n_assets,
            "Q_bar_diag":   np.diag(self._Q_bar).tolist() if self._Q_bar is not None else [],
            "fitted":       self._fitted,
        }

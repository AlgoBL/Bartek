"""
garch_midas.py — GARCH-MIDAS: Mixed-Data Sampling Volatility
Łączy krótkoterminową dynamikę GARCH z długoterminowymi makro trendami.

Ref: Engle, R.F. & Rangel, J.G. (2008) "The Spline GARCH Model for Unconditional
     Volatility and its Global Macroeconomic Causes", RFS 21(3).
     Conrad, C. & Loch, K. (2014) — rozszerzenie na wiele zmiennych makro.

Architektura:
    σ²(t) = τ(t) · g(t)
    g(t):  dzienna składowa GARCH(1,1)  — szybka (intraday volatility)
    τ(t):  długoterminowa składowa MIDAS — makro (PMI, CPI, claims, M2)

Zastosowanie w projekcie:
    - Symulator: risky_vol = σ_MIDAS(t) zamiast stałego wejścia
    - Control Center: τ(t) jako wskaźnik reżimu makro-zmienności
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Optional
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  MIDAS KERNEL — Beta polynomial weights
# ═══════════════════════════════════════════════════════════════════════════════

def _beta_weights(m: int, w1: float = 1.0, w2: float = 5.0) -> np.ndarray:
    """
    Beta polynomial MIDAS weights (Ghysels, Santa-Clara & Valkanov 2004).

    φ_k = (k/m)^(w1-1) * (1-k/m)^(w2-1)  → normalized to sum to 1.

    m   : lag order (number of macro periods)
    w1, w2 : shape parameters (w2 >> 1 → down-weighting recent obs)
    """
    k = np.arange(1, m + 1, dtype=float)
    raw = (k / m) ** (w1 - 1) * (1 - k / m) ** (w2 - 1)
    raw = np.maximum(raw, 1e-12)
    return raw / raw.sum()


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  GARCH-MIDAS CORE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class GARCHMIDASEngine:
    """
    GARCH-MIDAS volatility decomposition.

    Parameters
    ----------
    m_lags   : int — liczba opóźnień MIDAS (np. 12 = 12 miesięcy makro)
    w1, w2   : float — parametry kształtu kernela Beta
    """

    def __init__(self, m_lags: int = 12, w1: float = 1.0, w2: float = 5.0):
        self.m_lags = m_lags
        self.w1 = w1
        self.w2 = w2
        # Fitted parameters
        self.alpha_ = 0.05   # GARCH alpha (laggged eps^2)
        self.beta_  = 0.85   # GARCH beta  (persistence)
        self.gamma_ = 0.10   # MIDAS loading on macro variance
        self.theta_ = 1.0    # macro level scaling
        self.fitted_ = False

    # ── 2a. τ(t): Long-run Volatility from MIDAS ──────────────────────────────

    def compute_tau(self, macro_var: np.ndarray) -> np.ndarray:
        """
        Compute the long-run variance component τ(t) using MIDAS kernel.

        macro_var : ndarray of macro-frequency variance proxy (e.g. monthly
                    rolling variance of returns, RV_t = var(r_{t-21:t})*252)
        Returns   : τ(t) matched to same length as macro_var
        """
        w = _beta_weights(self.m_lags, self.w1, self.w2)
        n = len(macro_var)
        tau = np.zeros(n)
        for t in range(n):
            # Weighted sum of lagged macro variance
            lags = macro_var[max(0, t - self.m_lags + 1): t + 1]
            # Align weights (shorter at beginning of series)
            w_used = w[-len(lags):]
            w_used = w_used / w_used.sum()
            tau[t] = self.theta_ + self.gamma_ * np.dot(w_used, lags)
        return np.maximum(tau, 1e-8)

    # ── 2b. g(t): Short-run GARCH(1,1) component ──────────────────────────────

    def compute_garch_component(
        self, returns: np.ndarray, tau: np.ndarray
    ) -> np.ndarray:
        """
        Estimate g(t) via GARCH(1,1) with σ² normalized by τ(t).

        σ²(t) = τ(t) · g(t)
        g(t)  = (1-α-β) + α·(r_{t-1}/√τ_{t-1})² + β·g_{t-1}
        """
        n = len(returns)
        g = np.ones(n)
        omega = 1.0 - self.alpha_ - self.beta_
        for t in range(1, n):
            eps2 = (returns[t - 1] ** 2) / max(tau[t - 1], 1e-8)
            g[t] = omega + self.alpha_ * eps2 + self.beta_ * g[t - 1]
            g[t] = max(g[t], 0.0001)
        return g

    # ── 2c. Full Fit + Decomposition ──────────────────────────────────────────

    def fit_from_returns(
        self,
        daily_returns: pd.Series,
        macro_series: Optional[pd.Series] = None,
    ) -> dict:
        """
        Fit GARCH-MIDAS from daily returns and optional macro series.

        If macro_series is None → uses 21-day rolling realized variance of
        returns as the MIDAS regressor (a self-contained approximation).

        Returns
        -------
        dict with: tau (long-run vol trend), g (short-run GARCH),
                   total_vol (√(τ·g)), annualized_vol_current,
                   macro_regime (low/medium/high)
        """
        r = daily_returns.dropna().values

        # ── Macro variance proxy ───────────────────────────────────────────────
        if macro_series is not None:
            # External macro (PMI, CPI, Claims) → normalize to vol units
            macro_aligned = macro_series.reindex(daily_returns.dropna().index).ffill()
            macro_vals = macro_aligned.values
            # Normalize: convert to realized variance proxy
            macro_rv = (macro_vals - macro_vals.mean()) / (macro_vals.std() + 1e-8)
            macro_rv = (macro_rv ** 2) * (0.15 ** 2 / 252)  # scale to daily var
        else:
            # Self-contained: 21-day rolling realized variance
            series = daily_returns.dropna()
            macro_rv = series.rolling(21).var().fillna(series.var()).values

        # ── Simple MLE estimation (log-likelihood) ─────────────────────────────
        def neg_ll(params):
            a, b, g_load, theta = params
            if a <= 0 or b <= 0 or a + b >= 0.999 or g_load < 0 or theta <= 0:
                return 1e10
            self.alpha_, self.beta_, self.gamma_, self.theta_ = a, b, g_load, theta
            tau = self.compute_tau(macro_rv)
            g_t = self.compute_garch_component(r, tau)
            sigma2 = tau * g_t
            sigma2 = np.maximum(sigma2, 1e-10)
            ll = -0.5 * np.sum(np.log(sigma2) + r ** 2 / sigma2)
            return -ll

        # Optimize
        x0 = [0.05, 0.85, 0.10, np.var(r) * 252]
        bounds = [(0.001, 0.5), (0.001, 0.999), (0.001, 2.0), (1e-6, 10.0)]
        try:
            res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 200, "ftol": 1e-9})
            if res.success:
                self.alpha_, self.beta_, self.gamma_, self.theta_ = res.x
        except Exception:
            pass  # keep defaults

        # ── Compute final paths ────────────────────────────────────────────────
        tau = self.compute_tau(macro_rv)
        g_t = self.compute_garch_component(r, tau)
        total_var = tau * g_t
        total_vol = np.sqrt(np.maximum(total_var, 1e-10))

        # Annualized
        ann_vol_current = float(total_vol[-1]) * np.sqrt(252)
        ann_tau_current = float(np.sqrt(tau[-1])) * np.sqrt(252)
        persistence = self.alpha_ + self.beta_
        half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf

        # Macro regime based on τ relative to its own distribution
        tau_pct = (tau[-1] - tau.min()) / (tau.max() - tau.min() + 1e-10)
        macro_regime = "🔴 Wysoka" if tau_pct > 0.7 else ("🟡 Podwyższona" if tau_pct > 0.4 else "🟢 Niska")

        self.fitted_ = True
        return {
            "tau":                   tau,
            "g":                     g_t,
            "total_vol":             total_vol,
            "macro_rv":              macro_rv,
            "alpha":                 self.alpha_,
            "beta":                  self.beta_,
            "gamma":                 self.gamma_,
            "theta":                 self.theta_,
            "ann_vol_current":       ann_vol_current,
            "ann_tau_current":       ann_tau_current,
            "persistence":           persistence,
            "half_life_days":        half_life,
            "macro_regime":          macro_regime,
            "tau_pct":               float(tau_pct),
            "index":                 daily_returns.dropna().index,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  CONVENIENCE FUNCTION for Symulator Integration
# ═══════════════════════════════════════════════════════════════════════════════

def get_midas_adjusted_vol(
    daily_returns: pd.Series,
    base_vol: float = 0.20,
    macro_series: Optional[pd.Series] = None,
    m_lags: int = 12,
) -> float:
    """
    Returns GARCH-MIDAS adjusted annualized volatility for use in Symulator.

    If returns < 60 observations → falls back to base_vol.
    The output blends the fitted current σ with the user's base_vol input:
        adjusted = 0.5 * fitted_current + 0.5 * base_vol

    Parameters
    ----------
    daily_returns : pd.Series of daily returns
    base_vol      : user-specified annual volatility (fallback)
    macro_series  : optional macro signal (e.g., claims, PMI)
    m_lags        : MIDAS kernel lag order

    Returns
    -------
    float — adjusted annualized volatility
    """
    if len(daily_returns.dropna()) < 63:
        return base_vol

    try:
        engine = GARCHMIDASEngine(m_lags=m_lags)
        result = engine.fit_from_returns(daily_returns, macro_series)
        fitted_vol = result["ann_vol_current"]
        # Blend: trust the model 50%, keep user's prior 50%
        return 0.5 * fitted_vol + 0.5 * base_vol
    except Exception:
        return base_vol


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PLOTLY VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_garch_midas_decomposition(result: dict, title: str = "GARCH-MIDAS Volatility") -> "go.Figure":
    """
    Visualizes the τ(t) long-run and √g(t) short-run volatility components.
    """
    import plotly.graph_objects as go

    idx   = result.get("index")
    tau   = result["tau"]
    g     = result["g"]
    total = result["total_vol"]

    # Annualize
    tau_vol   = np.sqrt(np.maximum(tau, 1e-10)) * np.sqrt(252) * 100
    g_vol     = np.sqrt(np.maximum(g, 1e-10)) * np.sqrt(252) * 100
    total_vol = total * np.sqrt(252) * 100

    x = idx if idx is not None else np.arange(len(tau))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=total_vol, mode="lines", name="σ_MIDAS (Total)",
        line=dict(color="#00e676", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=tau_vol, mode="lines", name="√τ (Long-run Macro)",
        line=dict(color="#3498db", width=1.5, dash="dot"),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=g_vol, mode="lines", name="√g (Short-run GARCH)",
        line=dict(color="#f39c12", width=1, dash="dash"),
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="white")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.7)",
        yaxis_title="Zmienność roczna (%)",
        xaxis_title="",
        height=360,
        font=dict(color="white", family="Inter"),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        margin=dict(l=50, r=20, t=40, b=60),
    )
    return fig

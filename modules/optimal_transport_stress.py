"""
optimal_transport_stress.py — Optimal Transport Stress Testing
==============================================================
Generuje realistyczne scenariusze stresowe przez interpolację rozkładów
przy użyciu Optimal Transport (1D Wasserstein barycentrum).

Zamiast prostego przeskalowania parametrów:
  - Morphuje rozkład "normalny" w rozkład "kryzys 2008/2020"
  - Zachowuje strukturę ogonów i korelacji
  - Poziom stresu 0.0–1.0 daje ciągłe przejście między reżimami

Referencje:
  Villani (2009) — "Optimal Transport: Old and New", Springer.
  Peyré & Cuturi (2019) — "Computational Optimal Transport", FTML.
  Hallin et al. (2021) — "Multivariate Quantiles and Multiple-Output Regression
                          Quantiles: From L1 Optimization to Halfspace Depth"
  Blanchet & Murthy (2019) — "Quantifying Distributional Model Risk via
                               Optimal Transport", Math. OR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional
from modules.logger import setup_logger

logger = setup_logger(__name__)


# ─── Historyczne scenariusze kryzysowe (zwroty dzienne) ──────────────────────
# Parametry rozkładu (μ, σ, skewness_factor, kurt_factor) dla każdego kryzysu
CRISIS_SCENARIOS: dict[str, dict] = {
    "GFC 2008-09": {
        "label":    "Global Financial Crisis 2008-09",
        "mu":       -0.0025,      # ~-63% roczny zwrot
        "sigma":    0.035,         # ~55% roczna vol
        "tail_alpha": 1.5,         # α-stable tail index (Lévy heavy tail)
        "max_drawdown": -0.568,
        "color":    "#e74c3c",
        "description": "Kryzys subprime, upadek Lehman Brothers. Najgłębszy krach od Wielkiego Kryzysu.",
    },
    "COVID 2020": {
        "label":    "COVID-19 Crash 2020",
        "mu":       -0.0040,       # bardzo szybki spadek
        "sigma":    0.045,
        "tail_alpha": 1.3,
        "max_drawdown": -0.340,
        "color":    "#9b59b6",
        "description": "Najszybszy spadek -34% w historii S&P500 (22 dni handlowe).",
    },
    "Euro Crisis 2011": {
        "label":    "European Debt Crisis 2011",
        "mu":       -0.0008,
        "sigma":    0.018,
        "tail_alpha": 1.7,
        "max_drawdown": -0.19,
        "color":    "#e67e22",
        "description": "Kryzys zadłużeniowy Grecji/Włoch, stresy na rynkach europejskich.",
    },
    "Rate Shock 2022": {
        "label":    "Fed Rate Shock 2022",
        "mu":       -0.0012,
        "sigma":    0.022,
        "tail_alpha": 1.8,
        "max_drawdown": -0.25,
        "color":    "#f39c12",
        "description": "Najszybszy cykl podwyżek stóp Fed od 40 lat. Akcje -25%, obligacje -15%.",
    },
    "Flash Crash 2010": {
        "label":    "Flash Crash 2010",
        "mu":       -0.0005,
        "sigma":    0.025,
        "tail_alpha": 1.4,
        "max_drawdown": -0.09,
        "color":    "#1abc9c",
        "description": "Błyskawiczny krach algorytmiczny -10% w ciągu minut. Rynki wrróciły w godzinę.",
    },
}

NORMAL_MARKET: dict = {
    "label":    "Normal Market",
    "mu":       0.0004,       # ~10% roczny zwrot
    "sigma":    0.010,         # ~16% roczna vol
    "tail_alpha": 1.9,          # prawie normalny
    "color":    "#00e676",
}


# ─── 1. Generowanie zwrotów z rozkładu scenariusza ───────────────────────────

def _generate_scenario_returns(
    scenario: dict,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generuje n zwrotów z rozkładu scenariusza.
    Używa α-stable distribution dla heavy tails.
    """
    mu = scenario["mu"]
    sigma = scenario["sigma"]
    alpha_stable = scenario.get("tail_alpha", 1.8)

    if abs(alpha_stable - 2.0) < 0.1:
        # Prawie normalny
        return rng.normal(mu, sigma, n)

    # Chambers-Mallows-Stuck dla symmetric α-stable
    phi = rng.uniform(-np.pi / 2, np.pi / 2, n)
    w = rng.exponential(1.0, n)

    if abs(alpha_stable - 1.0) < 0.05:
        raw = np.tan(phi)
    else:
        part1 = np.sin(alpha_stable * phi) / (np.cos(phi) ** (1 / alpha_stable))
        part2 = (np.cos((1 - alpha_stable) * phi) / w) ** ((1 - alpha_stable) / alpha_stable)
        raw = part1 * part2

    # Skalowanie do (mu, sigma)
    raw_std = np.std(raw)
    if raw_std > 0:
        return mu + sigma * raw / raw_std
    return rng.normal(mu, sigma, n)


# ─── 2. 1D Optimal Transport (Wasserstein Interpolacja) ──────────────────────

def wasserstein_interpolation(
    returns_normal: np.ndarray,
    returns_crisis: np.ndarray,
    stress_level: float = 0.5,
) -> np.ndarray:
    """
    1D Wasserstein barycentrum (Optimal Transport interpolacja).

    Dla rozkładów 1D: W₂ barycentrum = liniowa interpolacja kwantyli.
    T*_{α}(x) = (1-α)·F₀⁻¹(p) + α·F₁⁻¹(p)   dla p ∈ [0,1]

    Gdzie:
    - F₀ = CDF rynku normalnego
    - F₁ = CDF scenariusza kryzysowego
    - α = stress_level ∈ [0, 1]

    Properties:
    - α=0 → rozkład normalny
    - α=1 → pełny kryzys
    - Zachowuje monotoniczność kwantylową

    Ref: Villani (2003) "Topics in Optimal Transportation", Thm 2.18
         Agueh & Carlier (2011) "Barycenters in the Wasserstein Space"

    Parameters
    ----------
    returns_normal : array zwrotów z normalnego rynku
    returns_crisis : array zwrotów z kryzysu
    stress_level   : 0.0=normal, 1.0=full crisis

    Returns
    -------
    array o długości len(returns_normal) — interpolowany rozkład
    """
    stress_level = float(np.clip(stress_level, 0.0, 1.0))
    n = len(returns_normal)

    # Kwantyle obu rozkładów
    quantile_probs = (np.arange(n) + 0.5) / n
    q_normal = np.quantile(returns_normal, quantile_probs)
    q_crisis = np.quantile(returns_crisis, quantile_probs)

    # Wasserstein barycentrum: interpolacja liniowa kwantyli
    q_stress = (1.0 - stress_level) * q_normal + stress_level * q_crisis

    # Losuj próbę z interpolowanego rozkładu (inverse CDF sampling)
    u = np.random.uniform(0, 1, n)
    # Sortujemy kwantyle → liniowa interpolacja
    stress_sample = np.interp(u, quantile_probs, q_stress)
    return stress_sample


# ─── 3. Główna klasa OT Stress Testing ───────────────────────────────────────

class OptimalTransportStressTester:
    """
    Stress Testing przez Optimal Transport.

    Pozwala:
    1. Wybrać scenariusz kryzysu (GFC, COVID, etc.)
    2. Ustawić poziom stresu 0.0–1.0
    3. Wygenerować zestaw stresowych ścieżek portfela
    4. Policzyć metryki ryzyka (VaR, CVaR, Max DD, Shortfall)

    Referencja: Blanchet & Murthy (2019) "Distributionally Robust OT Stress Tests"
    """

    def __init__(self, n_simulations: int = 2000, seed: int = 42):
        self.n_simulations = n_simulations
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # ─── Generowanie scenariuszy ──────────────────────────────────────────────

    def generate_stress_returns(
        self,
        portfolio_returns: pd.Series | np.ndarray,
        crisis_scenario: str = "GFC 2008-09",
        stress_level: float = 0.5,
        n_days: int | None = None,
    ) -> dict:
        """
        Generuje stresowe zwroty przez OT interpolację.

        Parameters
        ----------
        portfolio_returns : historyczne zwroty portfela (kalibracja rozkładu normalnego)
        crisis_scenario   : klucz z CRISIS_SCENARIOS
        stress_level      : 0=normal, 1=full crisis
        n_days            : liczba dni do zasymulowania (default: len(portfolio_returns))

        Returns
        -------
        dict z:
          stress_returns     : (n_sims, n_days) — stresowe zwroty
          normal_returns     : (n_sims, n_days) — normalne zwroty (benchmark)
          stress_level       : float
          crisis_scenario    : str
          scenario_params    : dict
        """
        if isinstance(portfolio_returns, pd.Series):
            hist_r = portfolio_returns.dropna().values
        else:
            hist_r = np.asarray(portfolio_returns, dtype=float)
            hist_r = hist_r[~np.isnan(hist_r)]

        if crisis_scenario not in CRISIS_SCENARIOS:
            logger.warning(f"Nieznany scenariusz: {crisis_scenario}. Dostępne: {list(CRISIS_SCENARIOS)}")
            crisis_scenario = "GFC 2008-09"

        scenario = CRISIS_SCENARIOS[crisis_scenario]
        n = n_days or max(len(hist_r), 252)

        # Generuj normalne zwroty (z historii lub parametrycznie)
        if len(hist_r) >= 60:
            mu_hist = float(hist_r.mean())
            sig_hist = float(hist_r.std())
            normal_base = self._rng.normal(mu_hist, sig_hist, (self.n_simulations, n))
        else:
            normal_base = self._rng.normal(NORMAL_MARKET["mu"], NORMAL_MARKET["sigma"],
                                           (self.n_simulations, n))

        # Generuj kryzysowe zwroty
        crisis_base = np.array([
            _generate_scenario_returns(scenario, n, self._rng)
            for _ in range(self.n_simulations)
        ])

        # OT interpolacja per symulacja
        stress_returns = np.zeros_like(normal_base)
        for i in range(self.n_simulations):
            stress_returns[i] = wasserstein_interpolation(
                normal_base[i], crisis_base[i], stress_level
            )

        return {
            "stress_returns":  stress_returns,
            "normal_returns":  normal_base,
            "stress_level":    stress_level,
            "crisis_scenario": crisis_scenario,
            "scenario_params": scenario,
            "n_simulations":   self.n_simulations,
            "n_days":          n,
        }

    # ─── Metryki ryzyka ze scenariusza ───────────────────────────────────────

    def compute_stress_metrics(
        self,
        stress_result: dict,
        initial_capital: float = 100.0,
        confidence_levels: list[float] = [0.95, 0.99, 0.999],
    ) -> dict:
        """
        Oblicza metryki ryzyka na stresowanych ścieżkach.

        Returns
        -------
        dict z:
          var_95/99/999       : VaR per poziom ufności
          cvar_95/99/999      : CVaR per poziom ufności
          max_drawdown_median : mediana max drawdown
          shortfall_5pct      : 5th percentyl finalnej wartości portfela
          stress_vs_normal    : porównanie metryk (stress / normal ratio)
        """
        stress_r = stress_result["stress_returns"]   # (n_sims, n_days)
        normal_r = stress_result["normal_returns"]

        def compute_portfolio_metrics(returns_matrix: np.ndarray, cap: float) -> dict:
            # Kumulatywna wartość portfela
            cum_returns = np.cumprod(1 + returns_matrix, axis=1) * cap
            final_values = cum_returns[:, -1]

            # Max Drawdown per ścieżka
            peaks = np.maximum.accumulate(cum_returns, axis=1)
            drawdowns = (cum_returns - peaks) / peaks
            max_dds = drawdowns.min(axis=1)

            # Finalne zwroty (całkowity)
            total_r = (final_values / cap) - 1.0

            # VaR / CVaR (na finalnych zwrotach)
            metrics = {"final_values": final_values, "total_returns": total_r}
            for cl in confidence_levels:
                var_q = np.percentile(total_r, (1 - cl) * 100)
                cvar_q = total_r[total_r <= var_q].mean() if (total_r <= var_q).any() else var_q
                cl_str = str(int(cl * 100))
                metrics[f"var_{cl_str}"] = float(var_q)
                metrics[f"cvar_{cl_str}"] = float(cvar_q)

            metrics["max_drawdown_median"] = float(np.median(max_dds))
            metrics["max_drawdown_5pct"] = float(np.percentile(max_dds, 5))
            metrics["shortfall_5pct"] = float(np.percentile(final_values, 5))
            metrics["expected_shortfall"] = float(final_values[final_values < np.percentile(final_values, 5)].mean())
            metrics["survival_rate"] = float((final_values > cap).mean())
            return metrics

        stress_metrics = compute_portfolio_metrics(stress_r, initial_capital)
        normal_metrics = compute_portfolio_metrics(normal_r, initial_capital)

        # Porównanie stress vs normal
        stress_vs = {}
        for k in ["var_95", "var_99", "cvar_95", "cvar_99", "max_drawdown_median"]:
            s = stress_metrics.get(k, 0.0)
            n = normal_metrics.get(k, 0.0)
            ratio = abs(s / n) if abs(n) > 1e-10 else 1.0
            stress_vs[k] = {
                "stress":   s,
                "normal":   n,
                "ratio":    ratio,
                "amplification_pct": (ratio - 1) * 100,
            }

        return {
            "stress":          stress_metrics,
            "normal":          normal_metrics,
            "comparison":      stress_vs,
            "scenario":        stress_result["crisis_scenario"],
            "stress_level":    stress_result["stress_level"],
        }

    # ─── Sweep przez poziomy stresu ───────────────────────────────────────────

    def stress_level_sweep(
        self,
        portfolio_returns: pd.Series,
        crisis_scenario: str = "GFC 2008-09",
        levels: list[float] | None = None,
        initial_capital: float = 100.0,
    ) -> pd.DataFrame:
        """
        Oblicza metryki ryzyka dla każdego poziomu stresu 0.0–1.0.
        Używane do wizualizacji 'stress dial'.

        Returns: DataFrame z columns [stress_level, var_95, cvar_95, max_dd, shortfall]
        """
        if levels is None:
            levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        rows = []
        for lvl in levels:
            result = self.generate_stress_returns(
                portfolio_returns, crisis_scenario, lvl, n_days=252,
            )
            metrics = self.compute_stress_metrics(result, initial_capital)
            rows.append({
                "stress_level":  lvl,
                "var_95":        metrics["stress"]["var_95"],
                "cvar_95":       metrics["stress"]["cvar_95"],
                "var_99":        metrics["stress"]["var_99"],
                "cvar_99":       metrics["stress"]["cvar_99"],
                "max_dd_median": metrics["stress"]["max_drawdown_median"],
                "shortfall_5pct":metrics["stress"]["shortfall_5pct"],
                "survival_rate": metrics["stress"]["survival_rate"],
            })

        return pd.DataFrame(rows)


# ─── 4. Plotly wizualizacja ───────────────────────────────────────────────────

def plot_ot_stress_paths(
    stress_result: dict,
    n_paths_to_show: int = 50,
    initial_capital: float = 100.0,
    title: str | None = None,
) -> go.Figure:
    """
    Wykres ścieżek portfela: normalne vs stresowe (OT interpolated).
    """
    stress_r = stress_result["stress_returns"][:n_paths_to_show]
    normal_r = stress_result["normal_returns"][:n_paths_to_show]
    scenario = stress_result["scenario_params"]
    sl = stress_result["stress_level"]

    cum_stress = np.cumprod(1 + stress_r, axis=1) * initial_capital
    cum_normal = np.cumprod(1 + normal_r, axis=1) * initial_capital

    n_days = stress_r.shape[1]
    x = np.arange(n_days)

    if title is None:
        title = (f"OT Stress Testing — {stress_result['crisis_scenario']} "
                 f"(stress_level={sl:.1f})")

    fig = go.Figure()

    # Normalne ścieżki (szary)
    for i in range(min(20, len(cum_normal))):
        fig.add_trace(go.Scatter(
            x=x, y=cum_normal[i],
            line=dict(color="rgba(200,200,200,0.15)", width=0.8),
            showlegend=(i == 0),
            name="Normalny rynek",
        ))

    # Stresowane ścieżki
    color_base = scenario.get("color", "#e74c3c")
    for i in range(min(n_paths_to_show, len(cum_stress))):
        opacity = 0.12 + 0.08 * sl
        fig.add_trace(go.Scatter(
            x=x, y=cum_stress[i],
            line=dict(color=f"{color_base}", width=0.8),
            opacity=opacity,
            showlegend=(i == 0),
            name=f"Stresowe ({stress_result['crisis_scenario']})",
        ))

    # Mediany
    med_stress = np.median(cum_stress[:n_paths_to_show], axis=0)
    med_normal = np.median(cum_normal[:n_paths_to_show], axis=0)
    p5_stress = np.percentile(cum_stress[:n_paths_to_show], 5, axis=0)

    fig.add_trace(go.Scatter(
        x=x, y=med_normal, line=dict(color="#aaa", width=2, dash="dash"),
        name="Mediana (normal)", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=med_stress, line=dict(color=color_base, width=2.5),
        name="Mediana (stress)", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p5_stress, line=dict(color=color_base, width=1.5, dash="dot"),
        name="5. percentyl (stress)", showlegend=True,
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e2e4f0")),
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.8)",
        font=dict(color="#e2e4f0", family="Inter"),
        xaxis_title="Dni",
        yaxis_title="Wartość portfela",
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=80),
    )
    fig.update_xaxes(gridcolor="#1c1c2e")
    fig.update_yaxes(gridcolor="#1c1c2e")
    return fig


def plot_stress_sweep(sweep_df: pd.DataFrame, title: str = "Metryki Ryzyka vs Poziom Stresu") -> go.Figure:
    """Wykres metryk ryzyka jako funkcja stress_level."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "VaR 95% i 99%", "CVaR 95% i 99%", "Max Drawdown (mediana)", "Stopa Przeżycia",
    ])

    sl = sweep_df["stress_level"]

    fig.add_trace(go.Scatter(x=sl, y=sweep_df["var_95"].abs(), name="VaR 95%",
                             line=dict(color="#f39c12")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sl, y=sweep_df["var_99"].abs(), name="VaR 99%",
                             line=dict(color="#e74c3c")), row=1, col=1)
    fig.add_trace(go.Scatter(x=sl, y=sweep_df["cvar_95"].abs(), name="CVaR 95%",
                             line=dict(color="#f39c12")), row=1, col=2)
    fig.add_trace(go.Scatter(x=sl, y=sweep_df["cvar_99"].abs(), name="CVaR 99%",
                             line=dict(color="#e74c3c")), row=1, col=2)
    fig.add_trace(go.Scatter(x=sl, y=sweep_df["max_dd_median"].abs(), name="Max DD",
                             line=dict(color="#9b59b6")), row=2, col=1)
    fig.add_trace(go.Scatter(x=sl, y=sweep_df["survival_rate"] * 100, name="Survival %",
                             line=dict(color="#00e676")), row=2, col=2)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e2e4f0")),
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.8)",
        font=dict(color="#e2e4f0", family="Inter"),
        showlegend=True,
        legend=dict(orientation="h", y=-0.12, font=dict(size=9)),
        margin=dict(l=60, r=20, t=70, b=80),
    )
    fig.update_xaxes(gridcolor="#1c1c2e")
    fig.update_yaxes(gridcolor="#1c1c2e")
    return fig

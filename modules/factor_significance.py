"""
factor_significance.py — Factor Zoo Multiple Testing Correction
================================================================
Korekta na Problem "Factor Zoo" — testowanie wielu faktorów bez
uwzględnienia inflacji p-value przez multiple testing.

Problem: Jeśli testujesz 50 faktorów przy α=5%, oczekujesz ~2.5 fałszywie
istotnych wyników NAWET jeśli żaden faktor nie jest naprawdę istotny.

Metody korekcji:
  1. BHY (Benjamini-Hochberg-Yekutieli) — zachowuje FDR, zalecane dla
     skorelowanych testów (jakimi są faktory finansowe)
  2. Bonferroni — konserwatywna, kontroluje FWER
  3. Holm-Bonferroni — mniej konserwatywna niż Bonferroni, step-down
  4. Storey q-value — adaptacyjna kontrola FDR (najlepsze mocy przy wielu testach)
  5. Harvey-Liu t*-correction — specyficzna dla finansów (Harvey, Liu & Zhu 2016)

Referencje:
  Harvey, Liu & Zhu (2016) — "...and the Cross-Section of Expected Returns"
                               Journal of Finance 71(5).
  Harvey & Liu (2020) — "False (and Missed) Discoveries in Financial Economics"
                         Journal of Financial Economics.
  Benjamini & Yekutieli (2001) — "The Control of the False Discovery Rate in
                                   Multiple Testing Under Dependency", Annals of Stats.
  Barras, Scaillet & Wermers (2010) — "False Discoveries in Mutual Fund Performance"
                                        Journal of Finance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional
from modules.logger import setup_logger

logger = setup_logger(__name__)


# ─── 1. Główna funkcja korekcji ───────────────────────────────────────────────

def correct_for_multiple_testing(
    t_stats: np.ndarray | list,
    n_obs: int | None = None,
    method: str = "BHY",
    alpha: float = 0.05,
    annual_factor: int = 252,
    return_all: bool = False,
) -> dict:
    """
    Korekta p-value i t-statystyk na multiple testing w Factor Zoo.

    Parameters
    ----------
    t_stats      : array t-statystyk faktorów (z testów OLS/regression)
    n_obs        : liczba obserwacji T (potrzebna dla Harvey-Liu)
    method       : 'BHY' | 'bonferroni' | 'holm' | 'storey' | 'harvey_liu'
    alpha        : poziom istotności (FDR lub FWER)
    annual_factor: annualizacja (252 dni handlowych)
    return_all   : czy zwrócić szczegóły per metoda

    Returns
    -------
    dict z:
      adjusted_pvalues  : (N,) — skorygowane p-values
      significant       : (N,) bool — czy faktor jest istotny po korekcji
      threshold_tstat   : float — minimalny t-stat dla istotności
      method            : str
      n_significant     : int
      n_factors         : int
      false_discovery_rate : float (szacunkowe FDR)
    """
    t_arr = np.asarray(t_stats, dtype=float)
    N = len(t_arr)

    if N == 0:
        return _empty_result(method)

    # P-values (dwustronne)
    p_raw = 2 * (1 - stats.t.cdf(np.abs(t_arr), df=max(n_obs or 252, N + 1) - 2))

    # ── Korekcja ──────────────────────────────────────────────────────────────
    if method == "BHY":
        p_adj, reject = _bhy_correction(p_raw, alpha)
    elif method == "bonferroni":
        p_adj = np.minimum(p_raw * N, 1.0)
        reject = p_adj <= alpha
    elif method == "holm":
        p_adj, reject = _holm_correction(p_raw, alpha)
    elif method == "storey":
        p_adj, reject = _storey_qvalue(p_raw, alpha)
    elif method == "harvey_liu":
        p_adj, reject = _harvey_liu_correction(t_arr, n_obs or 252, alpha)
    else:
        logger.warning(f"Nieznana metoda '{method}' — używam BHY")
        p_adj, reject = _bhy_correction(p_raw, alpha)

    # Minimalny t-stat dla istotności
    t_abs = np.abs(t_arr)
    t_threshold_by_method = _compute_t_threshold(method, N, n_obs or 252, alpha)

    # Szacowane FDR
    n_sig = int(reject.sum())
    n_expected_false = float(N * alpha)
    fdr_estimate = n_expected_false / max(n_sig, 1)

    result = {
        "t_stats":            t_arr,
        "p_values_raw":       p_raw,
        "adjusted_pvalues":   p_adj,
        "significant":        reject,
        "n_significant":      n_sig,
        "n_factors":          N,
        "n_false_positives_expected": n_expected_false,
        "false_discovery_rate":       min(fdr_estimate, 1.0),
        "threshold_tstat":    t_threshold_by_method,
        "method":             method,
        "alpha":              alpha,
    }

    if return_all:
        # Uruchom wszystkie metody dla porównania
        all_methods = {}
        for m in ["BHY", "bonferroni", "holm"]:
            if m == "BHY":
                p_m, r_m = _bhy_correction(p_raw, alpha)
            elif m == "bonferroni":
                p_m = np.minimum(p_raw * N, 1.0)
                r_m = p_m <= alpha
            else:
                p_m, r_m = _holm_correction(p_raw, alpha)
            all_methods[m] = {"n_significant": int(r_m.sum()), "reject": r_m}
        result["all_methods"] = all_methods

    return result


# ─── 2. Metody korekcji ───────────────────────────────────────────────────────

def _bhy_correction(p_values: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini-Hochberg-Yekutieli (2001) — zachowuje FDR przy dowolnej zależności.
    Zalecana dla faktorów finansowych (skorelowanych!).

    BHY jest bardziej konserwatywna niż BH bo mnoży przez c(N) = Σ1/k (harmoniczna).
    Ref: Yekutieli & Benjamini (1999), Ann. Stats.
    """
    N = len(p_values)
    c_N = np.sum(1.0 / np.arange(1, N + 1))  # Stała harmoniczna
    alpha_bhy = alpha / c_N

    # Posortuj i znajdź próg BH
    order = np.argsort(p_values)
    p_sorted = p_values[order]
    ranks = np.arange(1, N + 1)
    bh_threshold = ranks * alpha_bhy / N

    # Maksymalny k gdzie p_(k) <= k * alpha_bhy / N
    below = p_sorted <= bh_threshold
    if below.any():
        k_max = int(np.where(below)[0].max())
        reject_sorted = np.arange(N) <= k_max
    else:
        reject_sorted = np.zeros(N, dtype=bool)

    # Adjusted p-values (step-up)
    p_adj_sorted = np.minimum.accumulate(
        (N * c_N / ranks * p_sorted)[::-1]
    )[::-1]
    p_adj_sorted = np.minimum(p_adj_sorted, 1.0)

    # Odwróć permutację
    p_adj = np.empty(N)
    p_adj[order] = p_adj_sorted
    reject = np.zeros(N, dtype=bool)
    reject[order] = reject_sorted

    return p_adj, reject


def _holm_correction(p_values: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Holm-Bonferroni step-down — kontroluje FWER, mniej konserwatywna niż Bonferroni.
    """
    N = len(p_values)
    order = np.argsort(p_values)
    p_sorted = p_values[order]

    reject_sorted = np.zeros(N, dtype=bool)
    for i, p in enumerate(p_sorted):
        if p <= alpha / (N - i):
            reject_sorted[i] = True
        else:
            break  # step-down stops at first non-rejection

    # Adjusted p-values
    p_adj_sorted = np.minimum(p_sorted * (N - np.arange(N)), 1.0)
    p_adj_sorted = np.maximum.accumulate(p_adj_sorted)

    p_adj = np.empty(N)
    p_adj[order] = p_adj_sorted
    reject = np.zeros(N, dtype=bool)
    reject[order] = reject_sorted

    return p_adj, reject


def _storey_qvalue(p_values: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Storey q-value — adaptacyjna kontrola FDR (2002).
    Szacuje π₀ = udział prawdziwych hipotez zerowych.
    Ref: Storey (2002), J.R.Stat.Soc.B.
    """
    N = len(p_values)
    # Szacuj π₀ (Storey's lambda method)
    lambdas = np.linspace(0.05, 0.95, 19)
    pi0_list = [(p_values > l).sum() / (N * (1 - l)) for l in lambdas]
    # Spline fit dla stabilności — uproszczenie: użyj mediany
    pi0 = float(np.clip(np.median(pi0_list), 0, 1))

    order = np.argsort(p_values)
    p_sorted = p_values[order]
    ranks = np.arange(1, N + 1)

    q_sorted = pi0 * N * p_sorted / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.minimum(q_sorted, 1.0)

    q_values = np.empty(N)
    q_values[order] = q_sorted

    reject = q_values <= alpha
    return q_values, reject


def _harvey_liu_correction(
    t_stats: np.ndarray,
    n_obs: int,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Harvey & Liu (2016) — skorygowany próg t-statystyki dla Factor Zoo.

    Minimalna t-statystyka dla nowego faktora to nie 1.96 ale t* ≈ 3.0
    po uwzględnieniu Data Snooping Bias z ~300 testowanych faktorów.

    Wzór (Harvey, Liu & Zhu 2016, Table 1):
      Jeśli N faktorów testowanych, t* = ppf(1 - α/(2N)) [Bonferroni approx]
      Lub bardziej precyzyjnie: zależy od ρ̄ (average inter-factor correlation)

    Simplified version bez parametru ρ̄.
    """
    N = len(t_stats)
    # Harvey & Liu (2016): dla ~316 testów, t* ≈ 3.0 przy 5%
    # Skalujemy: t*(N) = ppf(1 - alpha / (2*N)) * korekt_hl
    HISTORICAL_N_FACTORS = 316  # Harvey & Liu (2016) Table 1
    t_star_base = 3.0  # HL zalecają 3.0 jako minimum dla nowego faktora

    # Adaptacja do bieżącego N
    if N <= 1:
        t_star = stats.norm.ppf(1 - alpha / 2)
    else:
        t_star = max(
            t_star_base * np.sqrt(N / HISTORICAL_N_FACTORS),
            stats.norm.ppf(1 - alpha / 2),
        )

    # P-values uwarunkowane na t*
    p_adj = 2 * (1 - stats.norm.cdf(np.abs(t_stats) / (t_star / stats.norm.ppf(1 - alpha / 2))))
    p_adj = np.clip(p_adj, 0, 1)
    reject = np.abs(t_stats) >= t_star

    return p_adj, reject


def _compute_t_threshold(method: str, N: int, n_obs: int, alpha: float) -> float:
    """Oblicza minimalny |t| dla istotności po korekcji."""
    if method == "bonferroni":
        return float(stats.t.ppf(1 - alpha / (2 * N), df=n_obs - 2))
    elif method == "harvey_liu":
        HISTORICAL_N_FACTORS = 316
        return max(3.0 * np.sqrt(N / HISTORICAL_N_FACTORS), stats.norm.ppf(1 - alpha / 2))
    else:
        # BHY/Holm/Storey — przybliżenie
        return float(stats.norm.ppf(1 - alpha * N / (2 * N * np.sum(1 / np.arange(1, N + 1)))))


def _empty_result(method: str) -> dict:
    return {
        "t_stats": np.array([]),
        "p_values_raw": np.array([]),
        "adjusted_pvalues": np.array([]),
        "significant": np.array([], dtype=bool),
        "n_significant": 0,
        "n_factors": 0,
        "n_false_positives_expected": 0,
        "false_discovery_rate": 0.0,
        "threshold_tstat": 1.96,
        "method": method,
        "alpha": 0.05,
    }


# ─── 3. Analiza faktorów portfela ─────────────────────────────────────────────

def analyze_portfolio_factors(
    returns_df: pd.DataFrame,
    factor_returns_df: pd.DataFrame,
    method: str = "BHY",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Przeprowadza factor regression i koryguje na multiple testing.

    Parameters
    ----------
    returns_df        : (T, n_assets) — zwroty aktywów/portfeli
    factor_returns_df : (T, n_factors) — zwroty faktorów (Fama-French, momentum etc.)
    method            : metoda korekcji
    alpha             : poziom FDR/FWER

    Returns
    -------
    DataFrame z kolumnami: factor, t_stat, p_raw, p_adj, significant, alpha_annual
    """
    from sklearn.linear_model import LinearRegression

    portfolio_ret = returns_df.mean(axis=1)  # równoważona portfelowo
    T = len(portfolio_ret)

    results = []
    for factor_name in factor_returns_df.columns:
        f = factor_returns_df[factor_name].dropna()
        common_idx = portfolio_ret.index.intersection(f.index)
        if len(common_idx) < 50:
            continue

        y = portfolio_ret.loc[common_idx].values
        X = f.loc[common_idx].values.reshape(-1, 1)

        try:
            from scipy.stats import linregress
            slope, intercept, r_val, p_val, std_err = linregress(X.flatten(), y)
            t_stat = slope / (std_err + 1e-10)
            alpha_ann = intercept * 252
        except Exception:
            continue

        results.append({
            "factor":    factor_name,
            "beta":      slope,
            "alpha_ann": alpha_ann,
            "t_stat":    t_stat,
            "r_squared": r_val ** 2,
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    correction = correct_for_multiple_testing(
        df["t_stat"].values,
        n_obs=T,
        method=method,
        alpha=alpha,
    )
    df["p_raw"] = correction["p_values_raw"]
    df["p_adj"] = correction["adjusted_pvalues"]
    df["significant"] = correction["significant"]
    df["threshold_t"] = correction["threshold_tstat"]
    df["method"] = method

    return df.sort_values("t_stat", ascending=False, key=abs).reset_index(drop=True)


# ─── 4. Plotly wizualizacja ───────────────────────────────────────────────────

def plot_factor_significance(
    factor_df: pd.DataFrame,
    threshold_t: float = 3.0,
    title: str = "Factor Zoo — Multiple Testing Analysis",
) -> "go.Figure":
    """Barplot t-statystyk z zaznaczeniem progu i istotności."""
    import plotly.graph_objects as go

    if factor_df.empty:
        return go.Figure()

    df = factor_df.sort_values("t_stat", ascending=True, key=abs)
    colors = ["#00e676" if s else "#e74c3c" for s in df["significant"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["factor"],
        x=df["t_stat"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in df["t_stat"]],
        textposition="outside",
        name="|t|",
    ))

    # Próg
    fig.add_vline(
        x=threshold_t, line_dash="dash", line_color="#ffea00", line_width=2,
        annotation_text=f"t* = {threshold_t:.2f} (po korekcji)",
        annotation_position="top right",
        annotation_font_size=10,
    )
    fig.add_vline(
        x=-threshold_t, line_dash="dash", line_color="#ffea00", line_width=2,
    )
    fig.add_vline(
        x=1.96, line_dash="dot", line_color="#aaa", line_width=1,
        annotation_text="t=1.96 (bez korekcji)",
        annotation_font_size=9,
    )
    fig.add_vline(x=-1.96, line_dash="dot", line_color="#aaa", line_width=1)

    n_sig = int(df["significant"].sum())
    n_total = len(df)

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup style='color:#aaa'>Istotnych po korekcji: {n_sig}/{n_total} faktorów</sup>",
            font=dict(size=14, color="#e2e4f0"),
        ),
        height=max(300, 30 * len(df) + 100),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.8)",
        font=dict(color="#e2e4f0", family="Inter"),
        xaxis_title="t-statystyka (|t| > t* → istotny po korekcji)",
        showlegend=False,
        margin=dict(l=150, r=60, t=80, b=40),
    )
    fig.update_xaxes(gridcolor="#1c1c2e", zeroline=True, zerolinecolor="#555")
    return fig


"""
scanner.py — Silnik EVT / Barbell Convexity Scanner v2.0

Matematyczne fundamenty (bez AI API):
  • Extreme Value Theory: GPD-POT dla prawego ORAZ lewego ogona (Pickands 1975, Balkema-de Haan 1974)
  • Hurst Exponent R/S Analysis (Peters 1994, Lo 1991) — trend vs. mean-reversion
  • Amihud (2002) Illiquidity Ratio — rzeczywista płynność transakcyjna
  • Omega Ratio (Shadwick & Keating 2002) — nie zakłada normalności rozkładu
  • Momentum 12-1 (Jegadeesh & Titman 1993) — czynnik trendu cenowego
  • Composite Barbell Score — ważony Z-Score 7 czynników (wzorowany AQR, Asness et al. 2012)
  • MST Mantegna (1999) + HRP Dendrogram (Lopez de Prado 2016)
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, genpareto
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
from modules.logger import setup_logger

logger = setup_logger(__name__)

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

from modules.metrics import calculate_sharpe, calculate_sortino, calculate_max_drawdown


# ─── EVT — Prawy Ogon (Zyski / Asymetryczne Wzrosty) ──────────────────────────

def evt_pot_right_tail(returns: np.ndarray, threshold_quantile: float = 0.90) -> float:
    """
    GPD Peaks-Over-Threshold — prawy ogon (zyski).
    xi > 0 = grube ogony zysku = wypukłe aktywo (idealne do Risky Sleeve).
    Odwołanie: Pickands (1975), Balkema & de Haan (1974).
    """
    if len(returns) < 50:
        return np.nan
    positive = returns[returns > 0]
    if len(positive) < 20:
        return np.nan
    u = np.quantile(positive, threshold_quantile)
    exc = positive[positive > u] - u
    if len(exc) < 5:
        return np.nan
    try:
        xi, _, _ = genpareto.fit(exc, floc=0)
        return float(xi)
    except Exception as e:
        logger.debug(f"EVT Right Tail fit failed: {e}")
        return np.nan


def evt_pot_left_tail(returns: np.ndarray, threshold_quantile: float = 0.05) -> float:
    """
    GPD Peaks-Over-Threshold — lewy ogon (straty / Crash Risk).
    xi > 0 = grube ogony STRATY = niebezpieczne aktywo do Barbella.
    Wysoki xi_left → dyskwalifikacja z Risky Sleeve.
    Odwołanie: Adrian & Brunnermeier (2011, CoVaR), Acharya et al. (2012, MES).
    """
    if len(returns) < 50:
        return np.nan
    losses = -returns[returns < 0]
    if len(losses) < 20:
        return np.nan
    u = np.quantile(losses, 1 - threshold_quantile)
    exc = losses[losses > u] - u
    if len(exc) < 5:
        return np.nan
    try:
        xi, _, _ = genpareto.fit(exc, floc=0)
        return float(xi)
    except Exception as e:
        logger.debug(f"EVT Left Tail fit failed: {e}")
        return np.nan


# ─── Wykładnik Hursta ─────────────────────────────────────────────────────────

def calculate_hurst_exponent(price_series: pd.Series, min_lag: int = 10, max_lag: int = 100) -> float:
    """
    R/S Analysis — Wykładnik Hursta.
    H > 0.55: trendowanie (persistent) — idealne dla strategii Momentum.
    H = 0.50: błądzenie losowe — brak przewagi.
    H < 0.45: mean-reversion — arbitraż statystyczny.
    Odwołanie: Hurst (1951), Peters (1994), Lo (1991).
    """
    returns = np.log(price_series / price_series.shift(1)).dropna().values
    n = len(returns)
    if n < max_lag * 2:
        max_lag = n // 4

    lags = range(min_lag, max(min_lag + 1, max_lag))
    rs_means = []
    valid_lags = []

    for lag in lags:
        chunks = [returns[i : i + lag] for i in range(0, n - lag, lag)]
        rs_chunk = []
        for chunk in chunks:
            adj = chunk - chunk.mean()
            cs  = np.cumsum(adj)
            r   = cs.max() - cs.min()
            s   = chunk.std(ddof=1)
            if s > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_means.append(np.mean(rs_chunk))
            valid_lags.append(lag)

    if len(valid_lags) < 5:
        return 0.5

    log_lags = np.log(valid_lags)
    log_rs   = np.log(rs_means)
    H, _     = np.polyfit(log_lags, log_rs, 1)
    return float(np.clip(H, 0.0, 1.0))


# ─── Omega Ratio ──────────────────────────────────────────────────────────────

def calculate_omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Omega Ratio (Shadwick & Keating 2002).
    Nie zakłada normalności — idealna miara dla aktywów z grubymi ogonami.
    Omega > 1.0 = aktywo zarabia więcej niż traci (Risky Sleeve worthy).
    """
    gains = np.maximum(returns - threshold, 0)
    losses = np.maximum(threshold - returns, 0)
    total_gains  = gains.sum()
    total_losses = losses.sum()
    if total_losses < 1e-10:
        return 999.0
    return float(total_gains / total_losses)


# ─── Amihud Illiquidity ───────────────────────────────────────────────────────

def calculate_amihud_ratio(price_series: pd.Series, volume_series: pd.Series,
                           lookback: int = 252) -> float:
    """
    Amihud (2002) Illiquidity Ratio.
    lower = bardziej płynne = lepsze do Risky Sleeve (małe koszty transakcyjne).
    """
    ret = np.abs(np.log(price_series / price_series.shift(1)).dropna())
    dollar_vol = (price_series * volume_series).dropna()
    common_idx = ret.index.intersection(dollar_vol.index)
    if len(common_idx) < 20:
        return np.nan
    ret_     = ret.loc[common_idx]
    dvol_    = dollar_vol.loc[common_idx]
    ratio    = (ret_ / dvol_.replace(0, np.nan)).dropna()
    return float(ratio.tail(lookback).mean() * 1e6)  # Skalujemy do czytelnej liczby


# ─── Price Momentum (12-1) ────────────────────────────────────────────────────

def calculate_momentum_12_1(price_series: pd.Series) -> float:
    """
    12-1 Momentum Factor (Jegadeesh & Titman 1993).
    Zwrot z ostatnich 12M pomniejszony o ostatni miesiąc (skip=21 dni).
    Silny momentum → aktywo trenduje → wyższy Score Barbella.
    """
    if len(price_series) < 273:  # 252 + 21
        if len(price_series) < 63:
            return 0.0
        # Krótka wersja: 3M momentum
        return float(price_series.iloc[-1] / price_series.iloc[-63] - 1)

    skip  = 21
    start = -(252 + skip)
    end   = -skip
    return float(price_series.iloc[-1 - skip] / price_series.iloc[start] - 1)


# ─── Metryki Wypukłości — Główna Funkcja ──────────────────────────────────────

def calculate_convecity_metrics(ticker: str, price_series: pd.Series,
                                volume_series: pd.Series | None = None) -> dict | None:
    """
    Oblicza pełen zestaw metryk dla skanera WYPUKŁOŚCI Barbella.
    Cel: znaleźć aktywa do RISKY SLEEVE — maksymalna asymetria zysk/strata.
    """
    if len(price_series) < 50:
        return None

    returns = np.log(price_series / price_series.shift(1)).dropna().values

    if len(returns) < 30:
        return None

    # 1. Podstawowe statystyki roczne
    vol_ann  = float(np.std(returns) * np.sqrt(252))
    mean_ann = float(np.mean(returns) * 252)

    # 2. Wyższe momenty
    skew_val = float(skew(returns))
    kurt_val = float(kurtosis(returns))  # Excess kurtosis (Fisher)

    # 3. Ratios hedgefundowe
    sharpe  = calculate_sharpe(returns)
    sortino = calculate_sortino(returns)
    max_dd  = calculate_max_drawdown(price_series)

    # 4. EVT — Prawy ogon (szukamy wypukłości)
    xi_right = evt_pot_right_tail(returns)

    # 5. EVT — Lewy ogon (Crash Risk) — niski = bezpieczniejszy
    xi_left  = evt_pot_left_tail(returns)

    # 6. Omega Ratio (nie zakłada normalności)
    omega = calculate_omega_ratio(returns)

    # 7. Hurst Exponent (trend vs. mean-reversion)
    hurst = calculate_hurst_exponent(price_series)

    # 8. Momentum 12-1
    momentum = calculate_momentum_12_1(price_series)

    # 9. Amihud Ratio (opcjonalne — wymaga wolumenu)
    amihud = np.nan
    if volume_series is not None and len(volume_series) > 50:
        amihud = calculate_amihud_ratio(price_series, volume_series)

    # 10. Kelly (Fat-Tail Safe) — połowa Kelly jako bezpieczna dawka
    risk_free = 0.04
    if vol_ann > 0 and not np.isnan(vol_ann):
        kelly_full = (mean_ann - risk_free) / (vol_ann ** 2)
        kelly_safe = kelly_full * 0.5  # Shrinkage 50%
    else:
        kelly_full = kelly_safe = 0.0

    # 11. Variance Drag (koszt zmienności)
    var_drag = 0.5 * (vol_ann ** 2)

    return {
        "Ticker":             ticker,
        "Annual Return":      mean_ann,
        "Volatility":         vol_ann,
        "Skewness":           skew_val,
        "Kurtosis":           kurt_val,
        "EVT Shape (Tail)":   xi_right,   # Prawy ogon (zyski) — wysoki = wypukłe
        "EVT Left Tail":      xi_left,    # Lewy ogon (straty) — niski = bezpieczne
        "Omega":              omega,
        "Hurst":              hurst,
        "Momentum_1Y":        momentum,
        "Amihud":             amihud,
        "Sharpe":             sharpe,
        "Sortino":            sortino,
        "Max Drawdown":       max_dd,
        "Variance Drag":      var_drag,
        "Kelly Full":         kelly_full,
        "Kelly Safe (50%)":   kelly_safe,
    }


# ─── Composite Barbell Score (Ważony Z-Score) ─────────────────────────────────

def score_asset_composite(metrics_df: pd.DataFrame) -> pd.Series:
    """
    Composite Barbell Score — ważony Z-Score 7 czynników.
    Metodologia: AQR Multi-Factor Composite (Asness, Frazzini, Pedersen 2012).
    
    Czynniki POSITIVE (wysoki → lepszy kandydat do Risky Sleeve):
      EVT Right Tail, Skewness, Omega, Hurst, Momentum
    Czynniki NEGATIVE (wysoki → gorszy, karujemy):
      EVT Left Tail (Crash Risk), Amihud (Illiquidity)
    """
    factor_weights = {
        "EVT Shape (Tail)": +0.30,   # Główny cel: wypukłość prawego ogona
        "Skewness":         +0.20,   # Asymetria zysk/strata (Taleb)
        "Omega":            +0.20,   # Gain/Loss ratio (bez założenia normalności)
        "Momentum_1Y":      +0.15,   # Czynnik trendu (JT 1993)
        "Hurst":            +0.10,   # Persistencja (Peters 1994)
        "EVT Left Tail":    -0.20,   # Crash Risk — KARA za gruby lewy ogon
        "Amihud":           -0.10,   # Illiquidity — KARA za brak płynności
    }

    composite = pd.Series(0.0, index=metrics_df.index)

    for col, weight in factor_weights.items():
        if col not in metrics_df.columns:
            continue
        series = metrics_df[col].copy()
        # Pomiń kolumny z samymi NaN
        if series.isna().all():
            continue
        mu  = series.mean(skipna=True)
        std = series.std(skipna=True)
        if std < 1e-9:
            continue
        zscore = (series - mu) / std
        composite += weight * zscore.fillna(0)

    return composite


def score_asset(metrics: dict) -> float:
    """
    Legacy single-asset score (używany dla kompatybilności z pipeline V1).
    UWAGA: Należy używać score_asset_composite() dla pełnej analiz df.
    """
    if metrics is None:
        return -999.0

    score = 0.0

    # EVT Prawy Ogon (Zyski)
    xi_right = metrics.get("EVT Shape (Tail)", np.nan)
    if not np.isnan(xi_right):
        if xi_right > 0.4:
            score += 60
        elif xi_right > 0.2:
            score += 30
        elif xi_right > 0:
            score += 10

    # EVT Lewy Ogon (Crash Risk) — KARA
    xi_left = metrics.get("EVT Left Tail", np.nan)
    if not np.isnan(xi_left):
        if xi_left > 0.6:
            score -= 50  # Wysokie ryzyko krachu — dyskwalifikacja
        elif xi_left > 0.3:
            score -= 20

    # Skośność (musi być +)
    skew_val = metrics.get("Skewness", 0.0)
    if skew_val > 0:
        score += 25 * skew_val
    else:
        score -= 40  # Ujemna skewness = ryzyko lewego ogona

    # Omega
    omega = metrics.get("Omega", 1.0)
    if omega is not None and not np.isnan(omega):
        if omega > 1.5:
            score += 25
        elif omega > 1.0:
            score += 10
        else:
            score -= 15

    # Momentum
    mom = metrics.get("Momentum_1Y", 0.0)
    if mom is not None and not np.isnan(mom):
        if mom > 0.20:
            score += 15
        elif mom > 0:
            score += 5
        else:
            score -= 10

    # Kelly (aktywo musi zarabiać)
    kelly = metrics.get("Kelly Full", 0.0)
    if kelly > 0.1:
        score += 15
    elif kelly <= 0:
        score -= 25

    return float(score)


# ─── Hierarchical Dendrogram (HRP) ────────────────────────────────────────────

def compute_hierarchical_dendrogram(returns_df: pd.DataFrame):
    """
    HRP Dendrogram — Lopez de Prado (2016).
    Zastępuje MST wizualizacją zagnieżdżonej struktury ryzyka.
    """
    tickers = returns_df.columns.tolist()
    if len(tickers) < 2:
        return None

    corr = returns_df.corr().fillna(0)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
    condensed = ssd.squareform(dist.values, checks=False)
    Z = sch.linkage(condensed, method="ward")

    fig = ff.create_dendrogram(
        dist.values,
        labels=tickers,
        linkagefun=lambda x: sch.linkage(x, method="ward"),
        color_threshold=float(np.percentile(Z[:, 2], 70)),
    )
    fig.update_layout(
        title="🌳 Dendrogram Hierarchiczny (HRP — Lopez de Prado 2016)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,25,0.9)",
        height=500,
        xaxis_title="Zgrupowane Aktywa",
        yaxis_title="Dystans Korelacyjny",
        font=dict(family="Inter", color="white"),
    )
    fig.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
    return fig


# ─── MST Correlation Network (Mantegna 1999) ──────────────────────────────────

def compute_correlation_network(returns_df: pd.DataFrame, metrics_df: pd.DataFrame = None):
    """
    Minimum Spanning Tree (Mantegna 1999) — Sieć korelacji.
    """
    if not HAS_NETWORKX:
        return None

    tickers = returns_df.columns.tolist()
    if len(tickers) < 2:
        return None

    corr = returns_df.corr()
    dist = np.sqrt(2 * (1 - corr))

    G = nx.Graph()
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i < j:
                G.add_edge(t1, t2, weight=float(dist.loc[t1, t2]),
                           corr=float(corr.loc[t1, t2]))

    mst = nx.minimum_spanning_tree(G, weight="weight")
    pos = nx.spring_layout(mst, seed=42, k=2.0)

    if metrics_df is not None and "Barbell Score" in metrics_df.columns:
        score_map    = metrics_df.set_index("Ticker")["Barbell Score"].to_dict() \
                       if "Ticker" in metrics_df.columns else {}
        node_colors  = [score_map.get(t, 0.0) for t in mst.nodes()]
        color_label  = "Barbell Score"
    elif metrics_df is not None and "Sharpe" in metrics_df.columns:
        sharpe_map   = metrics_df.set_index("Ticker")["Sharpe"].to_dict() \
                       if "Ticker" in metrics_df.columns else {}
        node_colors  = [sharpe_map.get(t, 0.0) for t in mst.nodes()]
        color_label  = "Sharpe Ratio"
    else:
        node_colors  = [dict(mst.degree())[t] for t in mst.nodes()]
        color_label  = "Degree"

    edge_x, edge_y = [], []
    for u, v, _ in mst.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="rgba(150,150,200,0.5)"),
        hoverinfo="none", mode="lines", name="Korelacja (MST)",
    )

    node_x = [pos[t][0] for t in mst.nodes()]
    node_y = [pos[t][1] for t in mst.nodes()]
    labels = list(mst.nodes())

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=labels, textposition="top center",
        textfont=dict(color="white", size=11),
        marker=dict(
            size=20, color=node_colors,
            colorscale="RdYlGn",
            colorbar=dict(title=color_label, thickness=12),
            showscale=True, line=dict(color="white", width=1.5),
        ),
        hovertemplate=[
            f"<b>{t}</b><br>{color_label}: {c:.2f}<extra></extra>"
            for t, c in zip(labels, node_colors)
        ],
        name="Aktywa",
    )

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title="🕸️ Sieć Korelacji MST (Mantegna 1999)",
        showlegend=False, hovermode="closest",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,25,0.9)",
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        font=dict(family="Inter", color="white"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
    return fig

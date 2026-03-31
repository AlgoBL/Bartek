"""
entropy_regime.py — Entropy-Based Regime Detection
====================================================
Ref:
  - Bandt & Pompe (2002) PRL — Permutation Entropy
  - Behrendt et al. (2023) IJFE — Transfer Flow Entropy
  - Dong & Gao (2024) "Entropy-Based Market Regime Identification"

Permutation Entropy (PermEn):
  • Niskie PermEn → rynek jest uporządkowany, silny trend (Bull/Bear)
  • Wysokie PermEn → rynek chaotyczny, konsolidacja / pre-crash

Sample Entropy (SampEn):
  • Mierzy nielosowosc - brak regularności oznacza wysoki szum (Mediocristan)
  • Niska SampEn → persistentne/przewidywalne serie

Warunki wejściowe: szereg zwrotów lub cen (numpy array / pd.Series)
"""

from __future__ import annotations
import math as _math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import permutations
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  PERMUTATION ENTROPY
# ─────────────────────────────────────────────────────────────────────────────

def permutation_entropy(
    series: np.ndarray | pd.Series,
    m: int = 3,
    tau: int = 1,
    normalize: bool = True,
) -> float:
    """
    Oblicza Permutation Entropy szeregu czasowego.

    Parameters
    ----------
    series    : szereg zwrotów / cen
    m         : długość wzorca (embedding dimension), zwykle 3-6
    tau       : opóźnienie (lag), zwykle 1
    normalize : czy normalizować do [0, 1] (dzielenie przez log2(m!))

    Returns
    -------
    float — entropia permutacyjna ∈ [0, 1] (jeśli normalize=True)
    """
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)

    if n < m:
        return np.nan

    # Wszystkie możliwe permutacje długości m
    all_perms = list(permutations(range(m)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    counts = np.zeros(len(all_perms))

    for i in range(n - (m - 1) * tau):
        # Pobierz okno
        window = x[i : i + m * tau : tau]
        # Wzorzec rzędowy
        pattern = tuple(np.argsort(window))
        if pattern in perm_to_idx:
            counts[perm_to_idx[pattern]] += 1

    # Rozkład prawdopodobieństwa
    total = counts.sum()
    if total == 0:
        return np.nan
    probs = counts[counts > 0] / total

    # Entropia Shannona
    H = -np.sum(probs * np.log2(probs))

    if normalize:
        H_max = np.log2(_math.factorial(m))
        H = H / H_max if H_max > 0 else 0.0

    return float(np.clip(H, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
#  SAMPLE ENTROPY
# ─────────────────────────────────────────────────────────────────────────────

def sample_entropy(
    series: np.ndarray | pd.Series,
    m: int = 2,
    r: float = 0.2,
) -> float:
    """
    Oblicza Sample Entropy (SampEn).

    Parameters
    ----------
    series : szereg czasowy
    m      : długość szablonu
    r      : tolerancja (jako ułamek std serii, typowo 0.1-0.25)

    Returns
    -------
    float — SampEn (wyższe = bardziej losowe, niższe = bardziej regularne)
    """
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    std_x = np.std(x)

    if std_x == 0 or n < m + 2:
        return np.nan

    r_abs = r * std_x

    def count_templates(m_len):
        count = 0
        for i in range(n - m_len):
            template = x[i:i + m_len]
            for j in range(n - m_len):
                if i == j:
                    continue
                if np.max(np.abs(x[j:j + m_len] - template)) < r_abs:
                    count += 1
        return count

    A = count_templates(m + 1)
    B = count_templates(m)

    if B == 0:
        return np.nan

    return float(-np.log(A / B))


# ─────────────────────────────────────────────────────────────────────────────
#  ROLLING ENTROPY — do wykresów czasowych
# ─────────────────────────────────────────────────────────────────────────────

def rolling_permutation_entropy(
    series: pd.Series,
    window: int = 60,
    m: int = 3,
    tau: int = 1,
    step: int = 1,
    normalize: bool = True,
) -> pd.Series:
    """
    Oblicza kroczącą Permutation Entropy.

    Parameters
    ----------
    series : pd.Series szeregu (zwroty lub ceny)
    window : długość okna (np. 60 = kwartał handlowy)
    m, tau : parametry PE
    step   : krok przesuwania okna
    normalize : normalizacja do [0,1]

    Returns
    -------
    pd.Series z entropią dla każdego okna
    """
    n   = len(series)
    idx = []
    pe  = []

    i = window
    while i <= n:
        window_data = series.iloc[i - window:i].values
        val = permutation_entropy(window_data, m=m, tau=tau, normalize=normalize)
        idx.append(series.index[i - 1])
        pe.append(val)
        i += step

    return pd.Series(pe, index=idx, name="PermEn")


def compute_entropy_regime(
    returns: pd.Series | np.ndarray,
    window: int = 60,
    m: int = 3,
    tau: int = 1,
    bear_threshold: float = 0.75,
    bull_threshold: float = 0.40,
) -> pd.DataFrame:
    """
    Pełna analiza reżimu entropi — zwraca DataFrame z:
    - PermEn (krocząca)
    - Regime: 'Bull' / 'Neutral' / 'Bear/Chaos'
    - Regime_Color

    Parameters
    ----------
    returns         : szereg zwrotów (dziennych lub tygodniowych)
    window          : okno kroczącej entropii (dni)
    bear_threshold  : PermEn > bear_threshold → "Chaotic/Pre-crash"
    bull_threshold  : PermEn < bull_threshold → "Trending/Bull"
    """
    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns)

    pe_series = rolling_permutation_entropy(returns, window=window, m=m, tau=tau)

    regimes = []
    colors  = []
    for pe in pe_series:
        if np.isnan(pe):
            regimes.append("Unknown")
            colors.append("#555")
        elif pe < bull_threshold:
            regimes.append("Trending")
            colors.append("#00e676")
        elif pe > bear_threshold:
            regimes.append("Chaotic")
            colors.append("#ff1744")
        else:
            regimes.append("Neutral")
            colors.append("#ffea00")

    return pd.DataFrame({
        "PermEn":  pe_series.values,
        "Regime":  regimes,
        "Color":   colors,
    }, index=pe_series.index)


# ─────────────────────────────────────────────────────────────────────────────
#  WIZUALIZACJA PLOTLY
# ─────────────────────────────────────────────────────────────────────────────

def plot_entropy_regime(
    entropy_df: pd.DataFrame,
    price_series: Optional[pd.Series] = None,
    title: str = "Entropia Permutacyjna — Chaos vs Porządek Rynkowy",
) -> go.Figure:
    """
    Tworzy wykres double-panel: cena + entropia z kolorowaniem stref.

    Parameters
    ----------
    entropy_df   : DataFrame z kolumnami PermEn, Regime, Color (z compute_entropy_regime)
    price_series : opcjonalny szereg cen do wykreślenia w górnym panelu
    title        : tytuł wykresu

    Returns
    -------
    go.Figure
    """
    has_price = price_series is not None and len(price_series) > 0
    rows = 2 if has_price else 1
    row_heights = [0.40, 0.60] if has_price else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(["Cena (log)", "Permutation Entropy"] if has_price
                        else ["Permutation Entropy"]),
        row_heights=row_heights,
    )

    price_row = 1 if has_price else None
    pe_row    = 2 if has_price else 1

    # ── Górny panel: cena ─────────────────────────────────────────────────────
    if has_price:
        # Wyrównaj indeksy
        common_idx = price_series.index.intersection(entropy_df.index)
        p = price_series.loc[common_idx]
        fig.add_trace(go.Scatter(
            x=p.index, y=np.log(p / p.iloc[0] + 1) if p.min() > 0 else p,
            mode="lines",
            line=dict(color="#e2e4f0", width=1.5),
            name="Cena (log)",
        ), row=price_row, col=1)

        # Kolorowanie tła wg reżimu
        e = entropy_df.loc[entropy_df.index.isin(common_idx)]
        prev_regime = None
        seg_start   = None
        for i, (idx, row) in enumerate(e.iterrows()):
            regime = row["Regime"]
            color  = row["Color"]
            if regime != prev_regime:
                if seg_start is not None and prev_regime != "Unknown":
                    fill_color = {
                        "Trending": "rgba(0,230,118,0.08)",
                        "Neutral":  "rgba(255,234,0,0.06)",
                        "Chaotic":  "rgba(255,23,68,0.10)",
                    }.get(prev_regime, "rgba(0,0,0,0)")
                    fig.add_vrect(
                        x0=seg_start, x1=idx,
                        fillcolor=fill_color,
                        layer="below", line_width=0,
                        row=price_row, col=1,
                    )
                seg_start   = idx
                prev_regime = regime

    # ── Dolny panel: Permutation Entropy ─────────────────────────────────────
    pe = entropy_df["PermEn"].dropna()

    # Zona referencyjna
    fig.add_hrect(
        y0=0.0, y1=0.40,
        fillcolor="rgba(0,230,118,0.07)", layer="below", line_width=0,
        row=pe_row, col=1,
    )
    fig.add_hrect(
        y0=0.75, y1=1.0,
        fillcolor="rgba(255,23,68,0.07)", layer="below", line_width=0,
        row=pe_row, col=1,
    )

    # Linia PE z gradientem kolorów
    fig.add_trace(go.Scatter(
        x=pe.index, y=pe.values,
        mode="lines",
        line=dict(color="#00ccff", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,204,255,0.06)",
        name="PermEn",
    ), row=pe_row, col=1)

    # Progi
    fig.add_hline(y=0.40, line_dash="dash", line_color="#00e676", line_width=1,
                  annotation_text="Trending (<0.40)", annotation_position="bottom right",
                  annotation_font_size=9, row=pe_row, col=1)
    fig.add_hline(y=0.75, line_dash="dash", line_color="#ff1744", line_width=1,
                  annotation_text="Chaotic (>0.75)", annotation_position="top right",
                  annotation_font_size=9, row=pe_row, col=1)

    # Layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e2e4f0")),
        height=500 if has_price else 320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e4f0", family="Inter"),
        showlegend=False,
        margin=dict(l=50, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="#1c1c2e", gridwidth=0.5)
    fig.update_yaxes(gridcolor="#1c1c2e", gridwidth=0.5)
    fig.update_yaxes(range=[0, 1.05], row=pe_row, col=1)

    return fig


def plot_entropy_distribution(entropy_df: pd.DataFrame) -> go.Figure:
    """Rozkład entropii per reżim — violin + rug plot."""
    pe = entropy_df["PermEn"].dropna()
    regime_data = {}
    for regime in ["Trending", "Neutral", "Chaotic"]:
        mask = entropy_df["Regime"] == regime
        vals = entropy_df.loc[mask, "PermEn"].dropna().values
        if len(vals) > 0:
            regime_data[regime] = vals

    colors_map = {
        "Trending": "#00e676",
        "Neutral":  "#ffea00",
        "Chaotic":  "#ff1744",
    }

    fig = go.Figure()
    for regime, vals in regime_data.items():
        fig.add_trace(go.Violin(
            x=[regime] * len(vals), y=vals,
            name=regime,
            box_visible=True,
            meanline_visible=True,
            fillcolor=colors_map[regime].replace("#", "rgba(") + ",0.20)",
            line_color=colors_map[regime],
            points="outliers",
        ))

    fig.update_layout(
        title="Rozkład Entropii per Reżim",
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e4f0", family="Inter"),
        violinmode="group",
        legend=dict(orientation="h", y=1.05),
        xaxis=dict(gridcolor="#1c1c2e"),
        yaxis=dict(gridcolor="#1c1c2e", title="Permutation Entropy", range=[0, 1.05]),
        margin=dict(l=50, r=20, t=45, b=35),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  SZYBKI DEMO (bez danych zewnętrznych)
# ─────────────────────────────────────────────────────────────────────────────

def demo_entropy(n: int = 500, seed: int = 42) -> dict:
    """
    Generuje syntetyczne dane z 3 reżimów i oblicza entropię.
    Przydatne do testów i demonstracji w Streamlit.
    """
    np.random.seed(seed)
    returns = []

    # Segment 1: silny trend (niskie PermEn)
    returns.extend(np.random.normal(0.001, 0.005, n // 3))
    # Segment 2: losowy szum (wysokie PermEn)
    returns.extend(np.random.normal(0.0, 0.025, n // 3))
    # Segment 3: mean-reversion + szum (umiarkowane)
    ar = [0.0]
    for _ in range(n - 2 * (n // 3)):
        ar.append(-0.6 * ar[-1] + np.random.normal(0, 0.012))
    returns.extend(ar[1:])

    idx = pd.date_range("2022-01-01", periods=len(returns), freq="B")
    series = pd.Series(returns, index=idx)
    entropy_df = compute_entropy_regime(series, window=40, m=3)

    return {
        "returns":    series,
        "entropy_df": entropy_df,
        "regime_counts": entropy_df["Regime"].value_counts().to_dict(),
    }

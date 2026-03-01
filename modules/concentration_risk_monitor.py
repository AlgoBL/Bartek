"""
concentration_risk_monitor.py — Monitor Ryzyka Koncentracji

Implementuje:
1. HHI (Herfindahl-Hirschman Index) — efektywna liczba aktywów
2. PCA Concentration — ile wariancji wyjaśnia 1. składowa główna
3. Factor Concentration — ekspozycja na czynniki ryzyka (Fama-French style)
4. Geographic/Currency Risk — koncentracja walutowa
5. Sector Overlap — ukryta korelacja przez sektor

Referencje:
  - Lopez de Prado (2016) — Hierarchical Risk Parity
  - Meucci (2010) — Managing Diversification
  - Choueifaty & Coignard (2008) — Maximum Diversification
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from modules.logger import setup_logger

logger = setup_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. HERFINDAHL-HIRSCHMAN INDEX — Effective N
# ══════════════════════════════════════════════════════════════════════════════

def compute_hhi(weights: np.ndarray | list | pd.Series) -> dict:
    """
    Herfindahl-Hirschman Index dla portfela.

    HHI = sum(w_i²) ∈ [1/N, 1]
    Effective N = 1 / HHI ∈ [1, N]

    HHI = 1/N → portfel równoważony (maksymalna dywersyfikacja)
    HHI = 1   → jeden asset (brak dywersyfikacji)

    Returns
    -------
    dict: hhi, effective_n, n_assets, concentration_pct, label
    """
    w = np.array(weights, dtype=float)
    w = np.abs(w)
    if w.sum() < 1e-10:
        return {"error": "Puste wagi"}
    w = w / w.sum()

    hhi = float(np.sum(w ** 2))
    n = len(w)
    effective_n = 1.0 / (hhi + 1e-10)
    hhi_min = 1.0 / n  # idealne
    concentration_pct = (hhi - hhi_min) / (1.0 - hhi_min + 1e-10)

    if effective_n >= n * 0.75:
        label = "✅ Dobrze zdywersyfikowany"
    elif effective_n >= n * 0.50:
        label = "🟡 Umiarkowana koncentracja"
    elif effective_n >= n * 0.25:
        label = "🟠 Wysoka koncentracja"
    else:
        label = "🔴 Krytyczna koncentracja"

    return {
        "hhi": hhi,
        "effective_n": effective_n,
        "n_assets": n,
        "concentration_pct": concentration_pct,
        "label": label,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. PCA CONCENTRATION — Variance Explained by PC1
# ══════════════════════════════════════════════════════════════════════════════

def pca_concentration(
    returns_df: pd.DataFrame,
    n_components: int | None = None,
) -> dict:
    """
    Analiza PCA portfela — ile wariancji wyjaśnia 1. składowa.

    Jeśli PC1 > 70% wariancji → portfel faktycznie jednoczynnikowy
    (wszystkie aktywa ładują się na jedno ryzyko).

    Returns
    -------
    dict:
      variance_explained : list[float] — wkład każdego PC
      pc1_variance       : float — wkład PC1
      n_dominant_factors : int — ile PC trzeba żeby wyjaśnić >80% wariancji
      diversification_ratio : float — entropia rozkładu (Meucci 2009)
      is_concentrated    : bool
      label              : str
    """
    df = returns_df.dropna(how="all")
    if len(df) < 30 or df.shape[1] < 2:
        return {"error": "Za mało danych (min 30 wierszy, 2 aktywa)"}

    # Standardize
    X = df.sub(df.mean()).div(df.std() + 1e-10)
    cov = X.cov().values
    n = cov.shape[0]

    if n_components is None:
        n_components = n

    try:
        eigvals, eigvecs = eigh(cov)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
    except Exception:
        return {"error": "Błąd dekompozycji PCA"}

    total = np.sum(np.maximum(eigvals, 0))
    if total < 1e-10:
        return {"error": "Macierz osobliwa"}

    var_explained = [max(0, v) / total for v in eigvals]

    # Dominant factors (80% kumulatywnie)
    cumsum = np.cumsum(var_explained)
    n_dom = int(np.searchsorted(cumsum, 0.80)) + 1

    pc1 = var_explained[0]

    # Entropy-based diversification (Meucci 2010)
    p = np.array([max(v, 1e-10) for v in var_explained])
    p = p / p.sum()
    diversity = float(np.exp(-np.sum(p * np.log(p))))

    is_concentrated = pc1 > 0.60

    if pc1 < 0.40:
        label = "✅ Wysoka dywersyfikacja czynnikowa"
    elif pc1 < 0.60:
        label = "🟡 Umiarkowana koncentracja czynnikowa"
    elif pc1 < 0.80:
        label = "🟠 Silna koncentracja — ryzyko jednoczynnikowe"
    else:
        label = "🔴 Krytyczna — portfel faktycznie jednoaktywa"

    return {
        "variance_explained": var_explained[:min(10, n)],
        "pc1_variance": pc1,
        "n_dominant_factors": n_dom,
        "diversification_ratio": diversity,
        "is_concentrated": is_concentrated,
        "label": label,
        "loadings": pd.DataFrame(
            eigvecs[:, :min(3, n)],
            index=df.columns,
            columns=[f"PC{i+1}" for i in range(min(3, n))],
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. FACTOR CONCENTRATION (BETA EXPOSURE)
# ══════════════════════════════════════════════════════════════════════════════

def compute_factor_betas(
    returns_df: pd.DataFrame,
    factor_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Oblicza bety portfela względem czynników ryzyka.

    Używa OLS: r_i = α_i + β_i,1 * F_1 + ... + β_i,k * F_k + ε_i

    Parameters
    ----------
    returns_df    : pd.DataFrame — dzienne zwroty aktywów
    factor_returns: dict — {nazwa_czynnika: pd.Series dzienne zwroty}

    Returns
    -------
    pd.DataFrame — aktywo × czynnik bety
    """
    from numpy.linalg import lstsq

    aligned = returns_df.copy()
    factors = pd.DataFrame(factor_returns)

    # Align on common dates
    common_idx = aligned.index.intersection(factors.index)
    if len(common_idx) < 30:
        return pd.DataFrame()

    X = factors.loc[common_idx].values
    X = np.column_stack([np.ones(len(X)), X])  # add intercept

    betas = {}
    for col in returns_df.columns:
        y = aligned.loc[common_idx, col].values
        mask = ~np.isnan(y)
        if mask.sum() < 20:
            continue
        b, _, _, _ = lstsq(X[mask], y[mask], rcond=None)
        betas[col] = dict(zip(["alpha"] + list(factor_returns.keys()), b))

    return pd.DataFrame(betas).T


# ══════════════════════════════════════════════════════════════════════════════
# 4. MAXIMUM DIVERSIFICATION RATIO
# ══════════════════════════════════════════════════════════════════════════════

def diversification_ratio(
    weights: np.ndarray,
    cov: np.ndarray,
    vols: np.ndarray,
) -> float:
    """
    Choueifaty & Coignard (2008) Diversification Ratio.

    DR = (w^T σ) / sqrt(w^T Σ w)
    DR = 1 → brak dywersyfikacji (jeden asset)
    DR = sqrt(N) → idealna dywersyfikacja uncorrelated assets

    Im wyższy DR, tym portfel lepiej dywersyfikowany.
    """
    w = np.array(weights)
    if w.sum() < 1e-10:
        return 1.0
    w = w / w.sum()
    portfolio_vol = np.sqrt(w @ cov @ w + 1e-12)
    weighted_avg_vol = w @ vols
    return float(weighted_avg_vol / (portfolio_vol + 1e-10))


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONCENTRATION RISK SCORE (0–100, 0=koncentracja, 100=dywersyfikacja)
# ══════════════════════════════════════════════════════════════════════════════

def concentration_risk_score(
    weights: np.ndarray | list | pd.Series,
    returns_df: pd.DataFrame | None = None,
    asset_names: list[str] | None = None,
) -> dict:
    """
    Kompleksowa ocena ryzyka koncentracji portfela.

    Składowe (łącznie 100 pkt):
      - HHI / Effective N : 35 pkt
      - PCA PC1            : 35 pkt
      - Top-3 concentration: 30 pkt

    Returns
    -------
    dict z:
      total_score       : float 0–100 (100 = doskonała dywersyfikacja)
      hhi_result        : dict
      pca_result        : dict | None
      top3_weight       : float
      recommendations   : list[str]
      grade             : str
    """
    w = np.array(weights, dtype=float)
    if len(w) == 0:
        return {"error": "Puste wagi"}

    names = asset_names or [f"Asset {i+1}" for i in range(len(w))]

    # HHI score (35 pkt)
    hhi_res = compute_hhi(w)
    eff_n = hhi_res.get("effective_n", 1)
    n = hhi_res.get("n_assets", 1)
    hhi_score = 35 * min(1.0, (eff_n - 1) / max(n - 1, 1))

    # PCA score (35 pkt)
    pca_score = 17.5  # neutral
    pca_res = None
    if returns_df is not None and returns_df.shape[1] >= 2:
        pca_res = pca_concentration(returns_df)
        pc1 = pca_res.get("pc1_variance", 0.5)
        pca_score = 35 * (1 - pc1)

    # Top-3 concentration (30 pkt)
    w_norm = np.abs(w) / (np.abs(w).sum() + 1e-10)
    top3_weight = float(np.sort(w_norm)[::-1][:3].sum())
    top3_score = 30 * (1 - max(0, (top3_weight - 1 / n) / (1 - 1 / n + 1e-10)))

    total = hhi_score + pca_score + top3_score

    grade = "A+" if total >= 85 else "A" if total >= 75 else "B" if total >= 60 else "C" if total >= 45 else "D" if total >= 30 else "F"

    # Recommendations
    recs = []
    if eff_n < n * 0.4:
        recs.append(f"🔴 Effectywna liczba aktywów ({eff_n:.1f}) to tylko {eff_n/n:.0%} portfela — dodaj aktywów")
    if pca_res and pca_res.get("pc1_variance", 0) > 0.60:
        pc1 = pca_res['pc1_variance']
        recs.append(f"🟠 PC1 wyjaśnia {pc1:.0%} wariancji — wszystko napędza jedno ryzyko rynkowe")
    if top3_weight > 0.70:
        recs.append(f"🟡 Top-3 aktywa stanowią {top3_weight:.0%} portfela — za duża koncentracja")
    if not recs:
        recs.append("✅ Portfel dobrze zdywersyfikowany pod każdym kątem")

    # Top contributors
    w_sorted_idx = np.argsort(w_norm)[::-1]
    top_assets = [(names[i], float(w_norm[i])) for i in w_sorted_idx[:5] if i < len(names)]

    return {
        "total_score": float(total),
        "grade": grade,
        "hhi_result": hhi_res,
        "pca_result": pca_res,
        "top3_weight": top3_weight,
        "top_assets": top_assets,
        "recommendations": recs,
        "effective_n": eff_n,
        "n_assets": n,
    }

"""
regime_adaptive_allocation.py — Dynamiczne Przełączanie Reżimów (HMM)

Implementuje:
1. Hidden Markov Model (3-state): Risk-On / Risk-Off / Crisis
2. Regime-conditional covariance matrices
3. Smooth transition weights (sigmoid blending)
4. Regime persistence / duration model

Referencje:
  - Hamilton (1989) — A New Approach to Economic Analysis (HMM macro)
  - Ang & Bekaert (2002) — International Asset Allocation with Regime Shifts
  - Guidolin & Timmermann (2008) — International Asset Allocation under Regime Switching
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from modules.logger import setup_logger

logger = setup_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SIMPLE 3-STATE REGIME DETECTOR (rule-based fallback)
# ══════════════════════════════════════════════════════════════════════════════

REGIMES = {
    0: {"name": "Risk-On (Bull)", "emoji": "🟢", "barbell_safe": 0.30, "barbell_risky": 0.70, "color": "#00e676"},
    1: {"name": "Risk-Off (Bear)", "emoji": "🟡", "barbell_safe": 0.55, "barbell_risky": 0.45, "color": "#ffea00"},
    2: {"name": "Crisis (Crash)", "emoji": "🔴", "barbell_safe": 0.80, "barbell_risky": 0.20, "color": "#ff1744"},
}


def detect_regime_rule_based(
    returns: pd.Series,
    vix: float | None = None,
    yield_curve: float | None = None,
    credit_spread_hy: float | None = None,
) -> dict:
    """
    Detekcja reżimu opartą na regułach (deterministyczna, szybka).

    Sygnały:
      - 63-dniowa rolling vol vs 252-dniowe baseline
      - VIX poziom
      - Yield curve
      - HY spread

    Returns
    -------
    dict z:
      regime_id    : int 0/1/2
      regime_name  : str
      confidence   : float 0–1
      signals      : dict
      recommended_barbell_weights : dict
    """
    r = returns.dropna()
    if len(r) < 63:
        return {"error": "Za mało danych (min 63 dni)"}

    score = 0.0  # 0 = Risk-On, 10 = Crisis

    signals = {}

    # --- Vol regime ---
    vol_63 = float(r.iloc[-63:].std() * np.sqrt(252))
    vol_252 = float(r.std() * np.sqrt(252)) if len(r) >= 252 else vol_63
    vol_ratio = vol_63 / (vol_252 + 1e-10)
    signals["vol_ratio"] = vol_ratio
    if vol_ratio > 2.0:
        score += 4.0
    elif vol_ratio > 1.5:
        score += 2.5
    elif vol_ratio > 1.2:
        score += 1.0

    # --- Drawdown ---
    cum = (1 + r).cumprod()
    dd = float((cum.iloc[-1] - cum.max()) / (cum.max() + 1e-10))
    signals["current_drawdown"] = dd
    if dd < -0.20:
        score += 4.0
    elif dd < -0.10:
        score += 2.0
    elif dd < -0.05:
        score += 1.0

    # --- VIX ---
    if vix is not None:
        signals["vix"] = vix
        if vix > 35:
            score += 3.0
        elif vix > 25:
            score += 1.5
        elif vix > 20:
            score += 0.5

    # --- Yield curve ---
    if yield_curve is not None:
        signals["yield_curve"] = yield_curve
        if yield_curve < -0.50:
            score += 2.0
        elif yield_curve < 0:
            score += 1.0

    # --- HY Spread ---
    if credit_spread_hy is not None:
        signals["hy_spread"] = credit_spread_hy
        if credit_spread_hy > 700:
            score += 3.0
        elif credit_spread_hy > 500:
            score += 1.5
        elif credit_spread_hy > 400:
            score += 0.5

    max_score = 16.0  # normalizacja
    normalized = min(1.0, score / max_score)

    if normalized < 0.25:
        regime_id = 0
    elif normalized < 0.60:
        regime_id = 1
    else:
        regime_id = 2

    regime = REGIMES[regime_id]

    return {
        "regime_id": regime_id,
        "regime_name": regime["name"],
        "emoji": regime["emoji"],
        "confidence": 1.0 - abs(normalized - (regime_id * 0.5)),
        "raw_score": score,
        "normalized_score": normalized,
        "signals": signals,
        "recommended_weights": {
            "safe": regime["barbell_safe"],
            "risky": regime["barbell_risky"],
        },
        "color": regime["color"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. HMM-INSPIRED REGIME SMOOTHING (Gaussian Mixture)
# ══════════════════════════════════════════════════════════════════════════════

def fit_gaussian_mixture_regimes(
    returns: pd.Series,
    n_states: int = 3,
    n_iter: int = 100,
    rng_seed: int = 42,
) -> dict:
    """
    Uproszczone EM dla Gaussian Mixture Model na zwrotach.
    Aproksymuje Hamilton (1989) HMM bez pełnej implementacji Viterbi.

    Returns
    -------
    dict z:
      state_probs   : pd.DataFrame — P(state | date) dla każdej daty
      state_means   : list[float] — średnia w każdym stanie
      state_vols    : list[float] — vol w każdym stanie
      state_labels  : list[str]   — etykiety stanów
      current_state : int
      current_probs : np.ndarray
    """
    r = returns.dropna()
    if isinstance(r, pd.DataFrame):
        r = r.iloc[:, 0]
    r = np.asarray(r).ravel()  # guarantee 1-D float array

    if len(r) < 60:
        return {"error": "Za mało danych"}

    rng = np.random.default_rng(rng_seed)

    # Initialize: z-score partition
    sorted_r = np.sort(r)
    n = len(r)
    breakpoints = [sorted_r[n // n_states * i] for i in range(1, n_states)]

    means = np.array([
        sorted_r[:n // n_states].mean(),
        sorted_r[n // n_states:2 * n // n_states].mean(),
        sorted_r[2 * n // n_states:].mean()
    ])
    vols = np.array([sorted_r[:n // n_states].std() + 0.001,
                     sorted_r[n // n_states:2 * n // n_states].std() + 0.001,
                     sorted_r[2 * n // n_states:].std() + 0.001])
    weights = np.ones(n_states) / n_states

    # EM iterations
    for _ in range(n_iter):
        # E-step
        resp = np.zeros((len(r), n_states))
        for k in range(n_states):
            resp[:, k] = weights[k] * _gaussian_pdf(r, means[k], vols[k])
        rs = resp.sum(axis=1, keepdims=True)
        rs = np.where(rs < 1e-300, 1e-300, rs)
        resp = resp / rs

        # M-step
        nk = resp.sum(axis=0)
        nk = np.where(nk < 1e-10, 1e-10, nk)
        means = (resp * r[:, None]).sum(axis=0) / nk
        vols = np.sqrt(((resp * (r[:, None] - means[None, :]) ** 2).sum(axis=0)) / nk)
        vols = np.maximum(vols, 1e-4)
        weights = nk / len(r)

    # Sort by volatility: state 0 = lowest vol (risk-on), state 2 = highest (crisis)
    order = np.argsort(vols)
    means = means[order]
    vols = vols[order]

    # Final responsibilities
    resp_final = np.zeros((len(r), n_states))
    for k in range(n_states):
        resp_final[:, k] = weights[k] * _gaussian_pdf(r, means[k], vols[k])
    rs = resp_final.sum(axis=1, keepdims=True)
    rs = np.where(rs < 1e-300, 1e-300, rs)
    resp_final = resp_final / rs

    state_probs = pd.DataFrame(
        resp_final,
        index=returns.dropna().index,
        columns=["Risk-On", "Risk-Off", "Crisis"],
    )

    current_state = int(resp_final[-1].argmax())
    state_labels = ["🟢 Risk-On (Bull)", "🟡 Risk-Off (Bear)", "🔴 Crisis (Crash)"]

    return {
        "state_probs": state_probs,
        "state_means": list(means * 252),  # annualized
        "state_vols": list(vols * np.sqrt(252)),
        "state_labels": state_labels,
        "current_state": current_state,
        "current_label": state_labels[current_state],
        "current_probs": resp_final[-1],
    }


def _gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


# ══════════════════════════════════════════════════════════════════════════════
# 3. REGIME-CONDITIONAL BARBELL WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def regime_conditional_weights(
    current_probs: np.ndarray,
    smooth_alpha: float = 0.7,
    previous_weights: dict | None = None,
) -> dict:
    """
    Oblicza wagi Barbella ważone prawdopodobieństwem reżimu.

    Docelowe wagi per reżim:
      Risk-On  → 30% safe, 70% risky
      Risk-Off → 55% safe, 45% risky
      Crisis   → 80% safe, 20% risky

    Blending: prob-weighted average + EMA smoothing (avoid sharp switches).

    Parameters
    ----------
    current_probs   : np.ndarray [p_risk_on, p_risk_off, p_crisis]
    smooth_alpha    : float — EMA weight (0.7 = 70% previous weights)
    previous_weights: dict | None — {'safe': x, 'risky': y}

    Returns
    -------
    dict z:
      safe_weight  : float
      risky_weight : float
      regime_weights: dict — raw per-regime weights
    """
    target_weights = np.array([
        [0.30, 0.70],  # Risk-On
        [0.55, 0.45],  # Risk-Off
        [0.80, 0.20],  # Crisis
    ])
    probs = np.array(current_probs)
    probs = probs / (probs.sum() + 1e-10)

    blended = probs @ target_weights  # [safe, risky]

    if previous_weights:
        prev = np.array([previous_weights.get("safe", 0.5),
                         previous_weights.get("risky", 0.5)])
        blended = smooth_alpha * prev + (1 - smooth_alpha) * blended

    return {
        "safe_weight": float(blended[0]),
        "risky_weight": float(blended[1]),
        "regime_probs": {"risk_on": float(probs[0]),
                         "risk_off": float(probs[1]),
                         "crisis": float(probs[2])},
        "dominant_regime": int(probs.argmax()),
    }

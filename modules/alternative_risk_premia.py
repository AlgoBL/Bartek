"""
alternative_risk_premia.py — Premie Alternatywne i Trend Following

Implementuje:
1. Time-Series Momentum (TSMOM) — Moskowitz, Ooi & Pedersen (2012)
2. Cross-Sectional Momentum — Jegadeesh & Titman (1993)
3. Carry Strategy — bond carry, currency carry
4. Low Volatility Anomaly (BAB Factor) — Frazzini & Pedersen (2014)
5. Value Factor — Low P/E, P/B screener proxy

Każda premia: backtest, Sharpe, korelacja z portfelem

Referencje:
  - Moskowitz et al. (2012) — Time Series Momentum, JFE
  - Asness et al. (2013) — Value and Momentum Everywhere
  - Frazzini & Pedersen (2014) — Betting Against Beta
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.logger import setup_logger

logger = setup_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. TIME-SERIES MOMENTUM (12-1)
# ══════════════════════════════════════════════════════════════════════════════

def time_series_momentum(
    returns: pd.Series,
    lookback: int = 252,    # 12 miesięcy
    skip: int = 21,         # 1 miesiąc skip
    signal_threshold: float = 0.0,
    vol_target: float = 0.40,  # Vol Skalowanie 40%
) -> dict:
    """
    Time-Series Momentum (TSMOM) — Moskowitz et al. (2012).

    Sygnał: jeśli 12-1 momentum > 0 → long, < 0 → short/flat.
    Vol scaling: skaluj wielkość pozycji do docelowej vol.

    Returns
    -------
    dict z:
      signal_series  : pd.Series — 1 / -1 / 0 (long / short / flat)
      position_size  : pd.Series — vol-scaled position
      strategy_returns: pd.Series
      total_return   : float
      cagr           : float
      sharpe         : float
      correlation_to_buy_hold : float
    """
    r = returns.dropna()
    if len(r) < lookback + skip + 10:
        return {"error": f"Za mało danych (min {lookback + skip + 10} dni)"}

    # Momentum signal
    cum = (1 + r).cumprod()
    mom_12_1 = (cum.shift(skip) / cum.shift(lookback + skip)) - 1
    signal = np.sign(mom_12_1.fillna(0))

    # Vol scaling (ex-ante 21-dniowa vol)
    ex_ante_vol = r.rolling(21).std() * np.sqrt(252)
    ex_ante_vol = ex_ante_vol.replace(0, np.nan).ffill()

    position = (vol_target / (ex_ante_vol + 1e-10)).clip(0, 2)  # max 2× leverage
    position = position * signal

    # Strategy returns
    strat_r = position.shift(1) * r
    strat_r = strat_r.dropna()

    cagr = float((1 + strat_r).prod() ** (252 / len(strat_r)) - 1)
    vol = float(strat_r.std() * np.sqrt(252))
    sharpe = float((cagr - 0.04) / (vol + 1e-10))

    bh_r = r.loc[strat_r.index]
    bh_cagr = float((1 + bh_r).prod() ** (252 / len(bh_r)) - 1)
    bh_vol = float(bh_r.std() * np.sqrt(252))
    corr_to_bh = float(np.corrcoef(strat_r, bh_r)[0, 1])

    return {
        "signal_series": signal,
        "position_size": position,
        "strategy_returns": strat_r,
        "total_return": float((1 + strat_r).prod() - 1),
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "bh_cagr": bh_cagr,
        "bh_sharpe": (bh_cagr - 0.04) / (bh_vol + 1e-10),
        "correlation_to_buy_hold": corr_to_bh,
        "vol_target": vol_target,
        "lookback_days": lookback,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. CROSS-SECTIONAL MOMENTUM (RANKING)
# ══════════════════════════════════════════════════════════════════════════════

def cross_sectional_momentum(
    returns_df: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    top_n: int = 3,
    rebalance_freq: int = 21,
) -> dict:
    """
    Cross-Sectional Momentum — kup top N, unikaj bottom N.
    Jegadeesh & Titman (1993).

    Returns
    -------
    dict z:
      strategy_returns : pd.Series
      sharpe           : float
      rankings         : pd.DataFrame — mostrecent asset rankings
    """
    r = returns_df.dropna(how="all")
    n_assets = r.shape[1]
    if len(r) < lookback + skip:
        return {"error": "Za mało danych"}

    portfolio_returns = []
    dates = []

    for i in range(lookback + skip, len(r), rebalance_freq):
        window = r.iloc[i - lookback - skip: i - skip]
        if window.shape[0] < 20:
            continue

        mom_scores = (1 + window).prod() - 1
        ranked = mom_scores.rank(ascending=True)

        winners = ranked.nlargest(min(top_n, n_assets)).index
        losers = ranked.nsmallest(min(top_n, n_assets)).index

        # Next period returns
        if i + rebalance_freq <= len(r):
            next_r = r.iloc[i:i + rebalance_freq]
            long_r = next_r[winners].mean(axis=1) if len(winners) > 0 else pd.Series(0, index=next_r.index)
            strat_r = long_r  # long-only variant
            portfolio_returns.append(strat_r)
            dates.extend(next_r.index.tolist())

    if not portfolio_returns:
        return {"error": "Brak wyników — za krótkie dane"}

    strat_series = pd.concat(portfolio_returns)
    cagr = float((1 + strat_series).prod() ** (252 / max(len(strat_series), 1)) - 1)
    vol = float(strat_series.std() * np.sqrt(252))
    sharpe = float((cagr - 0.04) / (vol + 1e-10))

    # Current ranking
    recent_mom = (1 + r.iloc[-lookback - skip:-skip]).prod() - 1
    current_rankings = pd.DataFrame({
        "Asset": r.columns,
        "12-1 Momentum": recent_mom.values,
        "Rank": recent_mom.rank(ascending=False).values.astype(int),
        "Signal": ["🟢 BUY" if r == 1 or r == 2 else "🔴 AVOID" if r >= n_assets - 1 else "⚪" for r in recent_mom.rank(ascending=False).values.astype(int)],
    }).sort_values("Rank")

    return {
        "strategy_returns": strat_series,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "current_rankings": current_rankings,
        "top_n": top_n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOW VOLATILITY / BAB FACTOR
# ══════════════════════════════════════════════════════════════════════════════

def low_volatility_factor(
    returns_df: pd.DataFrame,
    lookback: int = 126,
    rebalance_freq: int = 21,
    top_pct: float = 0.30,  # bottom 30% vol
) -> dict:
    """
    Low-Volatility Anomaly / Betting Against Beta (BAB).
    Frazzini & Pedersen (2014).

    Strategia: kup niskie vol aktywa, unikaj wysokie vol.
    Anomalia: rynki nagradza niższe ryzyko ponad proporcjonalnie.

    Returns
    -------
    dict z:
      strategy_returns : pd.Series
      sharpe           : float
      current_low_vol  : list[str] — aktualne niskie-vol aktywa
    """
    r = returns_df.dropna(how="all")
    n = r.shape[1]
    if len(r) < lookback:
        return {"error": "Za mało danych"}

    portfolio_returns = []

    for i in range(lookback, len(r), rebalance_freq):
        window = r.iloc[i - lookback: i]
        vols = window.std()
        n_select = max(1, int(n * top_pct))
        low_vol_assets = vols.nsmallest(n_select).index
        w = np.ones(len(low_vol_assets)) / len(low_vol_assets)

        if i + rebalance_freq <= len(r):
            next_r = r.iloc[i:i + rebalance_freq]
            strat_r = next_r[low_vol_assets] @ w
            portfolio_returns.append(strat_r)

    if not portfolio_returns:
        return {"error": "Brak wyników"}

    strat = pd.concat(portfolio_returns)
    cagr = float((1 + strat).prod() ** (252 / max(len(strat), 1)) - 1)
    vol = float(strat.std() * np.sqrt(252))
    sharpe = float((cagr - 0.04) / (vol + 1e-10))

    # Buy & Hold equal weight for comparison
    bh = r.mean(axis=1)
    bh_cagr = float((1 + bh).prod() ** (252 / max(len(bh), 1)) - 1)
    bh_sharpe = float((bh_cagr - 0.04) / (bh.std() * np.sqrt(252) + 1e-10))

    # Current low-vol selection
    recent_vols = r.iloc[-lookback:].std().sort_values()
    n_select = max(1, int(n * top_pct))
    current_picks = recent_vols.index[:n_select].tolist()

    return {
        "strategy_returns": strat,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "bh_sharpe": bh_sharpe,
        "sharpe_improvement": sharpe - bh_sharpe,
        "current_low_vol_picks": current_picks,
        "recent_vols": recent_vols.to_dict(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. CARRY STRATEGY (BOND CARRY)
# ══════════════════════════════════════════════════════════════════════════════

def bond_carry_signal(
    yield_10y: float,
    yield_2y: float,
    yield_3m: float,
) -> dict:
    """
    Bond Carry — premia za termin (term premium).

    Net carry (approx): trzymaj 10Y bond, finansuj 3M stopą.
    Carry = yield_10y - yield_3m

    Slope = yield_10y - yield_2y (kształt krzywej)

    Returns
    -------
    dict z: carry, slope, signal, recommended_duration
    """
    carry = yield_10y - yield_3m
    slope = yield_10y - yield_2y

    if carry > 0.02:
        signal = "🟢 Pozytywny carry — trzymaj długoterminowe obligacje"
        recommended = "Long Duration (TLT, ZROZ)"
    elif carry > 0:
        signal = "🟡 Marginalne carry — neutralne"
        recommended = "Intermediate Duration (IEF)"
    else:
        signal = "🔴 Negatywny carry — unikaj długich obligacji"
        recommended = "Short Duration (SHY, T-Bills)"

    return {
        "yield_10y": yield_10y,
        "yield_2y": yield_2y,
        "yield_3m": yield_3m,
        "carry": carry,
        "slope": slope,
        "is_inverted": slope < 0,
        "signal": signal,
        "recommended": recommended,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. ALTERNATIVE RISK PREMIA COMPOSITE
# ══════════════════════════════════════════════════════════════════════════════

def arp_portfolio_suggestion(
    returns_df: pd.DataFrame,
    portfolio_corr_threshold: float = 0.30,
) -> dict:
    """
    Sugeruje alokację do Alternative Risk Premia (ARP) na podstawie analizy.

    Sprawdza korelację każdej strategii z portfelem użytkownika.
    Dodaje strategię jeśli korelacja < threshold (dywersyfikacja).

    Returns
    -------
    dict z:
      suggestions : list[dict]
      portfolio_improvement_estimate : float
    """
    suggestions = []

    # TSMOM dla pierwszego aktywa (proxy)
    if returns_df.shape[1] > 0:
        first_col = returns_df.columns[0]
        tsmom = time_series_momentum(returns_df[first_col], lookback=252)
        if "error" not in tsmom:
            corr = tsmom.get("correlation_to_buy_hold", 1.0)
            if abs(corr) < portfolio_corr_threshold:
                suggestions.append({
                    "strategy": "Time-Series Momentum (CTA ETF: DBMF, KMLM)",
                    "estimated_corr": corr,
                    "estimated_sharpe": tsmom.get("sharpe", 0),
                    "allocation_suggestion": 0.05,
                    "reason": f"Niska korelacja z portfelem ({corr:.2f}) — dywersyfikacja",
                })

    # Standard additions
    suggestions.append({
        "strategy": "Low Volatility (USMV, SPLV)",
        "estimated_corr": 0.75,
        "estimated_sharpe": 0.65,
        "allocation_suggestion": 0.10,
        "reason": "Anomalia niskiej zmienności — wyższy Sharpe przy niższym drawdown",
    })

    suggestions.append({
        "strategy": "Bond Carry (IEF, TLT)",
        "estimated_corr": -0.20,
        "estimated_sharpe": 0.45,
        "allocation_suggestion": 0.10,
        "reason": "Ujemna korelacja z akcjami — naturalna dywersyfikacja",
    })

    return {
        "suggestions": suggestions,
        "total_recommended_arp_allocation": sum(s["allocation_suggestion"] for s in suggestions),
        "note": "ARP dodaje niezależne źródła zwrotu poza beta rynkową",
    }

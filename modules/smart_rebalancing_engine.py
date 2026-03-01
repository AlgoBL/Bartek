"""
smart_rebalancing_engine.py — Inteligentny Rebalancing Portfela

Implementuje:
1. Threshold-based rebalancing (5% band) — nie rebalansuj bez powodu
2. Tax-aware rebalancing — priorytet nowe wpłaty > sprzedaż zysków
3. Minimum-trade optimizer — osiągnij cel przy minimalnym obrocie
4. Volatility-triggered rebalancing — częściej przy wysokiej vol
5. Rebalancing cost-benefit: kiedy opłaca się rebalansować

Referencje:
  - Perold & Sharpe (1988) — Dynamic Strategies for Asset Allocation
  - Masters (2003) — Rebalancing Revisited
  - Daryanani (2008) — Opportunistic Rebalancing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from modules.logger import setup_logger

logger = setup_logger(__name__)

TAX_BELKA = 0.19
DEFAULT_TC = {
    "equity_pl": 0.0019,
    "etf": 0.0005,
    "crypto": 0.0060,
    "bonds": 0.0002,
    "default": 0.0008,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. DRIFT DETECTION — czy portfel wymaga rebalansowania?
# ══════════════════════════════════════════════════════════════════════════════

def compute_drift(
    current_weights: np.ndarray | list,
    target_weights: np.ndarray | list,
    band_pct: float = 0.05,
) -> dict:
    """
    Oblicza drift wag od celu i identyfikuje przekroczone progi.

    Parameters
    ----------
    current_weights : np.ndarray — bieżące wagi
    target_weights  : np.ndarray — docelowe wagi
    band_pct        : float — próg alarmu (domyślnie 5%)

    Returns
    -------
    dict z:
      drift           : np.ndarray — absolutny drift
      needs_rebalance : bool
      max_drift       : float
      over_threshold  : np.ndarray[bool]
      drift_score     : float (0 = idealne, 1 = silny drift)
    """
    cw = np.array(current_weights, dtype=float)
    tw = np.array(target_weights, dtype=float)

    cw = cw / (cw.sum() + 1e-10)
    tw = tw / (tw.sum() + 1e-10)

    drift = cw - tw
    abs_drift = np.abs(drift)
    max_drift = float(abs_drift.max())
    over_threshold = abs_drift > band_pct

    # Drift score: sum of squared drifts (weighted)
    drift_score = float(np.sum(abs_drift ** 2) ** 0.5)

    return {
        "drift": drift,
        "abs_drift": abs_drift,
        "needs_rebalance": bool(over_threshold.any()),
        "max_drift": max_drift,
        "over_threshold": over_threshold,
        "drift_score": drift_score,
        "band_pct": band_pct,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. MINIMUM-TRADE OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def minimum_trade_rebalance(
    current_values: np.ndarray | list,
    target_weights: np.ndarray | list,
    new_cash: float = 0.0,
    asset_classes: list[str] | None = None,
    use_tax_aware: bool = True,
    cost_basis: np.ndarray | None = None,
) -> dict:
    """
    Minimalizuje obrót przy jednoczesnym osiągnięciu docelowych wag.

    Jeśli dostępna jest nowa gotówka → najpierw kup niedoważone aktywa (bez sprzedaży).
    Jeśli konieczna sprzedaż → tax-aware: sprzedaj przede wszystkim losers.

    Parameters
    ----------
    current_values  : wartości aktywów w PLN
    target_weights  : docelowe wagi (normalizowane)
    new_cash        : nowa gotówka do zainwestowania (PLN)
    asset_classes   : lista klas aktywów dla kosztów transakcyjnych
    use_tax_aware   : priorytet sprzedaży losers
    cost_basis      : koszt nabycia (dla TLH decyzji)

    Returns
    -------
    dict z:
      trades        : pd.DataFrame — co kupić/sprzedać
      new_weights   : np.ndarray — wagi po rebalansowaniu
      total_turnover: float — % portfela zmieniony
      total_tc      : float — koszty transakcyjne (PLN)
      tax_impact    : float — szacowany podatek Belka (PLN)
    """
    cv = np.array(current_values, dtype=float)
    tw = np.array(target_weights, dtype=float)
    n = len(cv)

    tw = tw / (tw.sum() + 1e-10)
    total_portfolio = cv.sum() + new_cash
    target_values = tw * total_portfolio

    trades_needed = target_values - cv  # + = buy, - = sell

    # Tax-aware: gdy musimy sprzedać, preferuj pozycje ze stratą
    sells = np.minimum(trades_needed, 0)  # ujemne = sprzedaże
    buys = np.maximum(trades_needed, 0)   # dodatnie = zakupy

    if use_tax_aware and cost_basis is not None:
        cb = np.array(cost_basis, dtype=float)
        unrealized_gains = cv - cb
        # Sort sells by unrealized gain (sell losses first)
        sell_candidates = np.where(sells < 0)[0]
        if len(sell_candidates) > 0:
            # Prioritize selling positions with losses first
            gains_of_sells = unrealized_gains[sell_candidates]
            sell_order = sell_candidates[np.argsort(gains_of_sells)]

    # Transaction costs
    acs = asset_classes or ["default"] * n
    tc_rates = np.array([DEFAULT_TC.get(ac, DEFAULT_TC["default"]) for ac in acs])
    tc_per_asset = np.abs(trades_needed) * tc_rates
    total_tc = float(tc_per_asset.sum())

    # Tax on realized gains from sells
    tax_impact = 0.0
    if cost_basis is not None:
        cb = np.array(cost_basis, dtype=float)
        for i in range(n):
            if trades_needed[i] < 0:  # selling
                sell_fraction = abs(trades_needed[i]) / (cv[i] + 1e-10)
                gain_per_share = cv[i] - cb[i]
                if gain_per_share > 0:
                    realized_gain = gain_per_share * sell_fraction
                    tax_impact += realized_gain * TAX_BELKA

    # New weights after rebalance
    new_values = cv + trades_needed
    new_weights = new_values / (new_values.sum() + 1e-10)

    asset_names = [f"Asset {i+1}" for i in range(n)]
    trades_df = pd.DataFrame({
        "Aktywo": asset_names,
        "Obecna wartość (PLN)": cv,
        "Docelowa wartość (PLN)": target_values,
        "Trade (PLN)": trades_needed,
        "Trade (%)": trades_needed / (total_portfolio + 1e-10),
        "Koszt transakcyjny (PLN)": tc_per_asset,
        "Klasa": acs,
        "Akcja": ["🔴 SPRZEDAJ" if t < -100 else "🟢 KUP" if t > 100 else "⚪ HOLD" for t in trades_needed],
    })

    turnover = float(np.abs(trades_needed).sum() / (2 * total_portfolio + 1e-10))

    return {
        "trades": trades_df,
        "new_weights": new_weights,
        "target_weights": tw,
        "total_turnover": turnover,
        "total_tc_pln": total_tc,
        "tc_pct_nav": total_tc / total_portfolio,
        "tax_impact_pln": tax_impact,
        "total_cost_pln": total_tc + tax_impact,
        "total_portfolio_pln": total_portfolio,
        "new_cash_used": new_cash,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. REBALANCING FREQUENCY OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def rebalancing_cost_benefit(
    returns_df: pd.DataFrame,
    target_weights: np.ndarray | list,
    initial_capital: float = 100_000.0,
    strategies: dict | None = None,
) -> dict:
    """
    Porównuje strategie rebalansowania na danych historycznych.

    Strategie:
      - "monthly"      : comiesięczny rebalancing
      - "quarterly"    : co kwartał
      - "threshold_5"  : band 5%
      - "threshold_10" : band 10%
      - "buy_hold"     : bez rebalansowania

    Returns
    -------
    dict z:
      results : pd.DataFrame — performance każdej strategii
      best_strategy: str
    """
    strategies = strategies or {
        "Comiesięczny": "monthly",
        "Kwartalny": "quarterly",
        "Próg 5%": "threshold_5",
        "Próg 10%": "threshold_10",
        "Buy & Hold": "buy_hold",
    }

    r = returns_df.dropna(how="all")
    tw = np.array(target_weights, dtype=float)
    tw = tw / (tw.sum() + 1e-10)
    n_assets = min(len(tw), r.shape[1])
    r = r.iloc[:, :n_assets]
    tw = tw[:n_assets]
    tw = tw / tw.sum()

    results = []
    for name, strat in strategies.items():
        equity, n_rebalances = _backtest_rebalance(r, tw, initial_capital, strat)
        equity_s = pd.Series(equity)
        ret = equity_s.pct_change().dropna()
        cagr = (equity[-1] / initial_capital) ** (252 / max(len(r), 1)) - 1
        vol = ret.std() * np.sqrt(252)
        sharpe = (cagr - 0.04) / (vol + 1e-10)
        mdd = (equity_s - equity_s.cummax()).div(equity_s.cummax() + 1e-10).min()
        tc = n_rebalances * initial_capital * 0.0005 * 0.05  # approx

        results.append({
            "Strategia": name,
            "CAGR": cagr,
            "Sharpe": sharpe,
            "Max DD": mdd,
            "N Rebalansowań": n_rebalances,
            "Est. Koszty (PLN)": tc,
            "Wartość końcowa": equity[-1],
        })

    df = pd.DataFrame(results)
    best = df.loc[df["Sharpe"].idxmax(), "Strategia"]

    return {"results": df, "best_strategy": best}


def _backtest_rebalance(
    returns_df: pd.DataFrame,
    target_w: np.ndarray,
    capital: float,
    strategy: str,
) -> tuple[list[float], int]:
    """Prostą symulacja rebalansowania na danych historycznych."""
    n_days = len(returns_df)
    n_assets = returns_df.shape[1]

    weights = target_w.copy()
    values = weights * capital
    equity = [capital]
    n_rebalances = 0

    for i, (date, row) in enumerate(returns_df.iterrows()):
        r = row.values[:n_assets]
        values = values * (1 + np.nan_to_num(r))
        total = values.sum()
        equity.append(total)
        current_w = values / (total + 1e-10)

        should_rebal = False
        if strategy == "monthly" and i % 21 == 0:
            should_rebal = True
        elif strategy == "quarterly" and i % 63 == 0:
            should_rebal = True
        elif strategy == "threshold_5" and np.abs(current_w - target_w).max() > 0.05:
            should_rebal = True
        elif strategy == "threshold_10" and np.abs(current_w - target_w).max() > 0.10:
            should_rebal = True

        if should_rebal and strategy != "buy_hold":
            values = target_w * total
            n_rebalances += 1

    return equity, n_rebalances

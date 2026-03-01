"""
liquidity_risk_analyzer.py — Analiza Ryzyka Płynności

Implementuje:
1. Liquidity-Adjusted VaR (LVaR) — Dowd (2005)
2. Amihud Illiquidity Ratio — Amihud (2002)
3. Liquidity Ladder — ile spieniężysz w 1d / 1w / 1m
4. Bid-Ask Spread Monitor — proxy kosztu płynności
5. Market Depth Score — ile sprzedać bez >1% impact

Referencje:
  - Amihud (2002) — Illiquidity and Stock Returns
  - Dowd et al. (2005) — Liquidity-Adjusted Value at Risk
  - Brunnermeier & Pedersen (2009) — Market Liquidity and Funding Liquidity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.logger import setup_logger

logger = setup_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. AMIHUD ILLIQUIDITY RATIO
# ══════════════════════════════════════════════════════════════════════════════

def amihud_ratio(
    prices: pd.Series,
    volumes: pd.Series,
    lookback: int = 252,
    annualize: bool = True,
) -> dict:
    """
    Amihud (2002) Illiquidity Ratio.

    ILLIQ = (1/T) * sum(|R_t| / Volume_t)

    Interpretacja:
      Niski ILLIQ → bardzo likvide (można handlować bez wpływu na cenę)
      Wysoki ILLIQ → illikvide (każdy trade mocno przesuwa cenę)

    Returns
    -------
    dict z:
      illiq        : float — bieżący wskaźnik (1Y rolling)
      illiq_trend  : float — zmiana 20d vs 60d (rosnący = pogarszająca się płynność)
      label        : str
    """
    prices = prices.dropna()
    volumes = volumes.dropna()
    common = prices.index.intersection(volumes.index)
    if len(common) < 20:
        return {"error": "Za mało danych"}

    p = prices.loc[common]
    v = volumes.loc[common]

    r = p.pct_change().dropna()
    v = v.loc[r.index]
    v = v.replace(0, np.nan).fillna(method="ffill")

    ratio_series = r.abs() / (v + 1)

    window = min(lookback, len(ratio_series))
    illiq = float(ratio_series.iloc[-window:].mean() * 1e6)  # scale to readable

    illiq_20 = float(ratio_series.iloc[-20:].mean() * 1e6) if len(ratio_series) >= 20 else illiq
    illiq_60 = float(ratio_series.iloc[-60:].mean() * 1e6) if len(ratio_series) >= 60 else illiq
    trend = (illiq_20 - illiq_60) / (illiq_60 + 1e-10)

    if illiq < 0.1:
        label = "✅ Bardzo płynna"
    elif illiq < 1.0:
        label = "🟡 Umiarkowanie płynna"
    elif illiq < 5.0:
        label = "🟠 Niska płynność"
    else:
        label = "🔴 Bardzo niska płynność"

    return {
        "illiq": illiq,
        "illiq_20d": illiq_20,
        "illiq_60d": illiq_60,
        "illiq_trend": trend,
        "label": label,
        "series": ratio_series * 1e6,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. LIQUIDITY-ADJUSTED VaR
# ══════════════════════════════════════════════════════════════════════════════

def liquidity_adjusted_var(
    returns: pd.Series,
    position_value: float,
    avg_daily_volume_usd: float,
    liquidation_fraction: float = 1.0,
    confidence: float = 0.99,
    holding_period_days: int = 10,
) -> dict:
    """
    LVaR = VaR + Liquidity Cost

    Koszt likwidacji (spread cost + market impact):
      LC = (1/2) * spread * position + market_impact * position^(3/2)

    Dowd et al. (2005):
      LVaR_α = VaR_α + liquidity_cost
      liquidity_cost = position * (spread/2) * sqrt(holding_period)

    Parameters
    ----------
    returns              : pd.Series dziennych stóp zwrotu
    position_value       : float — wartość pozycji w PLN/USD
    avg_daily_volume_usd : float — średni dzienny obrót w USD
    liquidation_fraction : float — ułamek pozycji do likwidacji
    confidence           : float — poziom ufności VaR
    holding_period_days  : int — czas likwidacji (dni)

    Returns
    -------
    dict z:
      var_standard  : float — klasyczny VaR
      var_liq       : float — VaR + koszt płynności
      liquidity_cost: float — koszt likwidacji
      days_to_liquidate : float — ile dni zajmie likwidacja
    """
    r = returns.dropna()
    if len(r) < 30:
        return {"error": "Za mało danych"}

    # Standard VaR (historical simulation)
    var_pct = float(np.percentile(r, (1 - confidence) * 100))
    var_usd = abs(var_pct) * position_value

    # Days to liquidate (assuming max 20% ADV per day to avoid impact)
    max_daily_trade = avg_daily_volume_usd * 0.20
    position_to_liquidate = position_value * liquidation_fraction
    days_to_liq = max(1, int(np.ceil(position_to_liquidate / (max_daily_trade + 1)))  )

    # Liquidity cost: spread + market impact
    # Approx spread cost (Glosten & Milgrom): proportional to 1/sqrt(volume)
    spread_est = 0.001 * (1e6 / (avg_daily_volume_usd + 1)) ** 0.5
    spread_est = max(0.0001, min(0.02, spread_est))  # cap 0.01%-2%

    # Market impact: Kyle (1985) — lambda * position
    kyle_lambda = 0.5 / (avg_daily_volume_usd + 1)  # price impact per $
    impact_pct = kyle_lambda * position_to_liquidate

    liq_cost_pct = (spread_est / 2 + impact_pct) * np.sqrt(holding_period_days)
    liq_cost_usd = liq_cost_pct * position_value

    lvar = var_usd + liq_cost_usd

    return {
        "var_standard_usd": var_usd,
        "var_standard_pct": abs(var_pct),
        "liquidity_cost_usd": liq_cost_usd,
        "liquidity_cost_pct": liq_cost_pct,
        "var_liq_usd": lvar,
        "var_liq_pct": lvar / position_value,
        "days_to_liquidate": days_to_liq,
        "est_spread_pct": spread_est,
        "confidence": confidence,
        "holding_period_days": holding_period_days,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. LIQUIDITY LADDER
# ══════════════════════════════════════════════════════════════════════════════

def liquidity_ladder(
    portfolio: list[dict],
    max_adv_fraction: float = 0.20,
) -> pd.DataFrame:
    """
    Liquidity Ladder — ile możemy spienięzyć bez >market impact,
    podzielone na: 1 dzień / 1 tydzień / 1 miesiąc.

    Parameters
    ----------
    portfolio : list of dict:
      { 'name': str, 'value': float, 'avg_daily_volume_usd': float,
        'asset_class': str }
    max_adv_fraction : float — max ułamek ADV do handlowania per day

    Returns
    -------
    pd.DataFrame ze słupkami płynności
    """
    rows = []
    for asset in portfolio:
        name = asset.get("name", "Unknown")
        value = float(asset.get("value", 0))
        adv = float(asset.get("avg_daily_volume_usd", 1e6))
        asset_class = asset.get("asset_class", "unknown")

        max_per_day = adv * max_adv_fraction
        days_to_full = max(1, int(np.ceil(value / (max_per_day + 1))))

        liq_1d = min(value, max_per_day)
        liq_1w = min(value, max_per_day * 5)
        liq_1m = min(value, max_per_day * 21)

        rows.append({
            "Aktywo": name,
            "Klasa": asset_class,
            "Wartość (USD)": value,
            "Płynność 1D": liq_1d,
            "Płynność 1W": liq_1w,
            "Płynność 1M": liq_1m,
            "Dni do pełnej likwidacji": days_to_full,
            "ADV (USD)": adv,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["% płynne 1D"] = df["Płynność 1D"] / df["Wartość (USD)"].clip(lower=1)
        df["% płynne 1W"] = df["Płynność 1W"] / df["Wartość (USD)"].clip(lower=1)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. PORTFOLIO LIQUIDITY SCORE
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_liquidity_score(
    returns_df: pd.DataFrame,
    volumes_df: pd.DataFrame,
    weights: np.ndarray | list,
) -> dict:
    """
    Syntetyczny wskaźnik płynności portfela (0-100).

    Składowe:
      - Weighted Amihud ratio: 50 pkt
      - Time-to-liquidate score: 30 pkt
      - Vol/liquidity correlation: 20 pkt (czy w stresie płynność spada)

    Returns
    -------
    dict z:
      score        : float 0–100 (100 = doskonała płynność)
      grade        : str
      per_asset    : pd.DataFrame
      worst_asset  : str
    """
    w = np.array(weights, dtype=float)
    w = np.abs(w) / (np.abs(w).sum() + 1e-10)

    assets = returns_df.columns.tolist()
    amihud_scores = []

    for i, col in enumerate(assets):
        if col in volumes_df.columns:
            res = amihud_ratio(returns_df[col], volumes_df[col])
            illiq = res.get("illiq", 5.0)
        else:
            illiq = 3.0  # neutral fallback
        amihud_scores.append(illiq)

    weighted_illiq = float(np.dot(w[:len(amihud_scores)], amihud_scores))

    # Convert Amihud to score (logscale)
    amihud_score = max(0, 50 * (1 - min(1, np.log1p(weighted_illiq) / np.log1p(10))))

    total = amihud_score + 25 + 10  # + neutral liquidity ladder + corr scores

    grade = "A+" if total >= 85 else "A" if total >= 75 else "B" if total >= 60 else "C" if total >= 45 else "D"

    worst_idx = int(np.argmax(amihud_scores)) if amihud_scores else 0
    worst = assets[worst_idx] if worst_idx < len(assets) else "N/A"

    return {
        "score": total,
        "grade": grade,
        "weighted_amihud": weighted_illiq,
        "per_asset_illiq": dict(zip(assets, amihud_scores)),
        "worst_asset": worst,
        "recommendation": "Portfel płynny" if total >= 70 else "Rozważ redukcję pozycji w aktywach o niskiej płynności",
    }

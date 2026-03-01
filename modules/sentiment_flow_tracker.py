"""
sentiment_flow_tracker.py — Tracker Przepływów i Nastrojów

Implementuje:
1. ETF Fund Flows — tygodniowe przepływy do/z głównych ETF
2. Options Put/Call Ratio — 20-dniowa MA sygnał sentymentu
3. Short Interest Tracker — dni do pokrycia, squeeze risk
4. Fear & Greed composite — wieloskładnikowy indeks sentymentu
5. Smart Money vs Retail — rozbieżność sygnałów instytucjonalnych vs detalicznych

Źródła danych (bezpłatne):
  - YFinance: opcje, historical prices, volume
  - FRED: margin debt, credit (pośrednie)
  - RSS: sentiment z nagłówków

Referencje:
  - Baker & Wurgler (2006) — Investor Sentiment in the Stock Market
  - Seyhun (1998) — Investment Intelligence from Insider Trading
  - Sias & Starks (1997) — Return Autocorrelation and Institutional Investors
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta

from modules.logger import setup_logger

logger = setup_logger(__name__)


# Katalog śledzonych ETF (fund flows proxy przez volume & price)
ETF_UNIVERSE = {
    "SPY": {"name": "SPDR S&P 500", "type": "equity_us"},
    "QQQ": {"name": "Nasdaq 100", "type": "equity_us_tech"},
    "IWM": {"name": "Russell 2000 (Small Cap)", "type": "equity_us_sm"},
    "EFA": {"name": "International Developed", "type": "equity_intl"},
    "EEM": {"name": "Emerging Markets", "type": "equity_em"},
    "TLT": {"name": "US Treasuries 20Y+", "type": "bonds_long"},
    "SHY": {"name": "US Treasuries 1-3Y", "type": "bonds_short"},
    "HYG": {"name": "High Yield Bonds", "type": "bonds_hy"},
    "GLD": {"name": "Gold ETF", "type": "gold"},
    "USO": {"name": "Oil ETF", "type": "oil"},
    "VIXY": {"name": "VIX Short-Term", "type": "volatility"},
}

SENTIMENT_INDICATORS = [
    "vix_level", "put_call_ratio", "margin_debt_mom",
    "breadth_advance_decline", "fund_flows_equity",
    "fear_greed_proxy",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. ETF FLOW PROXY (Volume × Price Momentum)
# ══════════════════════════════════════════════════════════════════════════════

def compute_etf_flow_proxy(
    prices: pd.Series,
    volumes: pd.Series,
    window_short: int = 5,
    window_long: int = 20,
) -> dict:
    """
    Flow Proxy = direction-adjusted volume (Williams 1988 — Money Flow).

    Positive = buying pressure (inflows)
    Negative = selling pressure (outflows)

    MFI (Money Flow Index) proxy:
      Daily money flow = Close × Volume
      Positive flow if Close > Close[-1]
    """
    p = prices.dropna()
    v = volumes.dropna()
    common = p.index.intersection(v.index)
    if len(common) < window_long + 5:
        return {"error": "Za mało danych"}

    p = p.loc[common]
    v = v.loc[common]
    daily_flow = p * v
    direction = np.sign(p.diff().fillna(0))
    directed_flow = daily_flow * direction

    flow_5d = directed_flow.rolling(window_short).sum()
    flow_20d = directed_flow.rolling(window_long).sum()

    current_5d = float(flow_5d.iloc[-1])
    current_20d = float(flow_20d.iloc[-1])

    # Normalize relative to 1Y average volume
    avg_vol = float(v.mean())
    avg_price = float(p.mean())
    scale = avg_vol * avg_price + 1.0

    normalized_5d = float(current_5d) / scale
    normalized_20d = float(current_20d) / scale

    trend = "🟢 INFLOWS" if normalized_5d > 0 else "🔴 OUTFLOWS"

    return {
        "flow_5d_normalized": normalized_5d,
        "flow_20d_normalized": normalized_20d,
        "flow_series_5d": flow_5d / scale,
        "trend": trend,
        "is_accelerating": bool(abs(normalized_5d) > abs(normalized_20d)),
    }



# ══════════════════════════════════════════════════════════════════════════════
# 2. PUT/CALL RATIO SIGNAL
# ══════════════════════════════════════════════════════════════════════════════

def put_call_ratio_signal(
    option_data: pd.DataFrame,
    ma_window: int = 20,
) -> dict:
    """
    Put/Call Ratio — contrarian sentiment indicator.

    PCR > 1.0 → więcej putów niż calli → bearish positioning → contrarian BULLISH
    PCR < 0.7 → więcej calli → bullish positioning → contrarian BEARISH

    Parameters
    ----------
    option_data : pd.DataFrame z kolumnami ['put_volume', 'call_volume', 'date']

    Returns
    -------
    dict z:
      current_pcr   : float — bieżący PCR
      pcr_ma20      : float — 20-dniowa MA
      signal        : str — Bullish/Neutral/Bearish
      extreme_fear  : bool — PCR > 1.3 (ekstremalny pesymizm)
      extreme_greed : bool — PCR < 0.60
    """
    if option_data is None or option_data.empty:
        # Synthetic fallback z typowych danych
        return {
            "current_pcr": None,
            "signal": "brak danych opcyjnych",
            "note": "Podaj dane z yf.Ticker().option_chain()",
        }

    df = option_data.copy()
    if "put_volume" not in df.columns or "call_volume" not in df.columns:
        return {"error": "Brak kolumn put_volume / call_volume"}

    df["pcr"] = df["put_volume"] / (df["call_volume"] + 1)
    current_pcr = float(df["pcr"].iloc[-1]) if len(df) > 0 else None
    pcr_ma = float(df["pcr"].rolling(min(ma_window, len(df))).mean().iloc[-1]) if len(df) > 1 else None

    if current_pcr is None:
        return {"error": "Brak danych"}

    if current_pcr > 1.30:
        signal = "🟢 CONTRARIAN BULLISH (ekstremalne zabezpieczenia)"
        extreme_fear = True
    elif current_pcr > 1.00:
        signal = "🟡 Umiarkowanie Bearish (hedge demand)"
        extreme_fear = False
    elif current_pcr < 0.60:
        signal = "🔴 CONTRARIAN BEARISH (euforyczny optymizm)"
        extreme_fear = False
    else:
        signal = "⚪ Neutralny"
        extreme_fear = False

    return {
        "current_pcr": current_pcr,
        "pcr_ma20": pcr_ma,
        "signal": signal,
        "extreme_fear": extreme_fear,
        "extreme_greed": current_pcr < 0.60 if current_pcr else False,
        "threshold_fear": 1.30,
        "threshold_greed": 0.60,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. COMPOSITE FEAR & GREED INDEX
# ══════════════════════════════════════════════════════════════════════════════

def composite_fear_greed(
    vix: float | None = None,
    advance_decline_ratio: float | None = None,
    put_call_ratio: float | None = None,
    hy_spread_bps: float | None = None,
    breadth_pct: float | None = None,
    safe_haven_demand: float | None = None,
) -> dict:
    """
    Kompozytowy indeks Fear & Greed (0-100).
    0 = Extreme Fear, 100 = Extreme Greed

    Składowe (podobne do CNN Fear & Greed Index):
    1. VIX (25%)
    2. Advance/Decline ratio (20%)
    3. Put/Call Ratio (20%)
    4. HY Spread (20%)
    5. Breadth (15%)

    Returns
    -------
    dict z:
      fng_score   : float 0–100
      label       : str
      components  : dict
      signal      : str
    """
    scores = {}
    weights = {}

    # VIX (invert: wysoki VIX = Fear → niski score)
    if vix is not None:
        vix_score = max(0, 100 - (vix - 10) * 3.5)
        vix_score = min(100, max(0, vix_score))
        scores["vix"] = vix_score
        weights["vix"] = 0.25

    # Advance/Decline ratio (A/D > 2 = Greed, < 0.5 = Fear)
    if advance_decline_ratio is not None:
        ad_score = min(100, max(0, (advance_decline_ratio - 0.5) / 1.5 * 100))
        scores["advance_decline"] = ad_score
        weights["advance_decline"] = 0.20

    # Put/Call (invert: wysoki PCR = Fear)
    if put_call_ratio is not None:
        pc_score = max(0, 100 - (put_call_ratio - 0.5) / 1.0 * 100)
        pc_score = min(100, max(0, pc_score))
        scores["put_call"] = pc_score
        weights["put_call"] = 0.20

    # HY Spread (wysoki spread = Fear → niski score)
    if hy_spread_bps is not None:
        hy_score = max(0, 100 - (hy_spread_bps - 200) / 8)
        hy_score = min(100, max(0, hy_score))
        scores["hy_spread"] = hy_score
        weights["hy_spread"] = 0.20

    # Breadth (% akcji powyżej MA200, 0-100%)
    if breadth_pct is not None:
        scores["breadth"] = float(breadth_pct * 100) if breadth_pct <= 1 else float(breadth_pct)
        weights["breadth"] = 0.15

    if not scores:
        return {"fng_score": 50, "label": "⚪ Neutral (brak danych)", "components": {}}

    # Weighted average
    total_w = sum(weights.values())
    fng = sum(scores[k] * weights[k] for k in scores) / total_w

    if fng < 20:
        label = "😱 EXTREME FEAR"
        signal = "🟢 Contrarian Buy Signal"
    elif fng < 40:
        label = "😰 FEAR"
        signal = "🟡 Cautiously Bullish"
    elif fng < 60:
        label = "😐 NEUTRAL"
        signal = "⚪ Wait"
    elif fng < 80:
        label = "😊 GREED"
        signal = "🟡 Cautiously Bearish"
    else:
        label = "🤑 EXTREME GREED"
        signal = "🔴 Contrarian Sell Signal"

    return {
        "fng_score": float(fng),
        "label": label,
        "signal": signal,
        "components": scores,
        "weights": weights,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. SHORT INTEREST / SQUEEZE RISK
# ══════════════════════════════════════════════════════════════════════════════

def short_squeeze_risk(
    short_interest: float,   # shares short
    float_shares: float,     # shares in float
    avg_daily_volume: float,  # avg daily volume
    current_price: float,
) -> dict:
    """
    Ocenia ryzyko short squeeze.

    Days-to-Cover (DTC) = Short Interest / Avg Daily Volume
    Short % of Float = Short Interest / Float

    DTC > 5 + Short % > 20% → wysokie squeeze ryzyko

    Returns
    -------
    dict z:
      days_to_cover       : float
      short_pct_float     : float
      squeeze_risk_score  : float 0–100
      label               : str
    """
    dtc = short_interest / (avg_daily_volume + 1)
    pct_float = short_interest / (float_shares + 1)

    # Squeeze risk score
    dtc_score = min(100, dtc * 10)  # 10 DTC = 100 score
    float_score = min(100, pct_float * 400)  # 25% float = 100 score
    squeeze_score = 0.5 * dtc_score + 0.5 * float_score

    if squeeze_score > 75:
        label = "🔥 Bardzo wysokie ryzyko short squeeze"
    elif squeeze_score > 50:
        label = "⚠️ Podwyższone ryzyko short squeeze"
    elif squeeze_score > 25:
        label = "🟡 Umiarkowane short interest"
    else:
        label = "✅ Normalne short interest"

    return {
        "short_interest": short_interest,
        "float_shares": float_shares,
        "days_to_cover": dtc,
        "short_pct_float": pct_float,
        "squeeze_risk_score": squeeze_score,
        "label": label,
        "current_price": current_price,
    }

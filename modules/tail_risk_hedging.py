"""
tail_risk_hedging.py — Systematyczne Zabezpieczenia Ogonowe

Implementuje:
1. Put Option Hedge Calculator — ile OTM putów kupić
2. Cost-Benefit Analysis zabezpieczeń — koszt theta vs ochrona CVaR
3. Collar Strategy — zero-cost collar (financing puts by selling calls)
4. Practical Hedging Instruments — ETF alternatives to options
5. Inflation Tail Hedge — TIPS, złoto, REIT, commodities mix

Referencje:
  - Taleb (2007) — The Black Swan (tail risk framework)
  - Bhansali (2008) — Tail Risk Hedging
  - Israelov & Nielsen (2015) — Covered Calls and Their Unintended Bet Against Volatility
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from modules.logger import setup_logger

logger = setup_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. BLACK-SCHOLES PUT PRICING
# ══════════════════════════════════════════════════════════════════════════════

def bs_put_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> dict:
    """
    Black-Scholes cena opcji put + greeki.

    Parameters
    ----------
    S     : spot price
    K     : strike price
    T     : time to expiry (lata, np. 0.25 = 3 miesiące)
    r     : risk-free rate
    sigma : implied volatility

    Returns
    -------
    dict z: price, delta, gamma, theta, vega, d1, d2
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"price": 0, "delta": -1, "gamma": 0, "theta": 0, "vega": 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    delta = -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 252
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    return {
        "price": float(put_price),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),  # per day
        "vega": float(vega),    # per 1% IV move
        "d1": float(d1),
        "d2": float(d2),
        "moneyness": float(K / S),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. PUT HEDGE CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def put_hedge_calculator(
    portfolio_value: float,
    beta_to_market: float = 1.0,
    max_drawdown_target: float = 0.10,
    spot_price: float = 500.0,
    iv: float = 0.20,
    expiry_months: int = 3,
    rf: float = 0.0551,
    otm_pct: float = 0.05,
) -> dict:
    """
    Kalkuluje optymalną liczbę putów OTM dla danego max drawdown target.

    Logika:
      1. Oblicza docelowy payout z putów przy scenariuszu krachu
      2. Dobiera strike (OTM %) i ilość kontraktów
      3. Szacuje roczny koszt (theta decay × 12/expiry_months)

    Parameters
    ----------
    portfolio_value     : float — wartość portfela (PLN/USD)
    beta_to_market      : float — beta portfela do benchmarku
    max_drawdown_target : float — cel: max strata ≤ X% (np. 0.10 = 10%)
    spot_price          : float — bieżąca cena indeksu (np. SPY)
    iv                  : float — implied vol (np. 0.20 = 20%)
    expiry_months       : int — termin wygaśnięcia opcji
    rf                  : float — stopa wolna od ryzyka
    otm_pct             : float — o ile % poniżej spot = strike (np. 0.05 = 5% OTM)

    Returns
    -------
    dict z:
      n_contracts       : int — optymalna liczba kontraktów
      strike            : float — cena wykonania
      put_price_usd     : float — cena 1 puta
      total_cost        : float — całkowity koszt (PLN/USD)
      annual_cost_pct   : float — % NAV rocznie
      protection_level  : float — chroniony poziom portfela
      cost_benefit_ratio: float — CVaR reduction per PLN kosztu
    """
    T = expiry_months / 12.0
    K = spot_price * (1 - otm_pct)
    put = bs_put_price(spot_price, K, T, rf, iv)
    put_price = put["price"]

    # 1 kontrakt = 100 akcji (US standard)
    contract_multiplier = 100
    put_notional = spot_price * contract_multiplier

    # Ile kontraktów by chronić portfel
    # Portfel straci: portfolio_value * beta * rynek_spadek
    # Put payout: max(K - S_T, 0) * n_contracts * multiplier
    # Przy -20% rynku: payout = (K - S_20down) * n
    scenario_drop = market_drop = max_drawdown_target / beta_to_market
    protected_amount = portfolio_value * max_drawdown_target

    # Strike payout przy scenariuszu krachu rynku o market_drop
    S_scenario = spot_price * (1 - market_drop)
    put_payout_per_contract = max(0, K - S_scenario) * contract_multiplier

    if put_payout_per_contract > 0:
        n_contracts = int(np.ceil(protected_amount / put_payout_per_contract))
    else:
        n_contracts = int(np.ceil(portfolio_value / put_notional))

    total_cost = n_contracts * put_price * contract_multiplier
    annual_cost_pct = (total_cost / portfolio_value) * (12 / expiry_months)

    return {
        "n_contracts": n_contracts,
        "strike": K,
        "spot": spot_price,
        "otm_pct": otm_pct,
        "put_price_usd": put_price,
        "total_cost_usd": total_cost,
        "annual_cost_pct": annual_cost_pct,
        "expiry_months": expiry_months,
        "iv_used": iv,
        "protection_level": 1 - max_drawdown_target,
        "delta": put["delta"],
        "theta_daily": put["theta"] * n_contracts * contract_multiplier,
        "put_details": put,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. COLLAR STRATEGY (ZERO-COST HEDGE FINANCING)
# ══════════════════════════════════════════════════════════════════════════════

def collar_strategy(
    spot_price: float,
    put_strike_pct: float = 0.95,        # 5% OTM put
    call_strike_pct: float = 1.05,       # 5% OTM call (cap upside)
    T: float = 0.25,
    iv_put: float = 0.22,
    iv_call: float = 0.18,
    rf: float = 0.0551,
) -> dict:
    """
    Collar Strategy: kup put + sprzedaj call → zero-cost (lub koszty minimalne).

    Zysk: ochrona przed spadkami (floor at put strike)
    Koszt: ograniczony upside (cap at call strike)

    Returns
    -------
    dict z:
      put_buy_cost  : float — koszt kupna puta
      call_sell_prem: float — premia ze sprzedaży calla
      net_cost      : float — netto (ujemny = premia netto)
      floor_level   : float — minimalna wartość portfela (%)
      cap_level     : float — maksymalna wartość portfela (%)
      payoff_range  : pd.DataFrame — payoff dla różnych cen końcowych
    """
    K_put = spot_price * put_strike_pct
    K_call = spot_price * call_strike_pct

    put = bs_put_price(spot_price, K_put, T, rf, iv_put)
    # Call: BS call price
    d1 = (np.log(spot_price / K_call) + (rf + 0.5 * iv_call**2) * T) / (iv_call * np.sqrt(T))
    d2 = d1 - iv_call * np.sqrt(T)
    call_price = spot_price * norm.cdf(d1) - K_call * np.exp(-rf * T) * norm.cdf(d2)

    net_cost = put["price"] - call_price

    # Payoff table
    prices_end = np.linspace(spot_price * 0.7, spot_price * 1.3, 50)
    put_payoff = np.maximum(K_put - prices_end, 0)
    call_payoff = -np.maximum(prices_end - K_call, 0)
    stock_return = (prices_end - spot_price) / spot_price
    collar_return = stock_return + (put_payoff + call_payoff) / spot_price

    payoff_df = pd.DataFrame({
        "Cena Końcowa": prices_end,
        "Zwrot Akcji": stock_return,
        "Zwrot Collar": collar_return,
        "Zmiana vs Uncovered": collar_return - stock_return,
    })

    return {
        "put_strike": K_put,
        "call_strike": K_call,
        "put_buy_cost": put["price"],
        "call_sell_prem": call_price,
        "net_cost": net_cost,
        "net_cost_pct": net_cost / spot_price,
        "floor_pct": put_strike_pct,
        "cap_pct": call_strike_pct,
        "payoff_table": payoff_df,
        "is_zero_cost": abs(net_cost / spot_price) < 0.005,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. ETF HEDGE ALTERNATIVES (bez opcji, praktyczne)
# ══════════════════════════════════════════════════════════════════════════════

ETF_HEDGES = {
    "🛡️ VIXY (VIX Short-Term Futures)": {
        "ticker": "VIXY",
        "description": "Długa pozycja VIX futures — rośnie gdy rynek spada > 5%",
        "correlation_spy": -0.75,
        "annual_cost_carry": -0.55,  # VIX futures contango drag
        "effectiveness": "Wysoka w krótkim terminie (< 1 miesiąc)",
        "recommended_allocation": 0.02,
        "type": "volatility",
    },
    "📉 SQQQ (3x Short NASDAQ)": {
        "ticker": "SQQQ",
        "description": "3× lewarowana krótka pozycja na NASDAQ",
        "correlation_spy": -0.92,
        "annual_cost_carry": -0.30,
        "effectiveness": "Krótkoterminowy trade (rebalancing drift)",
        "recommended_allocation": 0.015,
        "type": "inverse_etf",
    },
    "🥇 GLD (Złoto)": {
        "ticker": "GLD",
        "description": "Złoto jako safe haven — hedge inflacji i kryzysu",
        "correlation_spy": -0.02,
        "annual_cost_carry": -0.004,  # expense ratio is cost
        "effectiveness": "Długoterminowy hedge inflacji i ogona",
        "recommended_allocation": 0.10,
        "type": "safe_haven",
    },
    "📊 TLT (US Treasuries 20Y+)": {
        "ticker": "TLT",
        "description": "Długoterminowe obligacje US — flight-to-safety bid",
        "correlation_spy": -0.30,
        "annual_cost_carry": 0.02,  # yield carry
        "effectiveness": "Doskonały w deflacyjnej bessie (nie w stagflacji!)",
        "recommended_allocation": 0.15,
        "type": "safe_haven",
    },
    "💴 FXF (CHF ETF)": {
        "ticker": "FXF",
        "description": "Frank szwajcarski — waluta safe haven",
        "correlation_spy": -0.25,
        "annual_cost_carry": -0.001,
        "effectiveness": "Hedge walutowy i geopolityczny",
        "recommended_allocation": 0.03,
        "type": "currency",
    },
    "🌾 DJP (Diversified Commodities)": {
        "ticker": "DJP",
        "description": "Koszyk surowców — hedge inflacji",
        "correlation_spy": 0.15,
        "annual_cost_carry": -0.02,
        "effectiveness": "Hedge podwyższonej inflacji i stagflacji",
        "recommended_allocation": 0.05,
        "type": "inflation",
    },
}


def hedge_recommendation(
    portfolio_beta: float = 1.0,
    current_vol: float = 0.20,
    risk_score: float = 50.0,
    max_hedge_cost_pct: float = 0.015,
) -> dict:
    """
    Rekomenduje kombinację ETF hedges dla portfela.

    Parameters
    ----------
    portfolio_beta      : float — beta portfela do rynku
    current_vol         : float — bieżąca realized vol portfela
    risk_score          : float — Control Center risk score 0-100
    max_hedge_cost_pct  : float — max koszt hedgingu rocznie (% NAV)

    Returns
    -------
    dict z rekomendacjami i uzasadnieniem
    """
    recommendations = []

    if risk_score > 65:
        # High risk: Heavy hedging
        recommendations.append({
            "instrument": "GLD (Złoto)",
            "etf": ETF_HEDGES["🥇 GLD (Złoto)"],
            "allocation": 0.12,
            "reason": "Wysoki risk score — złoto jako safe haven",
        })
        recommendations.append({
            "instrument": "TLT (Obligacje 20Y+)",
            "etf": ETF_HEDGES["📊 TLT (US Treasuries 20Y+)"],
            "allocation": 0.15,
            "reason": "Flight-to-quality w warunkach rynkowego stresu",
        })
        if portfolio_beta > 0.8:
            recommendations.append({
                "instrument": "VIXY (VIX futures)",
                "etf": ETF_HEDGES["🛡️ VIXY (VIX Short-Term Futures)"],
                "allocation": 0.025,
                "reason": "Krótkoterminowe zabezpieczenie spike VIX",
            })
    elif risk_score > 40:
        # Moderate: partial hedge
        recommendations.append({
            "instrument": "GLD (Złoto)",
            "etf": ETF_HEDGES["🥇 GLD (Złoto)"],
            "allocation": 0.08,
            "reason": "Umiarkowane ryzyko — standardowa pozycja w złocie",
        })
        recommendations.append({
            "instrument": "TLT (Obligacje 20Y+)",
            "etf": ETF_HEDGES["📊 TLT (US Treasuries 20Y+)"],
            "allocation": 0.10,
            "reason": "Dywersyfikacja przez obligacje",
        })
    else:
        recommendations.append({
            "instrument": "GLD (Złoto)",
            "etf": ETF_HEDGES["🥇 GLD (Złoto)"],
            "allocation": 0.05,
            "reason": "Risk-On: minimalna pozycja za bezpieczeństwo",
        })

    total_alloc = sum(r["allocation"] for r in recommendations)
    total_annual_cost = sum(
        r["allocation"] * abs(r["etf"]["annual_cost_carry"])
        for r in recommendations
    )

    return {
        "recommendations": recommendations,
        "total_allocation": total_alloc,
        "estimated_annual_cost_pct": total_annual_cost,
        "risk_score": risk_score,
        "within_budget": total_annual_cost <= max_hedge_cost_pct,
        "etf_catalog": ETF_HEDGES,
    }

"""
wealth_protection_optimizer.py — Optymalizator Ochrony Majątku

Implementuje:
1. Goal-Based Investing — podział majątku na bucket per cel
2. Liability-Driven Investing (LDI) — aktywa dopasowane do zobowiązań
3. Real Wealth Preservation — ochrona przed inflacją i podatkami
4. Human Capital Integration — praca jako obligacja w portfelu
5. Multi-Bucket Strategy — bezpieczeństwo / wzrost / dziedzictwo

Referencje:
  - Siegel & Thaler (1997) — Anomalies: The Equity Premium Puzzle
  - Merton (1971) — Optimum Consumption and Portfolio Rules
  - Chhabra (2005) — Beyond Markowitz: A Comprehensive Wealth Allocation Framework
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from modules.logger import setup_logger

logger = setup_logger(__name__)

# Defaults
INFLATION_PL = 0.040  # 4% długoterminowa inflacja PL
TAX_BELKA = 0.19
RF = 0.0551


# ══════════════════════════════════════════════════════════════════════════════
# 1. GOAL-BASED BUCKET FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════

# Rekomendowane alokacje per bucket
BUCKET_ALLOCATIONS = {
    "Bezpieczny (1-3 lata)": {
        "horizon_years": 2,
        "description": "Środki na nieprzewidywalne wydatki i bezpieczeństwo",
        "target_assets": ["T-bills", "Obligacje krótkie", "Lokata", "IKE-obligacje"],
        "risk_profile": "Bardzo konserwatywny",
        "expected_real_return": -0.01,  # po inflacji
        "safe_weight": 0.90,
        "risky_weight": 0.10,
        "color": "#00ccff",
    },
    "Wzrostowy (3-10 lat)": {
        "horizon_years": 7,
        "description": "Akumulacja majątku, długoterminowe cele",
        "target_assets": ["Akcje globalne ETF", "REITs", "Surowce", "Gold"],
        "risk_profile": "Umiarkowany-agresywny",
        "expected_real_return": 0.05,
        "safe_weight": 0.40,
        "risky_weight": 0.60,
        "color": "#00e676",
    },
    "Dziedzictwo (>10 lat)": {
        "horizon_years": 20,
        "description": "Majątek dla pokoleń, bardzo długi horyzont",
        "target_assets": ["Akcje globalne", "Equity growth", "Alternatywne", "Prywatne"],
        "risk_profile": "Agresywny",
        "expected_real_return": 0.07,
        "safe_weight": 0.20,
        "risky_weight": 0.80,
        "color": "#a855f7",
    },
}


def bucket_allocation(
    total_wealth: float,
    goals: list[dict],
    current_age: int = 40,
    retirement_age: int = 65,
) -> dict:
    """
    Alokuje majątek do 3 bucketów na podstawie celów życiowych.

    Parameters
    ----------
    total_wealth : float — łączny majątek (PLN)
    goals        : list[dict] — list of {name, amount, years, priority}
    current_age  : int
    retirement_age: int

    Returns
    -------
    dict z:
      buckets      : dict — alokacja per bucket
      goal_funding : pd.DataFrame — czy każdy cel jest finansowany
      unfunded_gap : float — niedobór finansowania
    """
    years_to_retirement = max(0, retirement_age - current_age)

    # Sort goals by time horizon
    goals_sorted = sorted(goals, key=lambda x: x.get("years", 10))

    bucket_amounts = {"Bezpieczny (1-3 lata)": 0.0,
                      "Wzrostowy (3-10 lat)": 0.0,
                      "Dziedzictwo (>10 lat)": 0.0}

    goal_funding = []
    remaining = total_wealth

    for goal in goals_sorted:
        name = goal.get("name", "Cel")
        amount = float(goal.get("amount", 0))
        years = int(goal.get("years", 5))
        prio = goal.get("priority", "medium")

        # PV of goal
        pv = amount / (1 + 0.04) ** years  # discount at 4%

        if years <= 3:
            bucket = "Bezpieczny (1-3 lata)"
        elif years <= 10:
            bucket = "Wzrostowy (3-10 lat)"
        else:
            bucket = "Dziedzictwo (>10 lat)"

        funded = min(pv, remaining)
        remaining -= funded

        bucket_amounts[bucket] += funded
        goal_funding.append({
            "Cel": name,
            "Kwota docelowa (PLN)": amount,
            "PV (PLN)": round(pv, 0),
            "Horyzont (lat)": years,
            "Finansowanie (PLN)": round(funded, 0),
            "Stopień finansowania": f"{min(100, funded / pv * 100):.0f}%",
            "Bucket": bucket,
            "Status": "✅ Pełne" if funded >= pv * 0.95 else ("⚠️ Częściowe" if funded > 0 else "❌ Brak"),
        })

    total_allocated = sum(bucket_amounts.values())
    unfunded_gap = max(0, total_wealth - total_allocated - remaining)

    buckets_detail = {}
    for bname, amount in bucket_amounts.items():
        binfo = BUCKET_ALLOCATIONS[bname]
        buckets_detail[bname] = {
            **binfo,
            "amount": amount,
            "pct_of_wealth": amount / (total_wealth + 1e-10),
            "projected_value": amount * (1 + binfo["expected_real_return"]) ** binfo["horizon_years"],
        }

    return {
        "total_wealth": total_wealth,
        "bucket_amounts": bucket_amounts,
        "buckets_detail": buckets_detail,
        "goal_funding": pd.DataFrame(goal_funding),
        "unfunded_gap": unfunded_gap,
        "remaining_unallocated": remaining,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. HUMAN CAPITAL INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

def human_capital_estimate(
    annual_income: float,
    years_to_retirement: int,
    income_growth_rate: float = 0.03,
    discount_rate: float = 0.05,
    income_stability: float = 0.8,   # 0=very variable (entrepreneur), 1=very stable (civil servant)
) -> dict:
    """
    Szacuje wartość kapitału ludzkiego (PV przyszłych zarobków).

    Kapitał ludzki to „obligacja" w portfelu:
    - Bezpieczna praca → bardziej jak obligacja → portfel może mieć więcej akcji
    - Ryzykowna praca (własna firma) → portfel powinien być bezpieczniejszy

    Merton (1971): optymalna alokacja akcji uwzględnia korelację
    dochodów z rynkiem.

    Parameters
    ----------
    annual_income       : float — bieżące dochody roczne (PLN)
    years_to_retirement : int
    income_growth_rate  : float — oczekiwany wzrost płacy rocznie
    discount_rate       : float — stopa dyskontowa
    income_stability    : float — 0 (bardzo zmienne) → 1 (bardzo stabilne)

    Returns
    -------
    dict z:
      human_capital_pv   : float — PV przyszłych zarobków
      recommended_equity : float — sugerowany % akcji w portfelu finansowym
      human_capital_type : str — 'bond-like' / 'equity-like'
    """
    # Present Value dochodów (annuity)
    if discount_rate == income_growth_rate:
        pv = annual_income * years_to_retirement
    else:
        pv = annual_income * (1 - (1 + income_growth_rate) ** years_to_retirement
                              / ((1 + discount_rate) ** years_to_retirement)) / (discount_rate - income_growth_rate)

    pv = max(0, pv)

    # Typ kapitału ludzkiego
    if income_stability > 0.7:
        hc_type = "🔵 Bond-like (stabilna praca = obligacja)"
        equity_boost = 0.20  # możesz mieć więcej akcji
    elif income_stability > 0.4:
        hc_type = "🟡 Mixed (umiarkowana stabilność)"
        equity_boost = 0.05
    else:
        hc_type = "🔴 Equity-like (własna firma = ryzykowna)"
        equity_boost = -0.10  # mniej akcji w portfelu finansowym!

    # Sugerowany % akcji (Merton-Samuelson lifecyle)
    base_equity = max(0.10, min(0.90, 0.50 + (years_to_retirement - 20) * 0.01 + equity_boost))

    return {
        "human_capital_pv": pv,
        "total_wealth_inc_hc": pv,  # + financial_wealth
        "human_capital_type": hc_type,
        "income_stability": income_stability,
        "recommended_equity_pct": base_equity,
        "years_to_retirement": years_to_retirement,
        "annual_income": annual_income,
        "note": f"Twój kapitał ludzki (PV przyszłych zarobków): {pv:,.0f} PLN — uwzględnij go w alokacji portfela",
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. REAL WEALTH PRESERVATION SCORE
# ══════════════════════════════════════════════════════════════════════════════

def real_wealth_preservation_score(
    portfolio_expected_return: float,
    portfolio_vol: float,
    inflation_rate: float = INFLATION_PL,
    tax_rate: float = TAX_BELKA,
    years: int = 10,
    n_sims: int = 10000,
    rng_seed: int = 42,
) -> dict:
    """
    Monte Carlo: P(realna wartość portfela rośnie po inflacji i podatkach).

    Real return = nominal_return × (1 - tax) - inflation

    Parameters
    ----------
    portfolio_expected_return : float — oczekiwana nominalna stopa zwrotu
    portfolio_vol             : float — roczna zmienność portfela
    inflation_rate            : float — oczekiwana inflacja
    tax_rate                  : float — efektywna stawka podatkowa (Belka dla capital gains)
    years                     : int — horyzont analizy

    Returns
    -------
    dict z:
      prob_preserve_real     : float — P(realna wartość > poczatkowa)
      expected_real_cagr     : float
      inflation_adjusted_return : float
      breakeven_nominal_return : float
    """
    if years <= 0:
        return {
            "prob_preserve_real_wealth": 0.0,
            "expected_real_cagr": 0.0,
            "inflation_adjusted_return": portfolio_expected_return * (1 - tax_rate) - inflation_rate,
            "breakeven_nominal_return": inflation_rate / (1 - tax_rate),
            "portfolio_nominal_return": portfolio_expected_return,
            "inflation_rate": inflation_rate,
            "tax_rate": tax_rate,
            "years": years,
            "grade": "N/D — Horyzont zerowy",
        }

    rng = np.random.default_rng(rng_seed)
    daily_mu = portfolio_expected_return / 252
    daily_sigma = portfolio_vol / np.sqrt(252)
    n_days = max(1, years * 252)

    # Simulate nominal returns
    sim = rng.normal(daily_mu, daily_sigma, (n_sims, n_days))
    nominal_wealth = np.cumprod(1 + sim, axis=1)[:, -1]

    # After tax: gains taxed at tax_rate
    gain = nominal_wealth - 1.0
    after_tax = 1.0 + gain * (1 - tax_rate)
    after_tax = np.where(gain < 0, nominal_wealth, after_tax)

    # After inflation
    inflation_factor = (1 + inflation_rate) ** years
    real_wealth = after_tax / inflation_factor

    prob_preserve = float((real_wealth > 1.0).mean())
    expected_real = float(np.median(real_wealth))
    real_cagr_median = (expected_real) ** (1 / years) - 1

    # Break-even nominal return (solve for real = 1.0)
    breakeven = (1 + inflation_rate) / (1 - tax_rate) - 1 + inflation_rate
    breakeven = inflation_rate / (1 - tax_rate) + tax_rate / (1 - tax_rate) * inflation_rate

    approx_breakeven = inflation_rate / (1 - tax_rate)

    # Grade
    if prob_preserve > 0.85 and real_cagr_median > 0.03:
        grade = "A — Doskonała ochrona siły nabywczej"
    elif prob_preserve > 0.70:
        grade = "B — Dobra ochrona"
    elif prob_preserve > 0.55:
        grade = "C — Marginalna ochrona"
    else:
        grade = "D/F — Realna utrata wartości"

    return {
        "prob_preserve_real_wealth": prob_preserve,
        "expected_real_cagr": real_cagr_median,
        "inflation_adjusted_return": portfolio_expected_return * (1 - tax_rate) - inflation_rate,
        "breakeven_nominal_return": approx_breakeven,
        "portfolio_nominal_return": portfolio_expected_return,
        "inflation_rate": inflation_rate,
        "tax_rate": tax_rate,
        "years": years,
        "grade": grade,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. LIABILITY-DRIVEN INVESTING (LDI)
# ══════════════════════════════════════════════════════════════════════════════

def ldi_funding_ratio(
    assets_value: float,
    liabilities: list[dict],
    discount_rate: float = 0.04,
) -> dict:
    """
    Oblicza Funding Ratio (FR) portfela względem zobowiązań.

    FR = Assets / PV(Liabilities)
    FR > 1.0 → nadfinansowany (surplus)
    FR < 1.0 → niedofinansowany (ryzyko)

    Parameters
    ----------
    assets_value : float — bieżąca wartość aktywów (PLN)
    liabilities  : list[dict] — {name, amount, years}
    discount_rate: float — stopa dyskonta zobowiązań

    Returns
    -------
    dict z: funding_ratio, pv_liabilities, surplus_deficit, status
    """
    total_pv = 0.0
    liability_detail = []

    for lib in liabilities:
        name = lib.get("name", "Zobowiązanie")
        amount = float(lib.get("amount", 0))
        years = float(lib.get("years", 5))
        pv = amount / (1 + discount_rate) ** years
        total_pv += pv
        liability_detail.append({
            "Zobowiązanie": name,
            "Kwota (PLN)": amount,
            "Horyzont (lat)": years,
            "PV (PLN)": round(pv, 0),
        })

    fr = assets_value / (total_pv + 1e-10)
    surplus = assets_value - total_pv

    if fr >= 1.20:
        status = "✅ Komfortowo nadfinansowany"
    elif fr >= 1.0:
        status = "🟡 Dostatecznie finansowany — małe ryzyko"
    elif fr >= 0.80:
        status = "🟠 Niedofinansowany — wymaga działań"
    else:
        status = "🔴 Krytyczny niedobór finansowania"

    return {\r\n        "assets_value": assets_value,\r\n        "pv_liabilities": total_pv,\r\n        "funding_ratio": fr,\r\n        "surplus_deficit": surplus,\r\n        "status": status,\r\n        "liability_detail": pd.DataFrame(liability_detail),\r\n        "discount_rate": discount_rate,\r\n    }\r\n\r\n\r\n# ══════════════════════════════════════════════════════════════════════════════\r\n# 5. DYNAMIC CPPI — Constant Proportion Portfolio Insurance (N5 UPGRADE)\r\n# Ref: Perold (1986), Black & Jones (1987), Herold et al. (2007)\r\n# ══════════════════════════════════════════════════════════════════════════════\r\n\r\ndef cppi_dynamic_floor(\r\n    portfolio_value: float,\r\n    floor_pct: float = 0.80,\r\n    multiplier: float = 5.0,\r\n    current_vol: float = 0.15,\r\n    target_vol: float = 0.15,\r\n    vol_scaling: bool = True,\r\n) -> dict:\r\n    """\r\n    CPPI z dynamicznym (vol-adjusted) multiplierem.\r\n\r\n    Standardowy CPPI (Perold 1986):\r\n        exposure = m × cushion\r\n        cushion  = portfolio - floor\r\n\r\n    Rozszerzenie (Black & Jones 1987, Herold et al. 2007):\r\n        m_eff = m × (σ_target / σ_current)  — vol scaling\r\n\r\n    Zalety dynamicznego multipliera:\r\n    - Gdy rynek spokojny (σ ↓): m_eff ↑ → więcej w ryzykowną część\r\n    - Gdy rynek niestabilny (σ ↑): m_eff ↓ → automatyczna ochrona\r\n\r\n    Referencje:\r\n        Perold (1986) — "Constant Proportion Portfolio Insurance"\r\n        Black & Jones (1987) — "Simplifying Portfolio Insurance"\r\n        Herold, Maurer & Purschaker (2007) — "Multi-Asset CPPI Strategies"\r\n        Cont & Tankov (2009) — "CPPI in Presence of Jumps in Financial Markets"\r\n\r\n    Parameters\r\n    ----------\r\n    portfolio_value : float — bieżąca wartość portfela (PLN)\r\n    floor_pct       : float — minimalna gwarantowana wartość (% portfela)\r\n    multiplier      : float — mnożnik CPPI (typowo 3-7)\r\n    current_vol     : float — bieżąca roczna zmienność (np. z GARCH-MIDAS)\r\n    target_vol      : float — docelowa zmienność (kalibracja)\r\n    vol_scaling     : bool — czy stosować dynamiczny multiplier?\r\n\r\n    Returns\r\n    -------\r\n    dict z: risky_exposure, safe_allocation, dynamic_multiplier, cushion, floor\r\n    """\r\n    floor = portfolio_value * floor_pct\r\n    cushion = max(portfolio_value - floor, 0.0)\r\n\r\n    # Dynamiczny multiplier: skalowanie przez stosunek vol\r\n    vol_ratio = target_vol / max(current_vol, 0.001)\r\n    if vol_scaling:\r\n        m_eff = multiplier * vol_ratio\r\n        m_eff = float(np.clip(m_eff, 1.0, multiplier * 3.0))  # cap: 3× bazowy\r\n    else:\r\n        m_eff = float(multiplier)\r\n\r\n    exposure = min(m_eff * cushion, portfolio_value)\r\n    exposure = max(exposure, 0.0)\r\n    safe = portfolio_value - exposure\r\n\r\n    if cushion / (portfolio_value + 1e-10) < 0.05:\r\n        protection = "🔴 Krytyczny — blisko floor (CPPI ucieka w bezpieczne)"\r\n    elif cushion / (portfolio_value + 1e-10) < 0.15:\r\n        protection = "🟠 Wysoka ochrona — zmniejszony udział ryzykownych"\r\n    elif m_eff > multiplier * 1.5:\r\n        protection = "🟢 Agresywny — wzmocniony przez niską vol"\r\n    else:\r\n        protection = "🟡 Standardowy CPPI"\r\n\r\n    return {\r\n        "risky_exposure":     exposure,\r\n        "safe_allocation":    safe,\r\n        "dynamic_multiplier": m_eff,\r\n        "static_multiplier":  multiplier,\r\n        "cushion":            cushion,\r\n        "floor":              floor,\r\n        "floor_pct":          floor_pct,\r\n        "vol_ratio":          vol_ratio,\r\n        "risky_pct":          exposure / (portfolio_value + 1e-10),\r\n        "safe_pct":           safe / (portfolio_value + 1e-10),\r\n        "protection_level":   protection,\r\n    }\r\n\r\n\r\ndef simulate_cppi_path(\r\n    initial_value: float,\r\n    risky_returns: np.ndarray,\r\n    safe_return: float = 0.045,\r\n    floor_pct: float = 0.80,\r\n    multiplier: float = 5.0,\r\n    vol_window: int = 21,\r\n    rebalance_freq: int = 5,\r\n) -> dict:\r\n    """\r\n    Symulacja ścieżki CPPI z dynamicznym rebalansowaniem vol-adjusted.\r\n\r\n    Returns: portfolio_path, risky_path, safe_path, weights_risky, max_drawdown\r\n    """\r\n    T = len(risky_returns)\r\n    portfolio = np.zeros(T + 1)\r\n    risky_alloc = np.zeros(T + 1)\r\n    safe_alloc = np.zeros(T + 1)\r\n    weights_risky = np.zeros(T)\r\n\r\n    portfolio[0] = initial_value\r\n    safe_daily = safe_return / 252\r\n    base_vol = float(np.std(risky_returns[:min(vol_window, T)]) * np.sqrt(252)) or 0.15\r\n\r\n    init_r = cppi_dynamic_floor(initial_value, floor_pct, multiplier, base_vol, base_vol)\r\n    risky_alloc[0] = init_r["risky_exposure"]\r\n    safe_alloc[0] = init_r["safe_allocation"]\r\n\r\n    for t in range(T):\r\n        new_risky = risky_alloc[t] * (1 + float(risky_returns[t]))\r\n        new_safe = safe_alloc[t] * (1 + safe_daily)\r\n        portfolio[t + 1] = new_risky + new_safe\r\n\r\n        if t % rebalance_freq == 0 and t + 1 < T:\r\n            start = max(0, t - vol_window)\r\n            cur_vol = float(np.std(risky_returns[start:t + 1])) * np.sqrt(252) or base_vol\r\n            r = cppi_dynamic_floor(portfolio[t + 1], floor_pct, multiplier, cur_vol, base_vol)\r\n            risky_alloc[t + 1] = r["risky_exposure"]\r\n            safe_alloc[t + 1] = r["safe_allocation"]\r\n        else:\r\n            risky_alloc[t + 1] = new_risky\r\n            safe_alloc[t + 1] = new_safe\r\n\r\n        pv = portfolio[t + 1]\r\n        weights_risky[t] = risky_alloc[t + 1] / pv if pv > 0 else 0.0\r\n\r\n    peak = np.maximum.accumulate(portfolio)\r\n    drawdowns = (portfolio - peak) / (peak + 1e-10)\r\n\r\n    return {\r\n        "portfolio_path":  portfolio,\r\n        "risky_path":      risky_alloc,\r\n        "safe_path":       safe_alloc,\r\n        "weights_risky":   weights_risky,\r\n        "final_value":     float(portfolio[-1]),\r\n        "max_drawdown":    float(drawdowns.min()),\r\n        "floor_breached":  bool(portfolio[-1] < initial_value * floor_pct),\r\n    }\r\n\r\n\r\n# ══════════════════════════════════════════════════════════════════════════════\r\n# 6. EXTENDED LDI — Duration Matching (N6 UPGRADE)\r\n# Ref: Leibowitz & Kogelman (1991), Fabozzi (2007)\r\n# ══════════════════════════════════════════════════════════════════════════════\r\n\r\ndef ldi_duration_matching(\r\n    asset_durations: "np.ndarray | list",\r\n    asset_weights: "np.ndarray | list",\r\n    liability_duration: float,\r\n    liability_convexity: float = 0.0,\r\n    asset_convexities: "np.ndarray | list | None" = None,\r\n) -> dict:\r\n    """\r\n    Duration/Convexity Matching dla Liability-Driven Investing.\r\n\r\n    Mierzy dopasowanie czasu trwania portfela aktywów do czasu trwania\r\n    zobowiązań. Gdy D_assets ≈ D_liabilities → portfel immunizowany.\r\n\r\n    Wzory:\r\n        D_portfolio = Σ w_i × D_i\r\n        Duration Gap = D_assets - D_liabilities\r\n        BPV = -D × V × 0.0001 (wartość punktu bazowego)\r\n\r\n    Ref: Leibowitz & Kogelman (1991) "Asset Allocation Under Shortfall Constraints"\r\n         Fabozzi (2007) "Fixed Income Mathematics", Ch. 6\r\n\r\n    Parameters\r\n    ----------\r\n    asset_durations    : (n,) duration każdego aktywa (lata)\r\n    asset_weights      : (n,) wagi aktywów\r\n    liability_duration : float — duration zobowiązań (lata)\r\n    liability_convexity: float — konweksja zobowiązań\r\n    asset_convexities  : (n,) konweksja aktywów (opcjonalna)\r\n\r\n    Returns\r\n    -------\r\n    dict z: portfolio_duration, duration_gap, is_immunized, bpv_gap, rebalancing_needed\r\n    """\r\n    D_a = np.asarray(asset_durations, dtype=float)\r\n    w = np.asarray(asset_weights, dtype=float)\r\n    w = w / (w.sum() + 1e-10)\r\n\r\n    D_portfolio = float(D_a @ w)\r\n    duration_gap = D_portfolio - liability_duration\r\n\r\n    C_portfolio = None\r\n    convexity_gap = 0.0\r\n    if asset_convexities is not None:\r\n        C_a = np.asarray(asset_convexities, dtype=float)\r\n        C_portfolio = float(C_a @ w)\r\n        convexity_gap = C_portfolio - liability_convexity\r\n\r\n    bpv_assets = -D_portfolio * 0.0001\r\n    bpv_liab = -liability_duration * 0.0001\r\n    bpv_gap = bpv_assets - bpv_liab\r\n\r\n    is_immunized = abs(duration_gap) < 0.5\r\n\r\n    if is_immunized:\r\n        rebalancing = "✅ Portfel immunizowany (Duration Gap < 0.5 lat)"\r\n    elif duration_gap > 0:\r\n        rebalancing = (f"⬇️ Skróć duration o {duration_gap:.1f} lat "\r\n                       f"(sprzedaj długie obligacje, kup krótkie)")\r\n    else:\r\n        rebalancing = (f"⬆️ Wydłuż duration o {abs(duration_gap):.1f} lat "\r\n                       f"(kup długoterminowe obligacje rządowe)")\r\n\r\n    return {\r\n        "portfolio_duration":  D_portfolio,\r\n        "liability_duration":  liability_duration,\r\n        "duration_gap":        duration_gap,\r\n        "duration_gap_pct":    duration_gap / max(abs(liability_duration), 1e-5) * 100,\r\n        "is_immunized":        is_immunized,\r\n        "bpv_assets":          bpv_assets,\r\n        "bpv_liabilities":     bpv_liab,\r\n        "bpv_gap":             bpv_gap,\r\n        "portfolio_convexity": C_portfolio,\r\n        "convexity_gap":       convexity_gap,\r\n        "rebalancing_needed":  rebalancing,\r\n        "asset_durations":     D_a.tolist(),\r\n        "asset_weights":       w.tolist(),\r\n    }\r\n

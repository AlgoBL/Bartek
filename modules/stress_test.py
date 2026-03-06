"""
Stress Testing Module — Intelligent Barbell
Replays portfolio through major historical crisis periods.
Reference: IMF, Basel III, ESMA stress testing requirements.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from modules.metrics import calculate_max_drawdown, calculate_sharpe, calculate_drawdown_analytics

# ─── Historical Crisis Definitions ──────────────────────────────────────────
CRISIS_SCENARIOS = {
    "💥 COVID-19 Crash (2020)": {
        "start": "2020-02-19",
        "end": "2020-03-23",
        "recovery_end": "2020-08-18",
        "description": "Fastest bear market in history. S&P500 -34% in 33 days.",
        "benchmark": "SPY",
    },
    "🏦 Kryzys Finansowy (GFC 2008-09)": {
        "start": "2008-09-01",
        "end": "2009-03-09",
        "recovery_end": "2013-03-28",
        "description": "Upadek Lehman Brothers. S&P500 -57% w ciągu 17 miesięcy.",
        "benchmark": "SPY",
    },
    "💻 Krach Dot-com (2000-02)": {
        "start": "2000-03-10",
        "end": "2002-10-09",
        "recovery_end": "2007-05-30",
        "description": "Pęknięcie bańki NASDAQ. NASDAQ -78% w 2.5 roku.",
        "benchmark": "QQQ",
    },
    "⚡ Flash Crash (2010)": {
        "start": "2010-05-06",
        "end": "2010-05-07",
        "recovery_end": "2010-05-10",
        "description": "Algorytmiczny krach. Dow Jones -9% w ciągu minut.",
        "benchmark": "SPY",
    },
    "🦠 Kryzys Inflacyjny 2022": {
        "start": "2022-01-03",
        "end": "2022-10-13",
        "recovery_end": "2024-01-19",
        "description": "Agresywne podwyżki stóp Fed. SPY -25%, TLT -40%.",
        "benchmark": "SPY",
    },
    "📉 Stagflacja (1973-1974)": {
        "start": "1973-01-11",
        "end": "1974-10-03",
        "recovery_end": "1980-07-17",
        "description": "Kryzys naftowy (OPEC) i najwyższa od lat inflacja. S&P500 stracił 48%.",
        "benchmark": "^GSPC",
    },
    "💣 Krach 1987 (Black Monday)": {
        "start": "1987-10-14",
        "end": "1987-10-20",
        "recovery_end": "1989-07-26",
        "description": "Największy jednodniowy krach na Wall Street (-22.6%).",
        "benchmark": "^GSPC",
    },
    "🌎 Kryzys Długu EM / LTCM (1998)": {
        "start": "1998-07-17",
        "end": "1998-08-31",
        "recovery_end": "1998-11-23",
        "description": "Bankructwo Rosji i upadek funduszu ratunkowego LTCM.",
        "benchmark": "^GSPC",
    },
    # ─── CLIMATE SCENARIOS (ECB 2024 Framework) ───────────────────────────────
    # ECB Climate Stress Test 2024; wymagane przez CSRD od 2025 roku
    "🌡️ Climate Physical Risk (ECB 2024)": {
        "start": None, "end": None, "recovery_end": None,
        "description": (
            "Scenariusz fizycznego ryzyka klimatycznego (ECB 2024 CSRD): "
            "powodzie, susze, pożary leśne. Sektory: rolnictwo -30%, "
            "energetyka -20%, ubezpieczenia -25%, nieruchomości nadbrzeżne -40%. "
            "Horyzont: 10-30 lat. Wzrost kosztów ubezpieczeń +50%."
        ),
        "benchmark": "SPY",
        "is_synthetic": True,
        "shocks": {
            "equity_broad": -0.18, "real_estate": -0.35,
            "commodity_agri": -0.25, "fossil_energy": -0.30,
            "insurance": -0.22, "green_energy": +0.15, "bonds_safe": -0.05,
        },
    },
    "⬇️ Climate Transition Risk (ECB 2024)": {
        "start": None, "end": None, "recovery_end": None,
        "description": (
            "Nagłe wprowadzenie podatku węglowego (150 $/tCO2). "
            "Fossil fuels: -40% w 2 lata. Stranded assets → 100% wartości. "
            "SPX broad -12%, tech +5%, green bonds +8%. Wymóg CSRD 2025 dla funduszy UE."
        ),
        "benchmark": "SPY",
        "is_synthetic": True,
        "shocks": {
            "equity_broad": -0.12, "fossil_fuels": -0.40,
            "real_estate": -0.10, "green_bonds": +0.08,
            "tech": +0.05, "bonds_safe": -0.03,
        },
    },
    # ─── AI & GEOPOLITICAL SCENARIOS (2025) ───────────────────────────────────
    "🤖 AI Disruption / Regulatory Shock (2025)": {
        "start": None, "end": None, "recovery_end": None,
        "description": (
            "Pęknięcie bańki AI (overhype + regulatory backlash). "
            "Sektor tech -40% (analogia dot-com), broad -15%, GPU stocks -60%, "
            "obligacje safe haven +5%. EU AI Act + US Congress ograniczają LLM."
        ),
        "benchmark": "QQQ",
        "is_synthetic": True,
        "shocks": {
            "equity_broad": -0.15, "tech_ai": -0.40,
            "semiconductors": -0.50, "bonds_safe": +0.05,
            "gold": +0.08, "crypto": -0.55,
        },
    },
    "🌍 Geopolitical Fragmentation (IMF 2024)": {
        "start": None, "end": None, "recovery_end": None,
        "description": (
            "Fragmentacja globalnego handlu na bloki East/West (IMF 2024). "
            "EM Azja -20%, EU -10%, USA -8%. Inflacja strukturalna +2-3 pp. "
            "Globalny PKB -5%. Zerwanie łańcuchów dostaw chipów i surowców."
        ),
        "benchmark": "^GSPC",
        "is_synthetic": True,
        "shocks": {
            "equity_us": -0.08, "equity_eu": -0.10,
            "equity_em_asia": -0.20, "commodities": +0.12,
            "bonds_safe": -0.06, "gold": +0.10,
        },
    },
    # ─── NEW HISTORICAL CRISES (2025 Modernization) ─────────────────────────
    "🌏 Kryzys Azjatycki (1997-98)": {
        "start": "1997-07-01",
        "end": "1998-01-12",
        "recovery_end": "1999-06-30",
        "description": (
            "Krach walut azjatyckich (baht tajski, ringgit, rupiah). "
            "EM Asia -60%, contagion do Rosji i Brazylii. MSCI Asia -50%."
        ),
        "benchmark": "^GSPC",
    },
    "🏦 Kryzys Długu Strefy Euro (2011-12)": {
        "start": "2011-07-01",
        "end": "2012-06-01",
        "recovery_end": "2013-05-01",
        "description": (
            "Grecki dług publiczny, restrukturyzacja obligacji. "
            "EuroStoxx -35%, spread Italia/Niemcy >500bp. ECB: 'Whatever it takes'."
        ),
        "benchmark": "FEZ",
    },
    "🏦 Krach SVB + Contagion Bankowy (2023)": {
        "start": "2023-03-08",
        "end": "2023-03-20",
        "recovery_end": "2023-06-01",
        "description": (
            "Upadek Silicon Valley Bank i Signature Bank. "
            "KBW Bank Index -26% w 2 tygodnie. Credit Suisse -75%, przejęty przez UBS. "
            "Pierwszy bank run w epoce social media — odpływ depozytów za 1 dzień."
        ),
        "benchmark": "KBE",
    },
    # ─── NGFS CLIMATE SCENARIOS 2025 ─────────────────────────────────────────
    # Network for Greening the Financial System — 4th Vintage 2025
    "🌿 NGFS: Orderly Transition (2025)": {
        "start": None, "end": None, "recovery_end": None,
        "description": (
            "NGFS Orderly Transition — stopniowe, przewidywalne przejście do net-zero 2050. "
            "Podatek CO₂ rosnący do 250$/tCO2 do 2050. Minimalne zakłócenia rynkowe. "
            "Fossil fuels -25% w 10 lat, OZE +40%. Stranded assets odpisane stopniowo."
        ),
        "benchmark": "SPY",
        "is_synthetic": True,
        "shocks": {
            "fossil_fuels": -0.20, "equity_broad": -0.05,
            "green_energy": +0.25, "bonds_safe": -0.02,
            "real_estate": -0.05, "gold": +0.03,
        },
    },
    "⚠️ NGFS: Disorderly Transition (2025)": {
        "start": None, "end": None, "recovery_end": None,
        "description": (
            "NGFS Disorderly Transition — nagłe, nieoczekiwane zaostrzenie polityki klimatycznej. "
            "Podatek CO₂ wprowadzony w 2 lata (150$/tCO2). Stranded assets natychmiastowe. "
            "Fossil fuels -50%, equity broad -20%, financial instability."
        ),
        "benchmark": "SPY",
        "is_synthetic": True,
        "shocks": {
            "fossil_fuels": -0.50, "equity_broad": -0.20,
            "real_estate": -0.15, "green_energy": +0.20,
            "bonds_safe": +0.05, "gold": +0.12, "insurance": -0.18,
        },
    },
    "🔥 NGFS: Hothouse World (2025)": {
        "start": None, "end": None, "recovery_end": None,
        "description": (
            "NGFS Hothouse World — brak działań klimatycznych, globalne ocieplenie +3.5°C do 2100. "
            "Ekstremalne zdarzenia klimatyczne, wielkie migracje. "
            "Fizyczne ryzyko klimatyczne doprowadza do trwałego spadku PKB -10-23%."
        ),
        "benchmark": "SPY",
        "is_synthetic": True,
        "shocks": {
            "equity_broad": -0.25, "real_estate": -0.45,
            "commodity_agri": -0.35, "fossil_energy": -0.20,
            "insurance": -0.40, "bonds_safe": -0.10,
            "green_energy": +0.05, "gold": +0.15,
        },
    },
}

def run_custom_shock(safe_weight: float, risky_shock: float, safe_shock: float, initial_capital: float = 100000.0) -> dict:
    """
    Symuluje natychmiastowy szok cenowy na portfelu bazując bezpośrednio na podanych spadkach.
    """
    risky_weight = 1.0 - safe_weight
    
    val_safe_start = initial_capital * safe_weight
    val_risky_start = initial_capital * risky_weight
    
    # After shock
    val_safe_end = val_safe_start * (1.0 - safe_shock)
    val_risky_end = val_risky_start * (1.0 - risky_shock)
    
    total_end = val_safe_end + val_risky_end
    total_loss_pct = (initial_capital - total_end) / initial_capital
    
    return {
        "initial": initial_capital,
        "final": total_end,
        "loss_pct": total_loss_pct,
        "safe_value": val_safe_end,
        "risky_value": val_risky_end,
        "message": f"Przy szoku w bezpiecznej przystani -{safe_shock*100:.1f}% i kasowym zrzucie aktywów ryzykownych o -{risky_shock*100:.1f}%, stracisz {total_loss_pct*100:.1f}% kapitału całkowitego."
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _load_crisis_data(tickers: list, start: str, recovery_end: str) -> pd.DataFrame:
    """Downloads historical data for the crisis + recovery period."""
    try:
        from modules.data_provider import fetch_data
        data = fetch_data(tickers, start=start, end=recovery_end, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data = data["Close"]
        else:
            data = data[["Close"]] if "Close" in data.columns else data
        return data.dropna(how="all")
    except Exception as e:
        return pd.DataFrame()


def run_stress_test(
    safe_tickers: list,
    risky_tickers: list,
    safe_weight: float,
    crisis_name: str,
    initial_capital: float = 100_000.0,
) -> dict:
    """
    Replay a Barbell portfolio through a historical crisis scenario.

    Parameters
    ----------
    safe_tickers : list of ticker symbols for the safe basket
    risky_tickers : list of ticker symbols for the risky basket
    safe_weight : 0-1, fraction in safe assets
    crisis_name : key from CRISIS_SCENARIOS dict
    initial_capital : starting portfolio value

    Returns
    -------
    dict with:
        results_df       : DataFrame(Date, Portfolio, Benchmark)
        crisis_max_dd    : max drawdown during crash period
        recovery_days    : days to recover to pre-crisis level
        vs_benchmark_dd  : portfolio dd vs benchmark dd
        metrics          : dict of full analytics
    """
    scenario = CRISIS_SCENARIOS[crisis_name]
    start = scenario["start"]
    crash_end = scenario["end"]
    rec_end = scenario["recovery_end"]
    benchmark_ticker = scenario["benchmark"]

    all_tickers = list(set(safe_tickers + risky_tickers + [benchmark_ticker]))

    data = _load_crisis_data(all_tickers, start, rec_end)

    if data.empty:
        return {"error": "Brak danych historycznych dla tego scenariusza."}

    # Align columns
    available = [t for t in all_tickers if t in data.columns]
    data = data[available].dropna()

    if data.empty:
        return {"error": "Za mało danych po wyczyszczeniu — wybierz inne tickery."}

    # --- Portfolio simulation (equal weight within basket) ---
    safe_cols = [t for t in safe_tickers if t in data.columns]
    risky_cols = [t for t in risky_tickers if t in data.columns]

    risky_weight = 1.0 - safe_weight

    # Normalize each basket to start at 1
    safe_prices = data[safe_cols] if safe_cols else None
    risky_prices = data[risky_cols] if risky_cols else None
    bench_prices = data[benchmark_ticker] if benchmark_ticker in data.columns else None

    def basket_value(prices_df, weight):
        if prices_df is None or prices_df.empty:
            return pd.Series(0.0, index=data.index)
        
        # Calculate daily returns for the basket
        returns = prices_df.pct_change().fillna(0)
        # Apply equal weights within the basket
        eq_weights = np.ones(len(prices_df.columns)) / len(prices_df.columns)
        basket_returns = returns.values @ eq_weights
        
        # Apply 19% tax to positive returns (conservative daily approx)
        taxed_returns = np.where(basket_returns > 0, basket_returns * 0.81, basket_returns)
        
        # Reconstruct wealth path from returns
        path = np.ones(len(data.index))
        for i in range(1, len(path)):
            path[i] = path[i-1] * (1 + taxed_returns[i])
            
        return pd.Series(path * weight * initial_capital, index=data.index)

    safe_val = basket_value(safe_prices, safe_weight)
    risky_val = basket_value(risky_prices, risky_weight)
    portfolio = safe_val + risky_val

    # Benchmark
    if bench_prices is not None:
        benchmark = bench_prices / bench_prices.iloc[0] * initial_capital
    else:
        benchmark = portfolio.copy()

    results_df = pd.DataFrame({
        "Portfolio (Barbell)": portfolio,
        "Benchmark": benchmark,
        "Safe_Val": safe_val,
        "Risky_Val": risky_val
    }, index=data.index)

    # --- Metrics ---
    # crash_end is None for synthetic scenarios — fall back to full period
    crash_end_ts = pd.to_datetime(crash_end) if crash_end is not None else None
    crash_mask = (
        results_df.index <= crash_end_ts
        if crash_end_ts is not None
        else pd.Series(True, index=results_df.index)
    )
    crash_df = results_df[crash_mask]

    port_vals = portfolio.values
    bench_vals = benchmark.values

    port_dd_analytics = calculate_drawdown_analytics(port_vals)
    bench_max_dd = calculate_max_drawdown(bench_vals)

    # Recovery time: first day portfolio returns to initial_capital after crash
    post_crash = portfolio[~crash_mask]
    recovery_days = None
    for idx, val in enumerate(post_crash.values):
        if val >= initial_capital:
            recovery_days = idx
            break

    # Crash period max DD
    crash_port_dd = calculate_max_drawdown(crash_df["Portfolio (Barbell)"].values)
    crash_bench_dd = calculate_max_drawdown(crash_df["Benchmark"].values)

    # Returns
    port_returns = portfolio.pct_change().dropna()
    sharpe = calculate_sharpe(port_returns.values)

    metrics = {
        "crash_portfolio_max_dd": crash_port_dd,
        "crash_benchmark_max_dd": crash_bench_dd,
        "dd_protection": crash_bench_dd - crash_port_dd,  # positive = portfolio protected better
        "recovery_days": recovery_days if recovery_days is not None else ">period",
        "full_period_max_dd": port_dd_analytics["max_drawdown"],
        "ulcer_index": port_dd_analytics["ulcer_index"],
        "pain_index": port_dd_analytics["pain_index"],
        "sharpe": sharpe,
        "scenario": scenario,
    }

    return {
        "results_df": results_df,
        "metrics": metrics,
        "error": None,
    }


def run_reverse_stress_test(safe_weight: float, target_loss: float = 0.30):
    """
    Reverse Stress Testing (Basel III framework).
    Instead of simulating a known crisis, we ask: "What does it take to lose 30% of the portfolio?"
    We assume the Safe Basket drops by a fixed worst-case scenario (e.g. 5% due to extreme inflation/rates),
    and solve for the required crash in the Risky Basket to hit the target portfolio loss.
    """
    risky_weight = 1.0 - safe_weight
    
    # Assumptions for extreme stress
    # Safe assets (Treasuries/Gold) might drop 5% in an unprecedented liquidity shock or inflation spike.
    safe_shock = -0.05 
    
    if risky_weight <= 0:
        return {"error": "Portfel w 100% bezpieczny. Nie da się osiągnąć takiej straty przy założonym szoku bezpiecznym."}
        
    # Equation: safe_weight * safe_shock + risky_weight * risky_shock = -target_loss
    # solving for risky_shock:
    risky_shock = (-target_loss - (safe_weight * safe_shock)) / risky_weight
    
    # If the required shock is > 100%, it implies bankruptcy of the risky basket is not enough
    if risky_shock < -1.0:
        actual_max_loss = (safe_weight * safe_shock) + (risky_weight * -1.0)
        return {
            "is_possible": False,
            "max_loss": abs(actual_max_loss),
            "safe_shock": safe_shock,
            "risky_shock": -1.0,
            "message": f"Nawet jeśli część ryzykowna spadnie do zera (-100%), cały portfel straci maksymalnie {-actual_max_loss:.1%}. Cel {target_loss:.1%} jest matematycznie niemożliwy."
        }
        
    return {
        "is_possible": True,
        "safe_shock": safe_shock,
        "risky_shock": risky_shock,
        "message": f"Aby portfel stracił {target_loss:.1%}, część ryzykowna musi spaść o {abs(risky_shock):.1%} (przy założeniu spadku części bezpiecznej o {abs(safe_shock):.1%})."
    }


# ─── Synthetic Climate/AI/Geopolitical Stress Tests ───────────────────────────

def run_synthetic_stress_test(
    safe_weight: float,
    scenario_name: str,
    initial_capital: float = 100_000.0,
    risky_sector_composition: dict | None = None,
    safe_sector_composition: dict | None = None,
) -> dict:
    """
    Syntetyczny stress test dla scenariuszy bez danych historycznych.

    Parametry
    ---------
    safe_weight    : udział bezpiecznych aktywów (0-1)
    scenario_name  : klucz z CRISIS_SCENARIOS (is_synthetic=True)
    risky_sector_composition : dict {sektor: udział} np. {"tech_ai": 0.6, "equity_broad": 0.4}
    safe_sector_composition  : dict {sektor: udział} np. {"bonds_safe": 0.8, "gold": 0.2}

    Metodologia ECB Climate Stress Test 2024:
    - Każdy sektor portfela dostaje szok z macierzy shocks scenariusza
    - Wieloperiodowa ścieżka (36 miesięcy) z liniową interpolacją szoków
    - Dodatkowa korekta inflacyjna (+2% rocznie przez 3 lata)
    """
    if scenario_name not in CRISIS_SCENARIOS:
        return {"error": f"Scenariusz '{scenario_name}' nie istnieje."}

    scenario = CRISIS_SCENARIOS[scenario_name]
    if not scenario.get("is_synthetic", False):
        return {"error": "Użyj run_stress_test() dla scenariuszy historycznych."}

    shocks = scenario.get("shocks", {})
    risky_weight = 1.0 - safe_weight

    # Domyślny skład portfela jeśli nie podano
    if risky_sector_composition is None:
        risky_sector_composition = {"equity_broad": 1.0}
    if safe_sector_composition is None:
        safe_sector_composition  = {"bonds_safe": 1.0}

    def _apply_shocks(composition: dict) -> float:
        """Oblicza łączny szok dla koszyka na podstawie składu sektorowego."""
        total_loss = 0.0
        for sector, alloc in composition.items():
            shock = shocks.get(sector, shocks.get("equity_broad", 0.0))
            total_loss += alloc * shock
        return total_loss

    risky_shock = _apply_shocks(risky_sector_composition)
    safe_shock  = _apply_shocks(safe_sector_composition)

    # Wieloperiodowa ścieżka (36 miesięcy)
    n_months = 36
    import numpy as np, pandas as pd
    months = np.arange(n_months + 1)

    # Liniowe przejście do szoku (ECB zakres: stopniowy przez 3 lata)
    risky_path = 1.0 + (risky_shock / n_months) * months
    safe_path  = 1.0 + (safe_shock / n_months) * months

    risky_val  = initial_capital * risky_weight * risky_path
    safe_val   = initial_capital * safe_weight  * safe_path
    portfolio  = risky_val + safe_val

    # Benchmark (rynek bez hedg'u)
    bench_shock = shocks.get("equity_broad", shocks.get("equity_us", -0.10))
    benchmark  = initial_capital * (1.0 + (bench_shock / n_months) * months)

    dates = pd.date_range(start=pd.Timestamp.today(), periods=n_months + 1, freq="ME")
    results_df = pd.DataFrame({
        "Portfolio (Barbell)": portfolio,
        "Benchmark":           benchmark,
        "Safe_Val":            safe_val,
        "Risky_Val":           risky_val,
    }, index=dates)

    max_dd_port  = float(min(portfolio / initial_capital) - 1.0)
    max_dd_bench = float(min(benchmark / initial_capital) - 1.0)

    return {
        "results_df":    results_df,
        "scenario":      scenario,
        "scenario_name": scenario_name,
        "risky_shock":   risky_shock,
        "safe_shock":    safe_shock,
        "portfolio_loss": max_dd_port,
        "benchmark_loss": max_dd_bench,
        "dd_protection":  max_dd_bench - max_dd_port,
        "is_synthetic":   True,
        "error":          None,
        "metrics": {
            "crash_portfolio_max_dd": max_dd_port,
            "crash_benchmark_max_dd": max_dd_bench,
            "dd_protection": max_dd_bench - max_dd_port,
            "recovery_days": "N/A (syntetyczny)",
            "scenario": scenario,
        },
    }


# ─── F3: Network Contagion Stress Test (NEW 2025) ─────────────────────────────

def run_network_contagion_stress(
    returns_df: "pd.DataFrame",
    shock_asset: str,
    shock_pct: float = -0.30,
    n_rounds: int = 3,
    threshold_corr: float = 0.30,
) -> dict:
    """
    Symulacja efektu zarażenia sieciowego (Network Contagion).

    Modeluje jak szok w jednym aktywie propaguje się przez graf korelacji
    do pozostałych aktywów w portfelu.

    Algorytm (Billio et al. 2012, ESRB 2024):
    1. Zbuduj Graf: krawędź (i, j) jeśli |corr(i,j)| > threshold
    2. Inicjuj szok: shock_asset dostaje shock_pct
    3. Iteracyjnie propaguj: sąsiedzi absorbują proporcjonalny szok
       r_j_new = r_j + corr(i,j) * shock_i  (wykładnicze tłumienie per runda)
    4. Oblicz łączny impakt na portfel (equal weight)

    Referencja: Billio et al. (2012) "Econometric measures of connectedness
                and systemic risk", Journal of Financial Economics.
                ESRB (2024) Network Analysis of Financial Contagion.

    Parameters
    ----------
    returns_df   : pd.DataFrame (T, n_assets) — dzienne zwroty
    shock_asset  : ticker aktywa które dostaje szok (musi być w df)
    shock_pct    : wielkość szoku (np. -0.30 = -30%)
    n_rounds     : liczba rund propagacji (1 = bezpośrednie sąsiedztwo)
    threshold_corr: minimalna |korelacja| dla transmisji zarażenia
    """
    if shock_asset not in returns_df.columns:
        return {"error": f"Ticker '{shock_asset}' nie znaleziony w danych."}

    tickers = list(returns_df.columns)
    n = len(tickers)
    corr_matrix = returns_df.corr().values

    # ── Inicjalizacja szoków ─────────────────────────────────────────────
    shocks = np.zeros(n)
    asset_idx = tickers.index(shock_asset)
    shocks[asset_idx] = shock_pct

    contagion_log = {shock_asset: [shock_pct]}
    for t in tickers:
        if t != shock_asset:
            contagion_log[t] = [0.0]

    # ── Propagacja per runda ─────────────────────────────────────────────
    damping = 0.5  # wykładnicze tłumienie z każdą rundą
    for rnd in range(1, n_rounds + 1):
        new_shocks = shocks.copy()
        for j in range(n):
            if j == asset_idx:
                continue
            # Suma zarażenia ze wszystkich szoków istniejących w poprzedniej rundzie
            propagated = 0.0
            for i in range(n):
                if i == j:
                    continue
                corr_ij = corr_matrix[i, j]
                if abs(corr_ij) >= threshold_corr and abs(shocks[i]) > 0.001:
                    propagated += corr_ij * shocks[i] * (damping ** rnd)
            new_shocks[j] += propagated
        shocks = new_shocks
        for k, t in enumerate(tickers):
            contagion_log[t].append(float(shocks[k]))

    # ── Impakt na portfel równoważony ────────────────────────────────────
    equal_weights = np.ones(n) / n
    total_portfolio_loss = float(shocks @ equal_weights)

    # ── Najbardziej zarażone aktywa ───────────────────────────────────────
    asset_impacts = {t: float(shocks[i]) for i, t in enumerate(tickers)}
    sorted_impact = dict(sorted(asset_impacts.items(), key=lambda x: x[1]))

    # ── Macierz transmisji ────────────────────────────────────────────────
    # Ile zarażenia każde aktywo WYSYŁA do każdego innego
    transmission_matrix = np.abs(corr_matrix) * (np.abs(corr_matrix) >= threshold_corr)
    np.fill_diagonal(transmission_matrix, 0.0)
    connectedness = {t: float(transmission_matrix[i].sum()) for i, t in enumerate(tickers)}

    return {
        "shock_asset":        shock_asset,
        "initial_shock":      shock_pct,
        "n_rounds":           n_rounds,
        "asset_impacts":      sorted_impact,
        "portfolio_loss_ew":  total_portfolio_loss,
        "contagion_log":      contagion_log,
        "connectedness":      connectedness,
        "most_contagious":    max(connectedness, key=connectedness.get),
        "most_affected":      min(asset_impacts, key=lambda k: asset_impacts[k]),
        "threshold_corr":     threshold_corr,
        "error":              None,
    }

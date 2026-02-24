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
    crash_mask = results_df.index <= pd.to_datetime(crash_end)
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

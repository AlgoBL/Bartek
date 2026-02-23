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
}


@st.cache_data(ttl=3600, show_spinner=False)
def _load_crisis_data(tickers: list, start: str, recovery_end: str) -> pd.DataFrame:
    """Downloads historical data for the crisis + recovery period."""
    try:
        data = yf.download(tickers, start=start, end=recovery_end, progress=False, auto_adjust=True)
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

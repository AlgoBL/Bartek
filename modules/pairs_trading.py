"""
pairs_trading.py — Pełny moduł Pairs Trading / Statistical Arbitrage
=====================================================================
Modele:
  - Engle-Granger Two-Step Cointegration
  - Johansen Cointegration (VECM)
  - Kalman Filter Dynamic Hedge Ratio
  - Half-Life of Mean Reversion (OU process)
  - Entry/Exit Signal Generation (Z-score)
  - Backtest Engine (PnL, Sharpe, Drawdown)
  - Correlation Matrix + MST Clustering

Autorzy: Engle & Granger (1987), Johansen (1991), Elliott & Valavanis (2005)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy.stats import pearsonr
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  1. ENGLE-GRANGER COINTEGRATION (rozbudowany)
# ─────────────────────────────────────────────────────────────────────────────

def perform_cointegration_test(series1: pd.Series, series2: pd.Series) -> dict:
    """
    Engle-Granger Two-Step Cointegration Test.
    Krok 1: OLS regresja s1 ~ s2
    Krok 2: ADF test na resztach (spread)
    Zwraca pełny słownik diagnostyczny.
    """
    if len(series1) != len(series2) or len(series1) < 30:
        return {"is_cointegrated": False, "p_value": 1.0, "hedge_ratio": 0,
                "spread_z": pd.Series(), "current_z": 0.0, "spread_raw": pd.Series()}

    df = pd.concat([series1, series2], axis=1).dropna()
    if len(df) < 30:
        return {"is_cointegrated": False, "p_value": 1.0, "hedge_ratio": 0,
                "spread_z": pd.Series(), "current_z": 0.0, "spread_raw": pd.Series()}

    s1 = df.iloc[:, 0]
    s2 = df.iloc[:, 1]

    X = sm.add_constant(s2)
    model = sm.OLS(s1, X).fit()
    hedge_ratio = float(model.params.iloc[1])
    alpha = float(model.params.iloc[0])

    spread = s1 - hedge_ratio * s2 - alpha
    adf_result = adfuller(spread, maxlag=1, autolag=None)
    p_value = float(adf_result[1])
    adf_stat = float(adf_result[0])

    spread_z = (spread - spread.mean()) / (spread.std(ddof=1) + 1e-10)
    half_life = calculate_half_life(spread)

    return {
        "is_cointegrated": bool(p_value < 0.05),
        "p_value": p_value,
        "adf_stat": adf_stat,
        "hedge_ratio": hedge_ratio,
        "alpha": alpha,
        "spread_z": spread_z,
        "current_z": float(spread_z.iloc[-1]) if len(spread_z) > 0 else 0.0,
        "spread_raw": spread,
        "half_life_days": half_life,
        "r_squared": float(model.rsquared),
        "n_obs": len(df),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  2. JOHANSEN COINTEGRATION (VECM)
# ─────────────────────────────────────────────────────────────────────────────

def johansen_cointegration_test(series1: pd.Series, series2: pd.Series,
                                 det_order: int = 0, k_ar_diff: int = 1) -> dict:
    """
    Johansen Cointegration Test (Trace + Max Eigenvalue).
    Zwraca liczbę wektorów kointegrujących i beta (hedge ratio).
    
    det_order: -1=no trend, 0=constant, 1=trend
    """
    df = pd.concat([series1, series2], axis=1).dropna()
    if len(df) < 50:
        return {"n_cointegrating_vectors": 0, "trace_stat": [], "crit_vals": [], "beta": None}

    try:
        result = coint_johansen(df.values, det_order=det_order, k_ar_diff=k_ar_diff)
        # Trace statistic vs critical values (95%)
        trace_stat = result.lr1.tolist()
        crit_95 = result.cvt[:, 1].tolist()  # 95% critical values

        n_coint = 0
        for ts, cv in zip(trace_stat, crit_95):
            if ts > cv:
                n_coint += 1
            else:
                break

        # First cointegrating vector (normalized hedge ratio)
        beta = None
        if n_coint > 0 and result.evec is not None:
            evec = result.evec[:, 0]
            beta = float(-evec[1] / evec[0]) if abs(evec[0]) > 1e-10 else None

        return {
            "n_cointegrating_vectors": n_coint,
            "trace_stat": trace_stat,
            "crit_vals_95": crit_95,
            "beta": beta,
            "eigenvalues": result.eig.tolist() if result.eig is not None else [],
        }
    except Exception as e:
        return {"n_cointegrating_vectors": 0, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
#  3. KALMAN FILTER — DYNAMICZNY HEDGE RATIO
# ─────────────────────────────────────────────────────────────────────────────

def kalman_filter_hedge_ratio(series1: pd.Series, series2: pd.Series,
                               delta: float = 1e-4) -> dict:
    """
    Kalman Filter dla dynamicznego (time-varying) hedge ratio.
    State: [alpha, beta] — dynamicznie aktualizowane.
    
    delta: szybkość adaptacji (mniejszy = wolniej, stabilniej)
    Ref: Elliott & Valavanis (2005), Pole et al. (1994)
    """
    df = pd.concat([series1, series2], axis=1).dropna()
    if len(df) < 30:
        return {"hedge_ratios": pd.Series(), "spread_z": pd.Series()}

    y = df.iloc[:, 0].values
    x = df.iloc[:, 1].values
    n = len(y)

    # State-space parameters
    Wn = delta / (1 - delta) * np.eye(2)  # state noise
    Vt = 1.0  # observation noise (estimated iteratively)

    # Initial state
    beta = np.zeros(2)   # [alpha, beta]
    R = np.zeros((2, 2))  # state covariance
    P = np.zeros((2, 2))

    hedge_ratios = np.zeros(n)
    alphas = np.zeros(n)
    spreads = np.zeros(n)

    for i in range(n):
        F = np.array([[1.0], [x[i]]])  # observation matrix

        # Prediction
        R_pred = P + Wn

        # Update (Kalman gain)
        S = F.T @ R_pred @ F + Vt
        K = R_pred @ F / S[0, 0]

        # Measurement residual
        y_hat = float(F.T @ beta)
        resid = y[i] - y_hat

        # State update
        beta = beta + K.ravel() * resid
        P = (np.eye(2) - K @ F.T) @ R_pred

        # Store
        alphas[i] = beta[0]
        hedge_ratios[i] = beta[1]
        spreads[i] = y[i] - beta[1] * x[i] - beta[0]

    spread_series = pd.Series(spreads, index=df.index)
    spread_z = (spread_series - spread_series.rolling(60).mean()) / (
        spread_series.rolling(60).std(ddof=1) + 1e-10)

    return {
        "hedge_ratios": pd.Series(hedge_ratios, index=df.index),
        "alphas": pd.Series(alphas, index=df.index),
        "spread_raw": spread_series,
        "spread_z": spread_z,
        "current_hedge_ratio": float(hedge_ratios[-1]),
        "current_z": float(spread_z.iloc[-1]) if len(spread_z.dropna()) > 0 else 0.0,
        "current_spread": float(spreads[-1]),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  4. HALF-LIFE (ORNSTEIN-UHLENBECK)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_half_life(spread: pd.Series) -> float:
    """
    Oblicza Half-Life powrotu spreadu do średniej.
    Model OU: dX = κ(μ - X)dt + σdW
    Regresja: ΔX_t = κ(μ - X_{t-1}) → half-life = ln(2)/κ
    """
    if len(spread) < 20:
        return np.nan

    spread = spread.dropna()
    delta_spread = spread.diff().dropna()
    spread_lag = spread.shift(1).dropna()
    delta_spread = delta_spread.iloc[-len(spread_lag):]

    X = sm.add_constant(spread_lag.values)
    try:
        model = sm.OLS(delta_spread.values, X).fit()
        kappa = -model.params[1]
        if kappa <= 0:
            return np.inf
        half_life = np.log(2) / kappa
        return max(0.0, float(half_life))
    except Exception:
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
#  5. SIGNAL GENERATOR (Entry / Exit Rules)
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(spread_z: pd.Series,
                     entry_threshold: float = 2.0,
                     exit_threshold: float = 0.5,
                     stop_loss: float = 3.5) -> pd.DataFrame:
    """
    Generuje sygnały Long/Short/Flat na podstawie Z-score spreadu.
    
    Logika:
      - Z < -entry  → Long spread (kup s1, sprzedaj s2)
      - Z > +entry  → Short spread (sprzedaj s1, kup s2)
      - |Z| < exit  → Flat (zamknij pozycję)
      - |Z| > stop  → Stop-Loss (zamknij natychmiast)
    
    Returns DataFrame z kolumnami: z_score, position, signal
    """
    z = spread_z.dropna()
    positions = pd.Series(0, index=z.index, dtype=float)
    current_pos = 0

    for i in range(len(z)):
        val = z.iloc[i]

        # Stop-loss
        if abs(val) > stop_loss:
            current_pos = 0
        # Exit
        elif current_pos != 0 and abs(val) < exit_threshold:
            current_pos = 0
        # Entry
        elif current_pos == 0:
            if val < -entry_threshold:
                current_pos = 1   # Long spread
            elif val > entry_threshold:
                current_pos = -1  # Short spread

        positions.iloc[i] = current_pos

    signals = positions.diff().fillna(0)

    return pd.DataFrame({
        "z_score": z,
        "position": positions,
        "signal": signals,  # +1=open long, -1=open short, opposite=close
        "entry_threshold": entry_threshold,
        "exit_threshold": exit_threshold,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  6. BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def backtest_pair(series1: pd.Series, series2: pd.Series,
                  hedge_ratio: float,
                  signals_df: pd.DataFrame,
                  transaction_cost_bps: float = 5.0) -> dict:
    """
    Prosty backtest pary (long spread / short spread).
    Zwraca PnL, Sharpe, Max Drawdown, Win Rate.
    
    transaction_cost_bps: koszt transakcji w bp (1bp = 0.01%)
    """
    df = pd.concat([series1, series2], axis=1).dropna()
    if len(df) < 30 or signals_df.empty:
        return {}

    s1 = df.iloc[:, 0]
    s2 = df.iloc[:, 1]

    # Daily returns of spread position
    spread_ret = s1.pct_change() - hedge_ratio * s2.pct_change()
    spread_ret = spread_ret.reindex(signals_df.index).fillna(0)

    positions = signals_df["position"].shift(1).fillna(0)  # execute next day
    tc = transaction_cost_bps / 10000

    # Transaction costs on position changes
    position_changes = positions.diff().abs().fillna(0)
    pnl = positions * spread_ret - position_changes * tc

    cumret = (1 + pnl).cumprod()
    ann_ret = float((cumret.iloc[-1] ** (252 / max(len(cumret), 1)) - 1))
    ann_vol = float(pnl.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0

    # Max drawdown
    rolling_max = cumret.cummax()
    drawdown = (cumret - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    # Win rate
    trade_pnl = pnl[positions != 0]
    win_rate = float((trade_pnl > 0).mean()) if len(trade_pnl) > 0 else 0.0

    # Number of trades
    n_trades = int(position_changes[position_changes > 0].count())

    return {
        "cumulative_return": cumret,
        "pnl_series": pnl,
        "ann_return_pct": ann_ret * 100,
        "ann_vol_pct": ann_vol * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "win_rate_pct": win_rate * 100,
        "n_trades": n_trades,
        "total_return_pct": float((cumret.iloc[-1] - 1) * 100),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  7. MULTI-PAIR SCANNER
# ─────────────────────────────────────────────────────────────────────────────

def scan_pairs(returns_df: pd.DataFrame, min_correlation: float = 0.7) -> pd.DataFrame:
    """
    Skanuje wszystkie pary aktywów w poszukiwaniu kointegracji.
    Filtruje najpierw po korelacji (min_correlation), potem testuje kointegrację.
    
    Returns: DataFrame z rankedlistą par posortowaną wg p-value.
    """
    tickers = list(returns_df.columns)
    results = []

    prices = (1 + returns_df).cumprod()  # pseudo-prices from returns

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            s1 = prices[t1].dropna()
            s2 = prices[t2].dropna()
            common = s1.index.intersection(s2.index)
            if len(common) < 60:
                continue

            s1c, s2c = s1[common], s2[common]

            # Quick correlation filter
            corr, _ = pearsonr(s1c, s2c)
            if abs(corr) < min_correlation:
                continue

            # Cointegration test
            try:
                coint_t, p_value, _ = coint(s1c, s2c)
            except Exception:
                continue

            # Half-life
            X = sm.add_constant(s2c.values)
            try:
                model = sm.OLS(s1c.values, X).fit()
                hr = float(model.params[1])
                spread = s1c - hr * s2c - float(model.params[0])
                hl = calculate_half_life(spread)
                z_now = float((spread - spread.mean()) / (spread.std(ddof=1) + 1e-10))
                z_now = z_now if not np.isnan(z_now) else 0.0
            except Exception:
                hr, hl, z_now = 0.0, np.nan, 0.0

            results.append({
                "Pair": f"{t1} / {t2}",
                "Ticker_1": t1,
                "Ticker_2": t2,
                "Correlation": round(corr, 3),
                "p_value": round(p_value, 4),
                "Is_Cointegrated": p_value < 0.05,
                "Hedge_Ratio": round(hr, 4),
                "Half_Life_Days": round(hl, 1) if not np.isnan(hl) else None,
                "Current_Z": round(z_now, 2),
                "Signal": ("LONG" if z_now < -2 else "SHORT" if z_now > 2 else "FLAT"),
            })

    if not results:
        return pd.DataFrame()

    df_out = pd.DataFrame(results)
    df_out = df_out[df_out["Is_Cointegrated"]].sort_values("p_value")
    return df_out.reset_index(drop=True)

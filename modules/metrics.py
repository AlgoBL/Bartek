import numpy as np
from scipy.stats import norm
import pandas as pd

def calculate_sharpe(returns, rf=0.0324, periods=252):
    """
    Calculates Sharpe Ratio: (Mean Return - Risk Free) / Volatility
    """
    if len(returns) < 2: return 0.0
    
    # Convert annual RF to daily if needed, here we assume rf is annual 4%
    # Excess daily return = daily_return - (rf / 252)
    rf_daily = (1 + rf)**(1/periods) - 1
    excess_returns = returns - rf_daily
    
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0: return 0.0
    
    return (mean_excess / std_excess) * np.sqrt(periods)

def calculate_sortino(returns, rf=0.0324, periods=252, target_return=0.0):
    """
    Calculates Sortino Ratio: (Mean Return - Risk Free) / Downside Deviation
    """
    if len(returns) < 2: return 0.0
    
    rf_daily = (1 + rf)**(1/periods) - 1
    excess_returns = returns - rf_daily
    mean_excess = np.mean(excess_returns)
    
    # Downside deviation relative to target (usually 0 or rf)
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0:
        return np.inf # No downside risk
        
    # We calculate geometric downside deviation or standard deviation of negative returns
    # Sortino denominator is sqrt(mean(min(0, R - target)^2))
    downside_diff = np.minimum(0, returns - target_return)**2
    downside_dev = np.sqrt(np.mean(downside_diff))
    
    if downside_dev == 0: return np.inf
    
    return (mean_excess / downside_dev) * np.sqrt(periods)

def calculate_calmar(cagr, max_drawdown):
    """
    Calculates Calmar Ratio: CAGR / abs(Max Drawdown)
    """
    if max_drawdown == 0: return np.inf
    return cagr / abs(max_drawdown)

def calculate_max_drawdown(prices):
    """
    Calculates Max Drawdown from a price series.
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    peaks = np.maximum.accumulate(prices)
    drawdowns = (prices - peaks) / peaks
    return np.min(drawdowns)

def calculate_var_cvar(returns, confidence=0.95):
    """
    Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    Returns (VaR, CVaR) as positive percentages (losses).
    """
    if len(returns) == 0: return 0.0, 0.0
    
    # VaR is the quantile
    # e.g. 95% confidence means looking at 5% worst returns
    cutoff = (1.0 - confidence) * 100
    var = np.percentile(returns, cutoff)
    
    # CVaR is the mean of returns worse than VaR
    cvar = returns[returns <= var].mean()
    
    return var, cvar

def calculate_alpha_beta(asset_returns, benchmark_returns, periods=252):
    """
    Calculates Alpha and Beta relative to a benchmark.
    Returns (Alpha_Annual, Beta).
    """
    # Align data
    common_idx = asset_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 30: return 0.0, 1.0 # Not enough data
    
    y = asset_returns.loc[common_idx]
    x = benchmark_returns.loc[common_idx]
    
    covariance = np.cov(x, y)[0][1]
    variance = np.var(x)
    
    if variance == 0: return 0.0, 0.0
    
    beta = covariance / variance
    
    # Alpha (Jensen's Alpha) = R_p - [R_f + Beta * (R_m - R_f)]
    # Simplified Alpha from regression intercept: y = alpha + beta*x
    # alpha_daily = mean(y) - beta * mean(x)
    alpha_daily = np.mean(y) - beta * np.mean(x)
    alpha_annual = alpha_daily * periods
    
    return alpha_annual, beta

def calculate_information_ratio(asset_returns, benchmark_returns):
    """
    Calculates Information Ratio: (R_p - R_b) / Tracking Error
    """
    common_idx = asset_returns.index.intersection(benchmark_returns.index)
    if len(common_idx) < 30: return 0.0
    
    diff = asset_returns.loc[common_idx] - benchmark_returns.loc[common_idx]
    
    mean_active_return = np.mean(diff)
    tracking_error = np.std(diff)
    
    if tracking_error == 0: return 0.0
    
    return (mean_active_return / tracking_error) * np.sqrt(252)

def calculate_trade_stats(equity_curve):
    """
    Approximates trading stats from equity curve (daily resolution).
    Ideally needs trade logs, but we can infer 'winning days' vs 'losing days'.
    """
    returns = equity_curve.pct_change().dropna()
    
    winning_days = returns[returns > 0]
    losing_days = returns[returns < 0]
    
    win_rate = len(winning_days) / len(returns) if len(returns) > 0 else 0
    
    avg_win = winning_days.mean() if len(winning_days) > 0 else 0
    avg_loss = abs(losing_days.mean()) if len(losing_days) > 0 else 0
    
    risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
    
    gross_win = winning_days.sum()
    gross_loss = abs(losing_days.sum())
    
    profit_factor = gross_win / gross_loss if gross_loss > 0 else np.inf
    
    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "risk_reward": risk_reward,
        "profit_factor": profit_factor
    }


# ─────────────────────────────────────────────────────────────────────────────
# NEW SCIENTIFIC METRICS (2024 upgrade)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_omega(returns, threshold=0.0):
    """
    Omega Ratio (Shadwick & Keating, 2002).
    Omega = Integral of (1-F(r)) dr above threshold
           / Integral of F(r) dr below threshold
    Simplified as: sum of gains above threshold / sum of losses below threshold.
    Omega > 1 means more weighted return above threshold than below.
    Perfect for Barbell strategies with asymmetric return profiles.
    """
    if len(returns) == 0:
        return 1.0
    excess = returns - threshold
    gains = np.sum(np.maximum(excess, 0))
    losses = np.sum(np.maximum(-excess, 0))
    if losses == 0:
        return np.inf
    return gains / losses


def calculate_ulcer_index(prices):
    """
    Ulcer Index (Martin & McCann, 1989).
    Measures the depth and duration of drawdowns — investor 'pain'.
    UI = sqrt(mean(drawdown_pct^2))
    Unlike Max Drawdown, UI punishes prolonged drawdowns, not just deep ones.
    """
    if isinstance(prices, pd.Series):
        arr = prices.values
    else:
        arr = np.array(prices)
    if len(arr) < 2:
        return 0.0
    peaks = np.maximum.accumulate(arr)
    drawdown_pct = ((arr - peaks) / peaks) * 100  # as percentage
    return np.sqrt(np.mean(drawdown_pct ** 2))


def calculate_pain_index(prices):
    """
    Pain Index — average drawdown depth over the entire period.
    Simpler than Ulcer Index but useful as companion metric.
    """
    if isinstance(prices, pd.Series):
        arr = prices.values
    else:
        arr = np.array(prices)
    peaks = np.maximum.accumulate(arr)
    drawdowns = (arr - peaks) / peaks
    return np.mean(np.abs(drawdowns))


def calculate_drawdown_analytics(prices):
    """
    Full drawdown analytics suite.
    Returns dict with:
    - max_drawdown: deepest single drawdown
    - avg_drawdown_depth: average depth across all drawdown periods
    - avg_drawdown_duration: average number of days to recover
    - ulcer_index: Martin & McCann UI
    - pain_index: mean absolute drawdown
    - drawdown_at_risk_95: worst 5% case drawdown (analogous to CVaR)
    Reference: Magdon-Ismail & Atiya (2004), Chekhlov et al. (2005)
    """
    if isinstance(prices, pd.Series):
        arr = prices.values
    else:
        arr = np.array(prices)

    peaks = np.maximum.accumulate(arr)
    drawdowns = (arr - peaks) / peaks  # non-positive values

    # Max drawdown
    max_dd = float(np.min(drawdowns))

    # Ulcer Index and Pain Index
    dd_pct = drawdowns * 100
    ulcer = float(np.sqrt(np.mean(dd_pct ** 2)))
    pain = float(np.mean(np.abs(drawdowns)))

    # Drawdown-at-Risk 95% (worst 5% of daily drawdown values)
    dd_at_risk_95 = float(np.percentile(drawdowns, 5))  # 5th percentile (most negative)

    # Drawdown periods: duration analysis
    in_drawdown = drawdowns < 0
    durations = []
    current_duration = 0
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0
    if current_duration > 0:
        durations.append(current_duration)

    avg_duration = float(np.mean(durations)) if durations else 0.0
    max_duration = int(np.max(durations)) if durations else 0

    return {
        "max_drawdown": max_dd,
        "avg_drawdown_depth": float(np.mean(drawdowns[drawdowns < 0])) if np.any(drawdowns < 0) else 0.0,
        "avg_drawdown_duration_days": avg_duration,
        "max_drawdown_duration_days": max_duration,
        "ulcer_index": ulcer,
        "pain_index": pain,
        "drawdown_at_risk_95": dd_at_risk_95,
    }

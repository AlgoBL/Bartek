import numpy as np
import pandas as pd

def calculate_sharpe(returns, rf=0.04, periods=252):
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

def calculate_sortino(returns, rf=0.04, periods=252, target_return=0.0):
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

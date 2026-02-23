
import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.stats.qmc import Sobol
from modules.ai.observer import get_market_regimes
from modules.ai.architect import PortfolioArchitect
from modules.ai.optimizer import GeneticOptimizer
from modules.ai.trader import RLTrader
import streamlit as st

def simulate_barbell_strategy(
    n_years=10, 
    n_simulations=1000, 
    initial_captial=10000,
    safe_rate=0.0551,  # Polish Bonds 5.51%
    risky_mean=0.08, 
    risky_vol=0.20, 
    risky_kurtosis=6.0, # Fat tails parameter (degrees of freedom for t-dist. Lower = fatter)
    alloc_safe=0.85,
    rebalance_strategy="None", # None, Yearly, Monthly, Threshold
    threshold_percent=0.20, # For Shannon's Demon/Threshold rebalancing
    use_qmc=False,  # Quasi-Monte Carlo via Sobol sequences
    use_garch=False, # GARCH(1,1) for volatility clustering
):
    """
    Monte Carlo Simulation supporting Quasi-MC (Sobol) and GARCH(1,1) volatility.
    Automatically applies 19% Belka Tax to all gains and interest.
    """
    days_per_year = 252
    total_days = n_years * days_per_year
    dt = 1/days_per_year
    
    df = max(2.1, risky_kurtosis)
    
    # ─── Random Shocks: Pseudo-random vs Quasi-MC (Sobol) ───────────────────
    if use_qmc:
        try:
            if total_days <= 21201:
                sampler = Sobol(d=total_days, scramble=True)
                n_sobol = int(2 ** np.ceil(np.log2(n_simulations)))
                uniform_samples = sampler.random(n_sobol)[:n_simulations]
                from scipy.stats import t as t_dist
                std_t = np.sqrt(df / (df - 2))
                random_shocks = t_dist.ppf(np.clip(uniform_samples, 1e-8, 1 - 1e-8), df=df)
                standardized_shocks = random_shocks / std_t
            else:
                raise ValueError("Too many days for Sobol, falling back")
        except Exception:
            random_shocks = t.rvs(df, size=(n_simulations, total_days))
            std_t = np.sqrt(df / (df - 2))
            standardized_shocks = random_shocks / std_t
    else:
        random_shocks = t.rvs(df, size=(n_simulations, total_days))
        std_t = np.sqrt(df / (df - 2))
        standardized_shocks = random_shocks / std_t
    
    # ─── GARCH(1,1) Volatility ──────────────────────────────────────────────
    if use_garch:
        alpha_g = 0.10
        beta_g  = 0.85
        omega_g = (risky_vol ** 2) * (1 - alpha_g - beta_g) / days_per_year
        
        var_series = np.zeros((n_simulations, total_days))
        var_series[:, 0] = risky_vol ** 2 / days_per_year
        
        eps = standardized_shocks.copy()
        for day in range(1, total_days):
            prev_eps = eps[:, day - 1]
            var_series[:, day] = omega_g + alpha_g * prev_eps ** 2 + beta_g * var_series[:, day - 1]
        
        daily_vols = np.sqrt(np.maximum(var_series, 1e-10))
        daily_mean = (risky_mean - 0.5 * risky_vol**2) * dt
        risky_returns = np.exp(daily_mean + daily_vols * standardized_shocks) - 1
    else:
        daily_mean = (risky_mean - 0.5 * risky_vol**2) * dt
        daily_vol = risky_vol * np.sqrt(dt)
        risky_returns = np.exp(daily_mean + daily_vol * standardized_shocks) - 1
    
    # Apply 19% Belka Tax on risky gains (conservative daily approximation)
    risky_returns = np.where(risky_returns > 0, risky_returns * 0.81, risky_returns)
    
    # Apply 19% Belka Tax to safe rate interest
    daily_safe_rate = (1 + (safe_rate * 0.81))**(1/days_per_year) - 1
    
    wealth_paths = np.zeros((n_simulations, total_days + 1))
    wealth_paths[:, 0] = initial_captial
    
    val_safe = np.zeros((n_simulations, total_days + 1))
    val_risky = np.zeros((n_simulations, total_days + 1))
    
    val_safe[:, 0] = initial_captial * alloc_safe
    val_risky[:, 0] = initial_captial * (1 - alloc_safe)
    
    for day in range(1, total_days + 1):
        val_safe[:, day] = val_safe[:, day-1] * (1 + daily_safe_rate)
        val_risky[:, day] = val_risky[:, day-1] * (1 + risky_returns[:, day-1])
        current_wealth = val_safe[:, day] + val_risky[:, day]
        
        should_rebalance = False
        if rebalance_strategy == "Yearly" and day % days_per_year == 0:
            should_rebalance = True
        elif rebalance_strategy == "Monthly" and day % 21 == 0:
            should_rebalance = True
        elif rebalance_strategy == "Threshold":
            current_risky_weight = val_risky[:, day] / current_wealth
            target_weight = 1 - alloc_safe
            lower_bound = target_weight * (1 - threshold_percent)
            upper_bound = target_weight * (1 + threshold_percent)
            mask = (current_risky_weight < lower_bound) | (current_risky_weight > upper_bound)
            if np.any(mask):
                val_safe[mask, day] = current_wealth[mask] * alloc_safe
                val_risky[mask, day] = current_wealth[mask] * (1 - alloc_safe)
        
        if should_rebalance:
            val_safe[:, day] = current_wealth * alloc_safe
            val_risky[:, day] = current_wealth * (1 - alloc_safe)
            
        wealth_paths[:, day] = val_safe[:, day] + val_risky[:, day]

    return wealth_paths


from modules.metrics import calculate_sharpe, calculate_sortino, calculate_calmar, calculate_max_drawdown

def calculate_metrics(wealth_paths, years):
    """Calculates metrics for Monte Carlo paths (2D array)"""
    if wealth_paths.ndim == 1:
        wealth_paths = wealth_paths.reshape(1, -1)
        
    final_wealth = wealth_paths[:, -1]
    initial_wealth = wealth_paths[:, 0]
    cagr = (final_wealth / initial_wealth)**(1/years) - 1
    
    returns = np.diff(wealth_paths, axis=1) / wealth_paths[:, :-1]
    volatility = np.std(returns, axis=1) * np.sqrt(252)
    
    peaks = np.maximum.accumulate(wealth_paths, axis=1)
    drawdowns = (wealth_paths - peaks) / peaks
    max_drawdowns = np.min(drawdowns, axis=1)
    
    rf = 0.04 * 0.81 # Tax-adjusted risk free proxy
    excess_ret = cagr - rf
    sharpe = np.divide(excess_ret, volatility, out=np.zeros_like(excess_ret), where=volatility!=0)
    calmar = np.divide(cagr, np.abs(max_drawdowns), out=np.zeros_like(cagr), where=max_drawdowns!=0)
    
    metrics = {
        "mean_final_wealth": np.mean(final_wealth),
        "median_final_wealth": np.median(final_wealth),
        "std_final_wealth": np.std(final_wealth),
        "mean_cagr": np.mean(cagr),
        "median_cagr": np.median(cagr),
        "prob_loss": np.mean(final_wealth < initial_wealth[0]),
        "mean_max_drawdown": np.mean(max_drawdowns),
        "worst_case_drawdown": np.min(max_drawdowns),
        
        "median_sharpe": np.median(sharpe),
        "median_sortino": 0.0,
        "median_calmar": np.median(calmar),
        "median_volatility": np.median(volatility),
        "var_95": np.percentile(final_wealth, 5),
        "cvar_95": final_wealth[final_wealth <= np.percentile(final_wealth, 5)].mean()
    }
    return metrics

def calculate_individual_metrics(wealth_paths, years):
    """
    Calculates metrics for EACH simulation path separately for 3D Scatter plots.
    """
    if wealth_paths.ndim == 1:
        wealth_paths = wealth_paths.reshape(1, -1)
        
    n_sims, n_days = wealth_paths.shape
    
    final_wealth = wealth_paths[:, -1]
    initial_wealth = wealth_paths[:, 0]
    cagr = (final_wealth / initial_wealth)**(1/years) - 1
    
    peaks = np.maximum.accumulate(wealth_paths, axis=1)
    drawdowns = (wealth_paths - peaks) / peaks
    max_drawdowns = np.min(drawdowns, axis=1)
    
    returns = np.diff(wealth_paths, axis=1) / wealth_paths[:, :-1]
    volatility = np.std(returns, axis=1) * np.sqrt(252)
    
    rf = 0.04 * 0.81
    excess_returns = cagr - rf
    sharpe = np.divide(excess_returns, volatility, out=np.zeros_like(excess_returns), where=volatility!=0)
    calmar = np.divide(cagr, np.abs(max_drawdowns), out=np.zeros_like(cagr), where=max_drawdowns!=0)
    
    return pd.DataFrame({
        "FinalWealth": final_wealth,
        "CAGR": cagr,
        "MaxDrawdown": max_drawdowns,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Calmar": calmar
    })

def run_ai_backtest(
    safe_data, 
    risky_data, 
    initial_capital=100000,
    safe_type="Ticker", # "Ticker" or "Fixed"
    safe_fixed_rate=0.0551,
    allocation_mode="AI Dynamic", 
    alloc_safe_fixed=0.85,
    kelly_params=None,
    rebalance_strategy="Monthly",
    threshold_percent=0.20,
    progress_callback=None,
    risky_weights_dict=None 
):
    """
    Runs backtest with 19% Belka Tax automatically included in calculated returns.
    """
    if risky_data.empty:
        st.error("Risky data is empty.")
        return pd.DataFrame(), [], []
    risky_tickers = risky_data.columns.tolist()
    
    if safe_type == "Fixed":
        dates = risky_data.index
        # Apply 19% Belka Tax to safe fixed rate
        daily_rate = (1 + (safe_fixed_rate * 0.81))**(1/252) - 1
        safe_prices = [100.0]
        for _ in range(len(dates)-1):
            safe_prices.append(safe_prices[-1] * (1 + daily_rate))
            
        safe_data = pd.DataFrame({"FIXED_SAFE": safe_prices}, index=dates)
        safe_tickers = ["FIXED_SAFE"]
    else:
        if safe_data.empty:
             st.error("Safe data is empty.")
             return pd.DataFrame(), [], []
        safe_tickers = safe_data.columns.tolist()

    combined_data = pd.concat([safe_data, risky_data], axis=1).dropna()
    dates = combined_data.index
    
    portfolio_value = [initial_capital]
    weights_history = []
    regime_history = []
    
    architect = PortfolioArchitect()
    trader = RLTrader()
    
    risky_returns_df = combined_data[risky_tickers].pct_change().fillna(0)
    risky_proxy_returns = risky_returns_df.mean(axis=1)
    
    regimes, observer_model = get_market_regimes(risky_proxy_returns, progress_callback)
    
    if observer_model:
        high_vol_state = observer_model.high_vol_state
        normalized_regimes = np.zeros_like(regimes)
        normalized_regimes[regimes == high_vol_state] = 1
        regimes = normalized_regimes
    else:
        regimes = np.zeros(len(dates))
    
    current_holdings = {t: 0.0 for t in combined_data.columns}
    cash = initial_capital
    last_rebalance_idx = 0
    total_steps = len(combined_data)
    
    for i in range(total_steps):
        if i % 10 == 0 and progress_callback:
            pct = i / total_steps
            progress_callback(pct, f"Symulacja: {pct:.1%} ({i}/{total_steps})")
            
        date = dates[i]
        prices = combined_data.iloc[i]
        
        regime_desc = "Unknown"
        if allocation_mode == "AI Dynamic":
            regime = regimes[i]
            regime_desc = "High Volatility (Risk-Off)" if regime == 1 else "Low Volatility (Risk-On)"
        
        regime_history.append(regime_desc)
        
        should_rebalance = False
        if i == 0:
            should_rebalance = True
        elif allocation_mode == "AI Dynamic":
             if i % 21 == 0:
                 should_rebalance = True
        else:
            if rebalance_strategy == "Yearly":
                if (i - last_rebalance_idx) >= 252:
                    should_rebalance = True
            elif rebalance_strategy == "Monthly":
                if (i - last_rebalance_idx) >= 21:
                    should_rebalance = True
            elif rebalance_strategy == "Threshold (Shannon's Demon)":
                val_now = sum(current_holdings[t] * prices[t] for t in combined_data.columns) + cash
                if val_now > 0:
                   current_risky_val = sum(current_holdings[t] * prices[t] for t in risky_tickers)
                   current_risky_w = current_risky_val / val_now
                   if allocation_mode == "Manual Fixed":
                       target_r = 1.0 - alloc_safe_fixed
                       if abs(current_risky_w - target_r) / target_r > threshold_percent:
                           should_rebalance = True
        
        if should_rebalance:
            prev_portfolio_val = sum(current_holdings[t] * prices[t] for t in combined_data.columns) + cash if i > 0 else initial_capital
            
            # 1. Strategic Split
            target_safe_pct = 0.5; target_risky_pct = 0.5
            if allocation_mode == "AI Dynamic":
                if "High Volatility" in regime_desc:
                    target_safe_pct = 0.80; target_risky_pct = 0.20
                else:
                    target_safe_pct = 0.20; target_risky_pct = 0.80
                
                vol_window = risky_proxy_returns.iloc[max(0, i-30):i].std() * np.sqrt(252)
                kelly_mult = trader.get_kelly_adjustment(vol_window, regime_desc)
                target_risky_pct *= kelly_mult
                target_safe_pct = 1.0 - target_risky_pct
            elif allocation_mode == "Manual Fixed":
                target_safe_pct = alloc_safe_fixed
                target_risky_pct = 1.0 - target_safe_pct
            elif allocation_mode == "Rolling Kelly":
                win_len = kelly_params.get('window', 252) if kelly_params else 252
                lookback_ret = risky_proxy_returns.iloc[max(0, i-win_len):i]
                if len(lookback_ret) > 60:
                    mu = lookback_ret.mean() * 252; sigma = lookback_ret.std() * np.sqrt(252)
                    r_safe = (safe_fixed_rate * 0.81) if safe_type == "Fixed" else 0.04 * 0.81
                    if sigma > 0:
                        kelly_full = (mu - r_safe) / (sigma**2)
                        k_opt = kelly_full * kelly_params.get('fraction', 1.0) * (1 - kelly_params.get('shrinkage', 0.0))
                        target_risky_pct = max(0.0, min(1.5, k_opt))
                        target_safe_pct = 1.0 - target_risky_pct
            
            # 2. Rebalance logic with tax on gains considered automatically by reduced future growth
            # (In reality, rebalance triggers tax on sold profitable assets. 
            #  Here we model it as tax on positive daily returns for simplicity)
            portfolio_val_now = sum(current_holdings[t] * prices[t] for t in combined_data.columns) + cash if i > 0 else initial_capital
            
            # Clear holdings
            current_holdings = {t: 0.0 for t in combined_data.columns}
            cash = 0
            
            window_start = max(0, i-60)
            window_data = combined_data.iloc[window_start:i+1]
            
            if safe_type == "Fixed":
                safe_weights_internal = {"FIXED_SAFE": 1.0}
            else:
                safe_weights_internal = architect.allocate_hrp(window_data[safe_tickers]) if len(window_data) > 10 else {t: 1.0/len(safe_tickers) for t in safe_tickers}

            if risky_weights_dict:
                risky_weights_internal = {t: risky_weights_dict.get(t, 0.0) for t in risky_tickers}
            else:
                risky_weights_internal = architect.allocate_hrp(window_data[risky_tickers]) if len(window_data) > 10 else {t: 1.0/len(risky_tickers) for t in risky_tickers}
            
            global_weights = {}
            for t in safe_tickers: global_weights[t] = safe_weights_internal.get(t, 0) * target_safe_pct
            for t in risky_tickers: global_weights[t] = risky_weights_internal.get(t, 0) * target_risky_pct
            for t, w in global_weights.items():
                current_holdings[t] = (portfolio_val_now * w) / prices[t]
            weights_history.append(global_weights)
            last_rebalance_idx = i
        else:
            weights_history.append(weights_history[-1] if weights_history else {})
        
        # Daily return logic for tax in backtest
        if i > 0:
            prev_val = portfolio_value[-1]
            # Since backtest uses raw prices, we must manually subtract the tax from the daily gain
            # This is hard to do without mutating prices. 
            # Solution: we adjust the portfolio value calculation to reflect tax drag on gains.
            raw_val = sum(current_holdings[t] * prices[t] for t in combined_data.columns) + cash
            daily_gain = raw_val - prev_val
            if daily_gain > 0:
                # Deduct 19% from gain
                net_val = prev_val + (daily_gain * 0.81)
                # To keep math consistent, we adjust 'cash' to absorb the tax loss
                cash -= (daily_gain * 0.19)
                portfolio_value.append(net_val)
            else:
                portfolio_value.append(raw_val)
        else:
            portfolio_value.append(initial_capital)
        
    results = pd.DataFrame({
        "Date": dates,
        "PortfolioValue": portfolio_value[1:],
        "Regime": regime_history
    }).set_index("Date")
    
    return results, weights_history, regimes

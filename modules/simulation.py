
import numpy as np
import pandas as pd
from scipy.stats import t
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
    threshold_percent=0.20 # For Shannon's Demon/Threshold rebalancing
):
    """
    Original Monte Carlo Simulation (Kept for compatibility).
    """
    days_per_year = 252
    total_days = n_years * days_per_year
    dt = 1/days_per_year
    
    df = max(2.1, risky_kurtosis) 
    random_shocks = t.rvs(df, size=(n_simulations, total_days))
    std_t = np.sqrt(df / (df - 2))
    standardized_shocks = random_shocks / std_t
    
    daily_mean = (risky_mean - 0.5 * risky_vol**2) * dt
    daily_vol = risky_vol * np.sqrt(dt)
    risky_returns = np.exp(daily_mean + daily_vol * standardized_shocks) - 1
    
    daily_safe_rate = (1 + safe_rate)**(1/days_per_year) - 1
    
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

def calculate_metrics(wealth_paths, years):
    """Calculates metrics for Monte Carlo paths (2D array)"""
    # Handle 1D array (from backtest) by reshaping to (1, N)
    if wealth_paths.ndim == 1:
        wealth_paths = wealth_paths.reshape(1, -1)
        
    final_wealth = wealth_paths[:, -1]
    initial_wealth = wealth_paths[:, 0]
    cagr = (final_wealth / initial_wealth)**(1/years) - 1
    
    peaks = np.maximum.accumulate(wealth_paths, axis=1)
    drawdowns = (wealth_paths - peaks) / peaks
    max_drawdowns = np.min(drawdowns, axis=1)
    
    metrics = {
        "mean_final_wealth": np.mean(final_wealth),
        "median_final_wealth": np.median(final_wealth),
        "std_final_wealth": np.std(final_wealth),
        "mean_cagr": np.mean(cagr),
        "median_cagr": np.median(cagr),
        "prob_loss": np.mean(final_wealth < initial_wealth[0]),
        "mean_max_drawdown": np.mean(max_drawdowns),
        "worst_case_drawdown": np.min(max_drawdowns)
    }
    return metrics

def calculate_individual_metrics(wealth_paths, years):
    """
    Calculates metrics for EACH simulation path separately for 3D Scatter plots.
    Returns a DataFrame with columns: FinalWealth, CAGR, MaxDrawdown, Volatility, Sharpe.
    """
    if wealth_paths.ndim == 1:
        wealth_paths = wealth_paths.reshape(1, -1)
        
    n_sims, n_days = wealth_paths.shape
    dt = 1/252
    
    # 1. Final Wealth & CAGR
    final_wealth = wealth_paths[:, -1]
    initial_wealth = wealth_paths[:, 0]
    cagr = (final_wealth / initial_wealth)**(1/years) - 1
    
    # 2. Max Drawdown
    peaks = np.maximum.accumulate(wealth_paths, axis=1)
    drawdowns = (wealth_paths - peaks) / peaks
    max_drawdowns = np.min(drawdowns, axis=1) # Negative values
    
    # 3. Volatility (Annualized)
    # Calculate daily returns
    # We can do diff / shift, but for matrix it's easier:
    # returns = (paths[:, 1:] - paths[:, :-1]) / paths[:, :-1]
    returns = np.diff(wealth_paths, axis=1) / wealth_paths[:, :-1]
    volatility = np.std(returns, axis=1) * np.sqrt(252)
    
    # 4. Sharpe Ratio (assuming 4% risk free approx or just 0 for raw ratio)
    rf = 0.04
    excess_returns = cagr - rf
    sharpe = np.divide(excess_returns, volatility, out=np.zeros_like(excess_returns), where=volatility!=0)
    
    return pd.DataFrame({
        "FinalWealth": final_wealth,
        "CAGR": cagr,
        "MaxDrawdown": max_drawdowns,
        "Volatility": volatility,
        "Sharpe": sharpe
    })

def run_ai_backtest(
    safe_data, 
    risky_data, 
    initial_capital=100000,
    safe_type="Ticker", # "Ticker" or "Fixed"
    safe_fixed_rate=0.0551,
    allocation_mode="AI Dynamic", # "AI Dynamic", "Manual Fixed", "Rolling Kelly"
    alloc_safe_fixed=0.85, # For Manual Fixed
    kelly_params=None, # dict with fraction, shrinkage, window
    rebalance_strategy="Monthly", # "None", "Yearly", "Monthly", "Threshold"
    threshold_percent=0.20,
    progress_callback=None, # Function(float, str)
    risky_weights_dict=None # Dict {Ticker: Weight} for manual allocation
):
    """
    Runs the advanced AI-driven backtest using historical data with expanded options.
    """
    # 1. Prepare Data
    # Identify Risky Tickers & Data
    if risky_data.empty:
        st.error("Risky data is empty.")
        return pd.DataFrame(), [], []
    risky_tickers = risky_data.columns.tolist()
    
    # Handle Safe Data (Ticker vs Fixed)
    if safe_type == "Fixed":
        # Generate synthetic safe asset
        # We need an index matching risky_data
        dates = risky_data.index
        daily_rate = (1 + safe_fixed_rate)**(1/252) - 1
        
        # Create a DataFrame with a single 'FIXED_SAFE' column
        # Starts at 1.0 and grows
        # Note: In the simulation loop, we just need the returns or prices.
        # Let's create a price series starting at 100.
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

    # If manual weights provided, validate tickers match data
    if risky_weights_dict:
        # Check if all keys are in risky_tickers
        missing = [t for t in risky_weights_dict.keys() if t not in risky_tickers]
        if missing:
            # warn? or just ignore missing?
            pass
            
    # Align data
    combined_data = pd.concat([safe_data, risky_data], axis=1).dropna()
    dates = combined_data.index
    
    portfolio_value = [initial_capital]
    weights_history = []
    regime_history = []
    
    # Initialize Modules
    architect = PortfolioArchitect()
    trader = RLTrader()
    # optimizer = GeneticOptimizer(num_generations=20) # Optional/Slow
    
    # Run HMM on Risky Assets (to detect regime for AI mode)
    risky_returns_df = combined_data[risky_tickers].pct_change().fillna(0)
    risky_proxy_returns = risky_returns_df.mean(axis=1) # Average of risky basket
    
    regimes = np.zeros(len(dates))
    observer_model = None
    
    if allocation_mode == "AI Dynamic":
        # Pass a wrapper or the callback directly. 
        # Since observer pushes 0.1->0.9, we might want to scale it or just let it update.
        # "Trenowanie model HMM..." is the pre-step before the big loop.
        regimes, observer_model = get_market_regimes(risky_proxy_returns, progress_callback)
    
    current_holdings = {t: 0.0 for t in combined_data.columns}
    cash = initial_capital
    
    # Helper for Rebalancing Check
    last_rebalance_idx = 0
    
    total_steps = len(combined_data)
    
    for i in range(total_steps):
        # Update progress every 10 steps or if it's slow
        if i % 10 == 0 and progress_callback:
            pct = i / total_steps
            progress_callback(pct, f"Symulacja: {pct:.1%} ({i}/{total_steps})")
            
        date = dates[i]

        prices = combined_data.iloc[i]
        
        # Determine Regime (if AI)
        regime_desc = "Unknown"
        if allocation_mode == "AI Dynamic":
            regime = regimes[i]
            regime_desc = observer_model.get_regime_desc(regime)
        
        regime_history.append(regime_desc)
        
        # --- Check Rebalance Trigger ---
        should_rebalance = False
        
        # Always rebalance on day 0
        if i == 0:
            should_rebalance = True
            
        elif allocation_mode == "AI Dynamic":
             # AI typically rebalances Monthly (or could be dynamic, but let's stick to Monthly/21 days)
             if i % 21 == 0:
                 should_rebalance = True
                 
        else: # Manual or Kelly
            if rebalance_strategy == "Yearly":
                # Aprox 252 days
                if (i - last_rebalance_idx) >= 252:
                    should_rebalance = True
            elif rebalance_strategy == "Monthly":
                if (i - last_rebalance_idx) >= 21:
                    should_rebalance = True
            elif rebalance_strategy == "None":
                should_rebalance = False
            elif rebalance_strategy == "Threshold (Shannon's Demon)":
                 # Check deviation
                val_now = sum(current_holdings[t] * prices[t] for t in combined_data.columns) + cash
                if val_now > 0:
                   current_risky_val = sum(current_holdings[t] * prices[t] for t in risky_tickers)
                   current_risky_w = current_risky_val / val_now
                   
                   # What is the target? It depends on allocation mode...
                   # This is tricky for Kelly as target changes. 
                   # Assuming Fixed Target for Manual, Kelly triggers its own check? 
                   # For simplicity, Threshold applies to deviation from *current intended target*.
                   # We need to calculate target first? No, that causes chicken-egg.
                   # Let's assume Threshold only works well with Fixed Manual Targets.
                   if allocation_mode == "Manual Fixed":
                       target_r = 1.0 - alloc_safe_fixed
                       if abs(current_risky_w - target_r) / target_r > threshold_percent:
                           should_rebalance = True
        
        # --- Execute Rebalance ---
        if should_rebalance:
            last_rebalance_idx = i
            
            # 1. Determine Strategic Split (Safe vs Risky)
            target_safe_pct = 0.5
            target_risky_pct = 0.5
            
            if allocation_mode == "AI Dynamic":
                if "High Volatility" in regime_desc:
                    target_safe_pct = 0.80; target_risky_pct = 0.20
                else:
                    target_safe_pct = 0.20; target_risky_pct = 0.80
                
                # RL Adjustment
                vol_window = risky_proxy_returns.iloc[max(0, i-30):i].std() * np.sqrt(252)
                kelly_mult = trader.get_kelly_adjustment(vol_window, regime_desc)
                target_risky_pct *= kelly_mult
                target_safe_pct = 1.0 - target_risky_pct
                
            elif allocation_mode == "Manual Fixed":
                target_safe_pct = alloc_safe_fixed
                target_risky_pct = 1.0 - target_safe_pct
                
            elif allocation_mode == "Rolling Kelly":
                # Calculate rolling stats
                win_len = kelly_params.get('window', 252) if kelly_params else 252
                lookback_ret = risky_proxy_returns.iloc[max(0, i-win_len):i]
                
                if len(lookback_ret) > 60:
                    mu = lookback_ret.mean() * 252
                    sigma = lookback_ret.std() * np.sqrt(252)
                    r_safe = safe_fixed_rate if safe_type == "Fixed" else 0.04 # approx
                    
                    if sigma > 0:
                        kelly_full = (mu - r_safe) / (sigma**2)
                    else:
                        kelly_full = 0
                        
                    frac = kelly_params.get('fraction', 1.0)
                    shrink = kelly_params.get('shrinkage', 0.0)
                    k_opt = kelly_full * frac * (1 - shrink)
                    k_opt = max(0.0, min(1.0, k_opt)) # Long only, no leverage > 1 for this basic impl? 
                    # User might want leverage? "Dynamicznie zarządza lewarem"
                    # Allowing up to max (e.g. 1.5 or 3?) - Let's cap at 1.5 for safety or just 1.0 if not specified.
                    # Standard implementation usually caps at 1 or 2. Let's cap at 1.5.
                    k_opt = min(1.5, k_opt) # Cap leverage
                    
                    target_risky_pct = k_opt
                    target_safe_pct = 1.0 - target_risky_pct
                else:
                     target_risky_pct = 0.5
                     target_safe_pct = 0.5

            # 2. Asset Allocation within Baskets
            # Use HRP (Architect) for both
            window_start = max(0, i-60)
            window_data = combined_data.iloc[window_start:i+1]
            
            # Safe Basket Internal Weights
            if safe_type == "Fixed":
                safe_weights_internal = {"FIXED_SAFE": 1.0}
            else:
                if len(window_data) > 10:
                    safe_weights_internal = architect.allocate_hrp(window_data[safe_tickers])
                else:
                    safe_weights_internal = {t: 1.0/len(safe_tickers) for t in safe_tickers}

            # Risky Basket Internal Weights
            if risky_weights_dict:
                # Use Manual Weights
                # Normalize just in case 
                # (Though UI ensures 100%, let's be safe or just map them)
                risky_weights_internal = {}
                for t in risky_tickers:
                    # If ticker in dict, use it. If not, 0.
                    risky_weights_internal[t] = risky_weights_dict.get(t, 0.0)
            else:
                 if len(window_data) > 10:
                      risky_weights_internal = architect.allocate_hrp(window_data[risky_tickers])
                 else:
                      risky_weights_internal = {t: 1.0/len(risky_tickers) for t in risky_tickers}
            
            # 3. Combine to Global Weights
            global_weights = {}
            # Handle leverage if target_risky_pct > 1.0? 
            # If > 1, sum of weights > 1. Borrowing cost? 
            # Simplified: Assuming cash can be negative or 'leverage' is implicit 
            # (but here we simulate explicit holdings). 
            # If target_safe is negative (borrowing), we need a borrowing rate.
            # Let's Normalize to 1.0 for now to avoid complexity unless margin is requested.
            # User request: "zarządza lewarem". 
            # If safe is negative, it means borrowing.
            
            for t in safe_tickers:
                global_weights[t] = safe_weights_internal.get(t, 0) * target_safe_pct
            for t in risky_tickers:
                global_weights[t] = risky_weights_internal.get(t, 0) * target_risky_pct
            
            # Execute Trade
            portfolio_val_now = sum(current_holdings[t] * prices[t] for t in combined_data.columns) + cash
            
            # Clear holdings
            current_holdings = {t: 0.0 for t in combined_data.columns}
            cash = 0
            
            # Buy
            for t, w in global_weights.items():
                # w * val / price
                current_holdings[t] = (portfolio_val_now * w) / prices[t]
                
            weights_history.append(global_weights)
            
        else:
            # Hold
            weights_history.append(weights_history[-1] if weights_history else {})
            
        # Calculate daily value
        val = sum(current_holdings[t] * prices[t] for t in combined_data.columns) + cash
        portfolio_value.append(val)
        

    
    # Prepare results
    results = pd.DataFrame({
        "Date": dates,
        "PortfolioValue": portfolio_value[1:], # Align length
        "Regime": regime_history
    }).set_index("Date")
    
    return results, weights_history, regimes

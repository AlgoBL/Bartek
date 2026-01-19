import numpy as np
import pandas as pd
from scipy.stats import t

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
    Simulates portfolio trajectories using Monte Carlo.
    
    Safe Asset: Fixed growth (deterministic).
    Risky Asset: Student's t-distribution to model fat tails (Taleb's Extremistan).
    """
    
    # Time parameters
    days_per_year = 252
    total_days = n_years * days_per_year
    dt = 1/days_per_year
    
    # 1. Generate Risky Asset Returns (Fat Tailed)
    # We use Student's t-distribution standardized and scaled to desired vol
    # df (degrees of freedom) controls tail thickness. 
    # Normal dist has df = infinity. Crypto/Taleb-world has df approx 3-5.
    df = max(2.1, risky_kurtosis) # Avoid infinite variance if df <= 2
    
    # Generate random shocks [n_simulations, total_days]
    random_shocks = t.rvs(df, size=(n_simulations, total_days))
    
    # Normalize shocks to have std dev = 1 (approx) then scale by vol
    # Variance of t-dist is df/(df-2), so we divide by sqrt of that to standardize
    std_t = np.sqrt(df / (df - 2))
    standardized_shocks = random_shocks / std_t
    
    # Drift and Diffusion
    # Geometric Brownian Motion adapted for t-shocks: exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    daily_mean = (risky_mean - 0.5 * risky_vol**2) * dt
    daily_vol = risky_vol * np.sqrt(dt)
    
    risky_returns = np.exp(daily_mean + daily_vol * standardized_shocks) - 1
    
    # Safe Asset Returns (Daily constant)
    daily_safe_rate = (1 + safe_rate)**(1/days_per_year) - 1
    
    # Arrays to store Wealth
    # wealth_paths shape: [n_simulations, total_days + 1]
    wealth_paths = np.zeros((n_simulations, total_days + 1))
    wealth_paths[:, 0] = initial_captial
    
    # Track Safe and Risky components separately for rebalancing
    val_safe = np.zeros((n_simulations, total_days + 1))
    val_risky = np.zeros((n_simulations, total_days + 1))
    
    # Initial Allocation
    val_safe[:, 0] = initial_captial * alloc_safe
    val_risky[:, 0] = initial_captial * (1 - alloc_safe)
    
    # Simulation Loop
    for day in range(1, total_days + 1):
        # Evolve assets
        # Safe asset grows deterministically
        val_safe[:, day] = val_safe[:, day-1] * (1 + daily_safe_rate)
        
        # Risky asset grows stochastically
        val_risky[:, day] = val_risky[:, day-1] * (1 + risky_returns[:, day-1])
        
        # Current Wealth
        current_wealth = val_safe[:, day] + val_risky[:, day]
        
        # Rebalancing Logic
        should_rebalance = False
        
        if rebalance_strategy == "Yearly" and day % days_per_year == 0:
            should_rebalance = True
        elif rebalance_strategy == "Monthly" and day % 21 == 0:
            should_rebalance = True
        elif rebalance_strategy == "Threshold":
            # Check current weights
            current_risky_weight = val_risky[:, day] / current_wealth
            target_weight = 1 - alloc_safe
            
            # Vectorized check: which simulations need rebalancing?
            # Rebalance if deviation > threshold_percent (relative deviation)
            # Safe zone: target * (1 - threshold) < weight < target * (1 + threshold)
            
            lower_bound = target_weight * (1 - threshold_percent)
            upper_bound = target_weight * (1 + threshold_percent)
            
            # Mask of simulations to rebalance
            mask = (current_risky_weight < lower_bound) | (current_risky_weight > upper_bound)
            
            # Perform rebalancing ONLY for masked simulations
            if np.any(mask):
                val_safe[mask, day] = current_wealth[mask] * alloc_safe
                val_risky[mask, day] = current_wealth[mask] * (1 - alloc_safe)
                # No 'should_rebalance' flag here because we handled it vector-wise
        
        if should_rebalance:
            val_safe[:, day] = current_wealth * alloc_safe
            val_risky[:, day] = current_wealth * (1 - alloc_safe)
            
        wealth_paths[:, day] = val_safe[:, day] + val_risky[:, day]

    return wealth_paths

def calculate_metrics(wealth_paths, n_years):
    final_wealth = wealth_paths[:, -1]
    initial_wealth = wealth_paths[:, 0]
    
    # CAGR
    cagr = (final_wealth / initial_wealth)**(1/n_years) - 1
    
    # Drawdowns (path dependent)
    # Calculate max drawdown for each path
    # Peak to date
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

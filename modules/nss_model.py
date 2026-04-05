import numpy as np
import pandas as pd
from scipy.optimize import minimize

def nss_yield_curve(maturities: np.ndarray, beta0: float, beta1: float, beta2: float, beta3: float, tau1: float, tau2: float) -> np.ndarray:
    """
    Równanie Nelson-Siegel-Svensson (NSS) dla krzywej dochodowości.
    Zwraca wyliczone yieldy dla podanych zapadalności (maturities).
    """
    term1 = (1 - np.exp(-maturities / tau1)) / (maturities / tau1)
    term2 = term1 - np.exp(-maturities / tau1)
    term3 = ((1 - np.exp(-maturities / tau2)) / (maturities / tau2)) - np.exp(-maturities / tau2)
    
    y = beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3
    return y

def fit_nss(maturities: np.ndarray, yields: np.ndarray) -> dict:
    """
    Dopasowuje parametry NSS do empirycznej krzywej dochodowości używając Least Squares.
    Zwraca słownik z parametrami.
    """
    def objective(params):
        b0, b1, b2, b3, t1, t2 = params
        if t1 <= 0 or t2 <= 0:
            return 1e10 # Penalize negative taus
            
        y_pred = nss_yield_curve(maturities, b0, b1, b2, b3, t1, t2)
        return np.sum((y_pred - yields)**2)
        
    # Heuristic initial guess
    b0_init = yields[-1] # Long term rate
    b1_init = yields[0] - b0_init # Short rate - Long rate (Spread)
    b2_init = 0.0
    b3_init = 0.0
    t1_init = 1.0 # Medium term (around 1-2 years)
    t2_init = 5.0 # Longer medium term (5-10 years)
    
    init_params = [b0_init, b1_init, b2_init, b3_init, t1_init, t2_init]
    
    # bounds
    bounds = (
        (0.0, 0.20),      # b0
        (-0.20, 0.20),    # b1
        (-0.20, 0.20),    # b2
        (-0.20, 0.20),    # b3
        (0.1, 30.0),      # t1
        (0.1, 30.0)       # t2
    )
    
    # Optimization
    res = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds)
    
    if res.success:
        b0, b1, b2, b3, t1, t2 = res.x
        return {
            "beta0": b0, "beta1": b1, "beta2": b2, "beta3": b3, 
            "tau1": t1, "tau2": t2, "error": np.sqrt(res.fun/len(maturities))
        }
    else:
        # Fallback to initial guess if fails
        return {
            "beta0": b0_init, "beta1": b1_init, "beta2": b2_init, "beta3": b3_init, 
            "tau1": t1_init, "tau2": t2_init, "error": np.nan
        }

def get_simulated_yield_curve() -> dict:
    """ Dostarcza sztuczne bieżące yieldy (np. US Treasuries) dla demonstracji. """
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    
    # Symulacja lekkiej inwersji (charakterystyczna dla late-cycle / recession probability)
    yields = np.array([0.054, 0.053, 0.051, 0.048, 0.046, 0.044, 0.045, 0.046, 0.048, 0.049])
    
    return {"maturities": maturities, "yields": yields}

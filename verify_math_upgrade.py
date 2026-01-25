
import pandas as pd
import numpy as np
from modules.ai.architect import PortfolioArchitect
from modules.ai.optimizer import GeneticOptimizer

def test_hrp_cvar():
    print("Testing HRP with CVaR...")
    # Synthetic prices
    dates = pd.date_range(start="2020-01-01", periods=100)
    data = {
        'AssetA': np.cumprod(1 + np.random.normal(0.001, 0.02, 100)), # Normal
        'AssetB': np.cumprod(1 + np.random.normal(0.001, 0.05, 100)), # Volatile
        'AssetC': np.cumprod(1 + np.random.standard_t(df=3, size=100)*0.02) # Fat Tailed
    }
    prices = pd.DataFrame(data, index=dates)
    
    architect = PortfolioArchitect()
    try:
        weights = architect.allocate_hrp(prices, risk_metric='cvar')
        print(f"CVaR Weights: {weights}")
    except Exception as e:
        print(f"HRP CVaR FAIL: {e}")
        raise e

def test_ga_sortino():
    print("Testing GA with Sortino...")
    returns = pd.DataFrame(np.random.normal(0.001, 0.02, (100, 5)), columns=['A','B','C','D','E'])
    
    optimizer = GeneticOptimizer(num_generations=5)
    try:
        weights = optimizer.optimize_portfolio(returns)
        print(f"GA Weights: {weights}")
    except Exception as e:
        print(f"GA Sortino FAIL: {e}")
        raise e

if __name__ == "__main__":
    test_hrp_cvar()
    test_ga_sortino()
    print("ALL TESTS PASSED")

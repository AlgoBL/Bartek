
from modules.simulation import run_ai_backtest
from modules.ai.data_loader import load_data
import pandas as pd
import numpy as np

# Mock data
dates = pd.date_range("2020-01-01", "2021-01-01", freq="B")
risky_data = pd.DataFrame({
    "SPY": 100 * (1 + np.random.randn(len(dates)) * 0.01).cumprod(),
    "QQQ": 100 * (1 + np.random.randn(len(dates)) * 0.015).cumprod()
}, index=dates)

print("--- Test 1: Fixed Rate + Manual Fixed 50/50 ---")
results, weights, regimes = run_ai_backtest(
    safe_data=pd.DataFrame(), # Empty
    risky_data=risky_data,
    initial_capital=100000,
    safe_type="Fixed",
    safe_fixed_rate=0.05,
    allocation_mode="Manual Fixed",
    alloc_safe_fixed=0.50,
    rebalance_strategy="Monthly"
)
print("Final Value:", results["PortfolioValue"].iloc[-1])
print("Last Weight (First Entry):", weights[-1])

print("\n--- Test 2: Fixed Rate + Rolling Kelly ---")
results, weights, regimes = run_ai_backtest(
    safe_data=pd.DataFrame(),
    risky_data=risky_data,
    initial_capital=100000,
    safe_type="Fixed",
    allocation_mode="Rolling Kelly",
    kelly_params={"fraction": 1.0, "shrinkage": 0.0, "window": 30},
    rebalance_strategy="Monthly"
)
print("Final Value:", results["PortfolioValue"].iloc[-1])
print("Regime History len:", len(regimes))

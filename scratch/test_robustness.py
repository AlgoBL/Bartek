
import sys
import os
import pandas as pd
import streamlit as st

# Mock streamlit
class MockSt:
    def error(self, msg): print(f"ST ERROR: {msg}")
    def warning(self, msg): print(f"ST WARNING: {msg}")
    def info(self, msg): print(f"ST INFO: {msg}")
    def cache_data(self, **kwargs):
        def decorator(func): return func
        return decorator

sys.modules["streamlit"] = MockSt()

sys.path.append(os.getcwd())

from modules.ai.data_loader import load_data

def test_robustness():
    # Mix of valid and invalid tickers
    tickers = ["SPY", "QQQ", "INVALID_TICKER_BLABLA", "BTC-USD"]
    print(f"Testing with tickers: {tickers}")
    df = load_data(tickers, start_date="2024-01-01")
    
    print("\n--- RESULTS ---")
    if df.empty:
        print("Final DF is EMPTY.")
    else:
        print(f"Final DF columns: {list(df.columns)}")
        print(f"Final DF shape: {df.shape}")
        print(f"Head:\n{df.head()}")

if __name__ == "__main__":
    test_robustness()

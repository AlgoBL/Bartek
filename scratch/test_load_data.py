
import sys
import os
import pandas as pd
import streamlit as st

# Mock streamlit for the script
class MockSt:
    def error(self, msg): print(f"ST ERROR: {msg}")
    def cache_data(self, **kwargs):
        def decorator(func): return func
        return decorator

sys.modules["streamlit"] = MockSt()

# Setup path
sys.path.append(os.getcwd())

from modules.ai.data_loader import load_data

def test_load_data():
    tickers = ["SPY", "QQQ", "BTC-USD"]
    start_date = "2024-01-01"
    print(f"Loading {tickers} from {start_date}...")
    df = load_data(tickers, start_date=start_date)
    print(f"Result empty: {df.empty}")
    if not df.empty:
        print(f"Columns: {df.columns}")
        print(f"First rows:\n{df.head()}")
    
    print("\nTesting single ticker...")
    df2 = load_data(["SPY"], start_date=start_date)
    print(f"Result empty: {df2.empty}")
    if not df2.empty:
        print(f"Columns: {df2.columns}")
        print(f"First rows:\n{df2.head()}")

if __name__ == "__main__":
    test_load_data()

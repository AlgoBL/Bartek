
import sys
import os
import pandas as pd
import asyncio

# Setup path
sys.path.append(os.getcwd())

from modules.data_provider import fetch_data

def test_fetch():
    tickers = ["SPY", "QQQ"]
    print(f"Fetching {tickers}...")
    df = fetch_data(tickers, period="1mo")
    print(f"Result empty: {df.empty}")
    if not df.empty:
        print(f"Columns: {df.columns}")
        print(f"First rows:\n{df.head()}")

if __name__ == "__main__":
    test_fetch()

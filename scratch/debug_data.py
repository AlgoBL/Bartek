
import sys
import os
import pandas as pd
from datetime import datetime

# Setup path
sys.path.append(os.getcwd())

from modules.ai.data_loader import load_data

def debug_full_load():
    tickers = ["SPY", "QQQ", "NVDA", "BTC-USD"]
    ai_years = 5
    start_date = (pd.Timestamp.today() - pd.DateOffset(years=ai_years)).strftime('%Y-%m-%d')
    
    for base_curr_arg in [None, "PLN"]:
        print(f"\n--- TESTING WITH BaseCurrency={base_curr_arg} ---")
    
    # Bypass streamlit cache by using a fresh function or clearing it if possible
    # We just call it directly in the script (st decorators are mocked/disabled in scripts if st is not running)
    
    try:
        data = load_data(tickers, start_date=start_date, base_currency=base_curr_arg)
        print("\n--- RESULTS ---")
        print(f"Empty: {data.empty}")
        if not data.empty:
            print(f"Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
            print(f"Index range: {data.index[0]} to {data.index[-1]}")
        else:
            print("DATA IS EMPTY!")
            
            # Deeper look into fetch_data
            from modules.data_provider import fetch_data
            raw_data = fetch_data(tickers, start=start_date, auto_adjust=False)
            print("\n--- RAW DATA ---")
            print(f"Raw empty: {raw_data.empty}")
            if not raw_data.empty:
                print(f"Raw columns type: {type(raw_data.columns)}")
                print(f"Raw columns: {raw_data.columns}")
                if isinstance(raw_data.columns, pd.MultiIndex):
                    print(f"Levels: {raw_data.columns.levels}")
                print(f"Raw head:\n{raw_data.head()}")
            else:
                print("RAW DATA IS EMPTY! Yahoo/Stooq completely failed.")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

if __name__ == "__main__":
    debug_full_load()


import pandas as pd
import sys
import os

# Ensure the root directory is in sys.path
root = r"c:\Users\bartl\OneDrive\KOd\Bartek"
if root not in sys.path:
    sys.path.append(root)

from modules.factor_model import build_factor_returns, run_factor_decomposition

def verify():
    print("Testing build_factor_returns with fix...")
    # build_factor_returns uses FF5_PROXY_TICKERS internally
    try:
        factor_df = build_factor_returns({})
        
        if factor_df is None or factor_df.empty:
            print("FAIL: factor_df is empty or None!")
            return False
        
        print(f"SUCCESS: factor_df columns: {factor_df.columns.tolist()}")
        print(f"Row count: {len(factor_df)}")
        
        expected = ["Rm_Rf", "SMB", "HML", "RMW", "CMA"]
        found = [c for c in expected if c in factor_df.columns]
        print(f"Factors found: {found}")
        
        if not found:
            print("FAIL: No factors found.")
            return False
            
        # Test decomposition with dummy data
        dummy_returns = pd.Series(0.0001, index=factor_df.index, name="TestPort")
        res = run_factor_decomposition(dummy_returns, factor_df)
        
        if "error" in res:
            print(f"FAIL: run_factor_decomposition error: {res['error']}")
            return False
            
        print("SUCCESS: Full cycle completed.")
        return True
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)

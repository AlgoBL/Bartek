import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def perform_cointegration_test(series1: pd.Series, series2: pd.Series) -> dict:
    """
    Krok 1 (Engle-Granger): OLS pierwszej serii względem drugiej
    Krok 2: Augmented Dickey-Fuller na resztach z modelu
    Zwraca słownik z testem stacjonarności.
    """
    if len(series1) != len(series2) or len(series1) < 30:
        return {"is_cointegrated": False, "p_value": 1.0, "hf": 0, "spread_z": pd.Series()}

    # Upewnienie się, że brak NAs
    df = pd.concat([series1, series2], axis=1).dropna()
    if len(df) < 30:
        return {"is_cointegrated": False, "p_value": 1.0, "hf": 0, "spread_z": pd.Series()}
        
    s1 = df.iloc[:, 0]
    s2 = df.iloc[:, 1]
    
    # Krok 1: Regresja s1 = beta * s2 + alpha
    X = sm.add_constant(s2)
    model = sm.OLS(s1, X).fit()
    hedge_ratio = model.params.iloc[1]
    
    # Spread = s1 - hedge_ratio * s2 - alpha
    spread = s1 - hedge_ratio * s2 - model.params.iloc[0]
    
    # Krok 2: ADF test na stationarnosc spreadu
    adf_result = adfuller(spread, maxlag=1)
    p_value = adf_result[1]
    
    # Obliczenie Z-Score spreadu
    spread_z = (spread - spread.mean()) / spread.std(ddof=1)
    
    return {
        "is_cointegrated": bool(p_value < 0.05),
        "p_value": p_value,
        "hedge_ratio": hedge_ratio,
        "spread_z": spread_z,
        "current_z": spread_z.iloc[-1],
        "spread_raw": spread
    }

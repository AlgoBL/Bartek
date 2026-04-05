
import pandas as pd
import streamlit as st
from modules.data_provider import fetch_data, fetch_currency_adjusted_data
from modules.logger import setup_logger

logger = setup_logger(__name__)

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(tickers, start_date=None, end_date=None, period=None, base_currency=None):
    """
    Fetches historical adjusted close prices for given tickers.
    If period is provided (e.g. '5y'), start_date and end_date are ignored.
    Uses Streamlit caching to prevent rate limiting and improve speed.
    """
    if start_date is None and period is None:
        start_date = "2000-01-01"
    if isinstance(tickers, str):
        tickers = [tickers]
        
    try:
        if base_currency == "PLN":
            data = fetch_currency_adjusted_data(tickers, start=start_date, end=end_date, period=period)
        else:
            data = fetch_data(tickers, start=start_date, end=end_date, period=period, auto_adjust=False)
        
        # yfinance > 0.2 returns MultiIndex (Price, Ticker) if multiple tickers, 
        # or just (Price) if single ticker BUT sometimes still MultiIndex.
        # We need 'Adj Close'.
        
        if 'Adj Close' in data.columns:
            data = data['Adj Close']
        elif 'Close' in data.columns:
             # Fallback if Adj Close not found (shouldn't happen with auto_adjust=False but safety first)
             data = data['Close']
        else:
            st.error(f"Błąd struktury danych: Brak kolumny 'Adj Close' lub 'Close'. Dostępne: {data.columns}")
            return pd.DataFrame()

        # If single ticker, it might be a Series or DataFrame with 1 col.
        if isinstance(data, pd.Series):
             data = data.to_frame()
             if len(tickers) == 1:
                 data.columns = tickers
                 
        # If DataFrame but only 1 column and name is not the ticker
        if isinstance(data, pd.DataFrame) and data.shape[1] == 1 and len(tickers) == 1:
             # Sometimes yfinance names the column 'Adj Close' instead of ticker symbol when single
             data.columns = tickers

        # Check for emptiness
        if data.empty:
            st.error("Pobrano puste dane.")
            return pd.DataFrame()
            
        return data.dropna()
        
    except Exception as e:
        logger.error(f"Błąd pobierania danych z Yahoo Finance: {e}")
        return pd.DataFrame()

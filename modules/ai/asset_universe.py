
import pandas as pd
import streamlit as st
import requests
import io
from modules.logger import setup_logger

logger = setup_logger(__name__)

@st.cache_data
def get_sp500_tickers():
    """
    Pobiera aktualną listę tickerów S&P 500 z Wikipedii.
    W przypadku błędu zwraca awaryjną listę TOP 50.
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # Yahoo Finance używa '-' zamiast '.' (np. BRK-B zamiast BRK.B)
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        logger.warning(f"Nie udało się pobrać S&P 500: {e}. Używam zapasowej listy.")
        st.warning("Nie udało się pobrać pełnej listy S&P 500. Używam listy zapasowej.")
        # Top 20 liquid stocks (fallback)
        return ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ', 'JPM', 'XOM', 'V', 'PG', 'MA', 'HD', 'CVX', 'ABBV', 'MRK', 'PEP']

@st.cache_data
def get_global_etfs():
    """
    Zwraca listę TOP 50 Globalnych ETF-ów (US & World).
    Zawiera główne indeksy, sektory, surowce i obligacje.
    """
    return [
        # --- US Broad Market ---
        'SPY', 'IVV', 'VOO', # S&P 500
        'QQQ', 'QQQM', # Nasdaq 100
        'DIA', # Dow Jones
        'IWM', # Russell 2000
        'VTI', # Total Stock Market
        
        # --- Global & Emerging ---
        'VT', # Total World Stock
        'VEA', # Developed Markets ex-US
        'VWO', 'EEM', # Emerging Markets
        'VXUS', # Total Intl Stock
        
        # --- Factors / Styles ---
        'VTV', # Value
        'VUG', # Growth
        'RSP', # S&P 500 Equal Weight
        'USMV', # Min Volatility
        'MTUM', # Momentum
        'QUAL', # Quality
        
        # --- Sectors (SPDR) ---
        'XLK', # Tech
        'XLF', # Financials
        'XLV', # Healthcare
        'XLE', # Energy
        'XLC', # Comm Services
        'XLY', # Consumer Discretionary
        'XLP', # Consumer Staples
        'XLI', # Industrials
        'XLB', # Materials
        'XLU', # Utilities
        'XLRE', # Real Estate
        
        # --- Innovation / Thematic ---
        'ARKK', # Innovation
        'SMH', 'SOXX', # Semiconductors
        'TAN', # Solar
        'IBB', # Biotech
        
        # --- Bonds ---
        'TLT', # 20+ Year Treasury
        'IEF', # 7-10 Year Treasury
        'SHY', # 1-3 Year Treasury
        'LQD', # Investment Grade Corp
        'HYG', 'JNK', # High Yield (Junk)
        'BND', 'AGG', # Total Bond Market
        
        # --- Commodities & Alt ---
        'GLD', 'IAU', # Gold
        'SLV', # Silver
        'USO', # Oil
        'DBC', # Commodity Index
        'BITO', # Bitcoin Strategy
        
        # --- Volatility ---
        'UVXY', 'VIXY' # Short-term VIX (Careful!)
    ]

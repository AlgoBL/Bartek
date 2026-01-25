
import pandas as pd
import streamlit as st

@st.cache_data
def get_sp500_tickers():
    """
    Pobiera aktualną listę tickerów S&P 500 z Wikipedii.
    W przypadku błędu zwraca awaryjną listę TOP 50.
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        # Yahoo Finance używa '-' zamiast '.' (np. BRK-B zamiast BRK.B)
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        st.warning("Nie udało się pobrać pełnej listy S&P 500. Używam listy zapasowej.")
        # Top 20 liquid stocks (fallback)
        return ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ', 'JPM', 'XOM', 'V', 'PG', 'MA', 'HD', 'CVX', 'ABBV', 'MRK', 'PEP']

@st.cache_data
def get_euro_etfs():
    """
    Zwraca listę popularnych ETF-ów europejskich (UCITS).
    (Lista przykładowa, w pełnej wersji powinna być dynamiczna lub szersza)
    """
    # Przykładowe popularne ETFy na rynki UE
    return [
        'EWG', # Germany
        'EWQ', # France
        'EWI', # Italy
        'EWU', # UK
        'EWP', # Spain
        'EWL', # Switzerland
        'IEUR', # Europe Core
        'VGK', # Vanguard Europe
        'FEZ', # Euro Stoxx 50
        'HEDJ' # Europe Hedged
    ]

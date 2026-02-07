
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import skew, kurtosis
from modules.scanner import calculate_convecity_metrics, score_asset

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bulk_data_cached(tickers, period="2y"):
    """
    Pobiera dane dla wielu tickerów naraz z cachem.
    """
    try:
        # yfinance automatycznie używa wątków przy wielu tickerach
        data = yf.download(tickers, period=period, group_by='ticker', progress=False, auto_adjust=True)
        return data
    except Exception as e:
        return pd.DataFrame()

class ScannerEngine:
    def __init__(self):
        pass

    def fetch_bulk_data(self, tickers, period="2y"):
        return fetch_bulk_data_cached(tickers, period)

    def scan_markets(self, tickers, progress_bar=None):
        """
        Skanuje listę tickerów i oblicza metryki wypukłości.
        """
        results = []
        data = self.fetch_bulk_data(tickers)
        
        total = len(tickers)
        
        # Iteracja po tickerach
        # Data structure: Columns are MultiIndex (Ticker, PriceType) or just (PriceType) if one ticker
        # Przy group_by='ticker', top level to Ticker
        
        valid_tickers = [t for t in tickers if t in data.columns.levels[0]]
        
        for i, ticker in enumerate(valid_tickers):
            try:
                df = data[ticker]
                # Sprawdź czy mamy kolumnę Close
                if 'Close' not in df.columns:
                    continue
                    
                price_series = df['Close'].dropna()
                metrics = calculate_convecity_metrics(ticker, price_series)
                
                if metrics:
                    score = score_asset(metrics)
                    metrics['Score'] = score
                    results.append(metrics)
            except Exception as e:
                continue
                
            if progress_bar and i % 10 == 0:
                progress_bar.progress((i + 1) / len(valid_tickers), f"Analizowanie: {ticker}")
                
        return pd.DataFrame(results)

    def select_best_candidates(self, candidates_df, max_count=10):
        """
        Wybiera najlepszych kandydatów na podstawie wyniku matematycznego (Score).
        Zastępuje dawną metodę AI "select_with_gemini".
        """
        if candidates_df.empty:
            return []
            
        # Sortowanie po wyniku (Score)
        # Score promuje: Skewness > 0, Fat Tails (Low Alpha), High Return potential
        top_candidates = candidates_df.sort_values('Score', ascending=False)
        
        # Zwracamy top N tickerów
        return top_candidates.head(max_count)['Ticker'].tolist()

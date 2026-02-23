
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import skew, kurtosis
from modules.scanner import calculate_convecity_metrics, score_asset

from modules.ai.oracle import TheOracle
from modules.ai.agents import LocalEconomist, LocalGeopolitics, LocalCIO
from modules.ai.screener import FundamentalScreener

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

    def run_v5_autonomous_scan(self, horizon_years: int, progress_callback=None):
        """
        Główny silnik orkiestracji V5: Oracle -> Agenci -> Screener -> EVT
        """
        if progress_callback: progress_callback(0.1, "Wyrocznia: Pobieranie danych makro z rynków...")
        
        # 1. Oracle
        oracle = TheOracle()
        macro_snap = oracle.get_macro_snapshot()
        news = oracle.get_latest_news_headlines(30)
        
        if progress_callback: progress_callback(0.3, "Agenci AI: Analiza geopolityki i ekonomii...")
        
        # 2. Agents (Local, API-Free)
        economist = LocalEconomist()
        geo = LocalGeopolitics()
        cio = LocalCIO()
        
        econ_report = economist.analyze_macro(macro_snap)
        geo_report = geo.analyze_news(news)
        cio_thesis = cio.synthesize_thesis(econ_report, geo_report, horizon_years)
        
        if progress_callback: progress_callback(0.5, "Mikro-Skaner: Filtracja najpłynniejszych globalnych aktywów...")
        
        # 3. Screener
        screener = FundamentalScreener(min_volume=500000)
        # Pobieramy szeroką listę z focusu
        raw_universe = screener.fetch_broad_universe("global")
        # Zostawmy top 50 dla optymalizacji czasowej aby nie blokować UI 5 minut (normalnie 200+)
        liquid_assets = screener.filter_liquid_assets(raw_universe[:50]) 
        
        if progress_callback: progress_callback(0.7, f"EVT Engine: Ocenianie {len(liquid_assets)} wyselekcjonowanych aktywów...")
        
        # 4. EVT Math Scan
        metrics_df = self.scan_markets(liquid_assets)
        top_picks = self.select_best_candidates(metrics_df, max_count=10)
        
        if progress_callback: progress_callback(1.0, "Ukończono generowanie portfela!")
        
        return {
            "cio_thesis": cio_thesis,
            "econ_report": econ_report,
            "geo_report": geo_report,
            "scanned_universe_size": len(liquid_assets),
            "top_picks": top_picks,
            "metrics_df": metrics_df
        }

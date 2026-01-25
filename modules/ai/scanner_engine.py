
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
import streamlit as st
from scipy.stats import skew, kurtosis
from modules.scanner import calculate_convecity_metrics, score_asset

class ScannerEngine:
    def __init__(self):
        pass

    def fetch_bulk_data(self, tickers, period="2y"):
        """
        Pobiera dane dla wielu tickerów naraz.
        """
        try:
            # yfinance automatycznie używa wątków przy wielu tickerach
            data = yf.download(tickers, period=period, group_by='ticker', progress=True, auto_adjust=True)
            return data
        except Exception as e:
            st.error(f"Error fetching bulk data: {e}")
            return pd.DataFrame()

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

    def select_with_gemini(self, candidates_df, api_key, max_count=10):
        """
        Używa Gemini Pro do wybrania najlepszych aktywów z listy kandydatów.
        """
        if not api_key:
            return candidates_df.head(max_count)['Ticker'].tolist()
            
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Przygotuj dane dla AI (Top 30 dla kontekstu)
            candidates_subset = candidates_df.sort_values('Score', ascending=False).head(30)
            
            # Konwersja do stringa
            data_str = candidates_subset[['Ticker', 'Annual Return', 'Volatility', 'Skewness', 'Kurtosis', 'Score']].to_markdown(index=False)
            
            prompt = f"""
            Jesteś ekspertem od strategii inwestycyjnych typu Barbell (Nassim Taleb).
            Masz poniżej listę kandydatów do "ryzykownej" części portfela Barbell (High Convexity).
            Te aktywa mają wysoką zmienność, dodatnią skośność i potencjał do dużych zysków (grube ogony).
            
            Twoim zadaniem jest wybrać {max_count} najlepszych tickerów z tej listy, aby stworzyć zdywersyfikowany koszyk.
            Unikaj wybierania zbyt wielu spółek z tego samego sektora, jeśli to możliwe.
            
            Dane kandydatów:
            {data_str}
            
            Zwróć TYLKO listę tickerów oddzielonych przecinkami, bez zbędnego tekstu.
            Przykład: AAPL, MSFT, NVDA
            """
            
            response = model.generate_content(prompt)
            text = response.text.strip()
            
            # Parsowanie odpowiedzi
            selected_tickers = [t.strip() for t in text.replace('\n', '').split(',')]
            
            # Walidacja (czy tickery są na liście)
            valid_selected = [t for t in selected_tickers if t in candidates_df['Ticker'].values]
            
            # Jeśli AI zwróciło zły format, fallback do top score
            if len(valid_selected) == 0:
                print("Gemini returned invalid format. Fallback to Top Score.")
                return candidates_subset.head(max_count)['Ticker'].tolist()
                
            return valid_selected[:max_count]
            
        except Exception as e:
            st.error(f"Gemini Error: {e}")
            return candidates_df.sort_values('Score', ascending=False).head(max_count)['Ticker'].tolist()


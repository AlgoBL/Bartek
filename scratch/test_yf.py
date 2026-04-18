import sys
import os
import pandas as pd

# Dodaj główny katalog projektu do ścieżki
sys.path.append(r"c:\Users\bartl\OneDrive\KOd\Bartek")

from modules.ai.data_loader import load_data

tickers = ['SPY', 'QQQ', 'NVDA', 'BTC-USD']
print(f"--- Test load_data dla: {tickers} ---")

# Testujemy z krótkim okresem
data = load_data(tickers, period="1mo")

if data.empty:
    print("BŁĄD: load_data zwróciło pusty DataFrame!")
else:
    print(f"SUKCES: Załadowano {len(data)} wierszy.")
    print("Kolumny:", list(data.columns))
    print("Pierwsze 5 wierszy:")
    print(data.head())
    print("NaN count per column:")
    print(data.isna().sum())
    
    # Sprawdzenie czy mamy weekendy (BTC-USD powinien wypełnić weekends, a inne ffill)
    weekends = data.index.weekday >= 5
    if weekends.any():
         print(f"Znaleziono {weekends.sum()} dni weekendowych. Alignment działa.")
    else:
         print("Nie znaleziono weekendów. Sprawdź czy pobrano wystarczająco dużo danych lub czy Dropna usunął je.")


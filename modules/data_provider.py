import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import asyncio
from typing import List, Union
from modules.logger import setup_logger

logger = setup_logger(__name__)

def translate_ticker_for_stooq(ticker: str) -> str:
    """Konwertuje ticker Yahoo Finance na format zgodny ze Stooq.pl."""
    t = ticker.upper()
    
    # Indeksy
    if t == "^GSPC": return "^SPX"
    if t == "^NDX": return "^NDQ"
    if t == "^DJI": return "^DJI"
    if t.startswith("^"): return t
    
    # Giełda Papierów Wartościowych w Warszawie (GPW)
    if t.endswith(".WA"):
        return t.replace(".WA", ".PL")
        
    # Krypto
    if "-USD" in t:
        return t.replace("-USD", ".V") # np. BTC.V na stooq (czasami)
        
    # Akcje zagraniczne (US) - dodajemy .US jeśli nie ma sufiksu
    if "." not in t:
        return f"{t}.US"
        
    return t

import threading
_YF_LOCK = threading.Lock()

def _fetch_from_yfinance_sync(tickers: List[str], start: str = None, end: str = None, period: str = None, auto_adjust: bool = True) -> pd.DataFrame:
    # Zabezpieczenie na wyścigi wątków (thread-safety) we wbudowanym module cacheującym YFinance
    if len(tickers) == 1:
        # History jest bezpieczne wielowątkowo
        tkr = yf.Ticker(tickers[0])
        kw = {"auto_adjust": auto_adjust}
        if start and end:
            kw["start"] = start
            kw["end"] = end
        elif period:
            kw["period"] = period
        else:
            kw["period"] = "1y"
        return tkr.history(**kw)
        
    kwargs = {"progress": False, "auto_adjust": auto_adjust, "threads": False}
    if start and end:
        kwargs["start"] = start
        kwargs["end"] = end
    elif period:
        kwargs["period"] = period
    else:
        kwargs["period"] = "1y" # domyślnie 1 rok

    # Wymuszenie sekwencyjnego logowania w yfinance globals
    with _YF_LOCK:
        data = yf.download(tickers, **kwargs)
    return data

def _fetch_from_stooq_sync(tickers: List[str], start: str = None, end: str = None) -> pd.DataFrame:
    stooq_tickers = [translate_ticker_for_stooq(t) for t in tickers]
    
    # pandas_datareader data
    data = web.DataReader(stooq_tickers, 'stooq', start=start, end=end)
    
    # Stooq zwraca (Attributes, Symbols). Zamieniamy Symbols z powrotem na Yahoo
    if isinstance(data.columns, pd.MultiIndex):
        # Translate symbols back
        yahoo_stooq_map = {translate_ticker_for_stooq(t): t for t in tickers}
        
        # Jeśli mamy wiele tickerów: (Attribute, Symbol) np. ('Close', 'AAPL.US')
        new_columns = []
        for attr, sym in data.columns:
            original_sym = yahoo_stooq_map.get(sym, sym)
            new_columns.append((attr, original_sym))
            
        data.columns = pd.MultiIndex.from_tuples(new_columns)
        
        # Do not swap levels here to match yfinance default format: (Price, Ticker)
        data.sort_index(axis=1, level=0, inplace=True)
    else:
        # Jeden ticker, struktura:  Open High Low Close Volume
        pass
        
    # Posortuj chronologicznie, gdyż Stooq zwraca z reguły "od najnowszego"
    data = data.sort_index(ascending=True)
    return data

def fetch_data(tickers: Union[str, List[str]], start: str = None, end: str = None, period: str = None, auto_adjust: bool = True) -> pd.DataFrame:
    """
    Kluczowy system pobierania danych. Próbuje pobrać poprzez YFinance, 
    a następnie w wypadku błędu za pośrednictwem Stooq.pl.
    Zwraca ujednolicony pd.DataFrame.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
        
    tickers = list(tickers)
    
    # Wyjątek: pobieranie intraday 1m, 5m etc. jest wspierane natywnie tylko z YFinance 
    # i nie działa łatwo ze stooq na pandas_datareader,
    # ale nasze aplikacje głównie polegają na interwałach dziennych.
    
    try:
        logger.info(f"Pobieranie danych rynkowych przez yfinance dla: {tickers[:3]}...")
        data = _fetch_from_yfinance_sync(tickers, start, end, period, auto_adjust)
        if not data.empty:
            return data
        else:
            logger.warning("yfinance zwróciło puste dane. Przełączanie na stooq...")
    except Exception as e:
        logger.warning(f"Błąd yfinance ({e}). Przełączanie na stooq...")

    # Fallback to Stooq
    try:
        # Dla stooq potrzebne są daty start i end, brak parametru period (np '2y') prosto podawanego
        if not start:
            if period and period.endswith("y"):
                years = int(period.replace("y", ""))
                start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
            elif period and period.endswith("mo"):
                months = int(period.replace("mo", ""))
                start = (pd.Timestamp.today() - pd.DateOffset(months=months)).strftime('%Y-%m-%d')
            else:
                start = "2000-01-01"

        logger.info(f"Pobieranie stooq dla: {tickers[:3]}...")
        data = _fetch_from_stooq_sync(tickers, start, end)
        return data
        
    except Exception as e:
        logger.error(f"Krytyczny błąd pobierania danych awaryjnych (stooq): {e}")
        return pd.DataFrame()


async def fetch_ticker_async(ticker: str, period: str = "5d") -> tuple[str, float | None]:
    """Asynchroniczne pobieranie ceny zamknięcia. Yahoo -> StooqFallback"""
    try:
        data = await asyncio.to_thread(_fetch_from_yfinance_sync, [ticker], period=period)
        if not data.empty:
            # yfinance shape handle  
            if isinstance(data.columns, pd.MultiIndex):
                return ticker, round(float(data[ticker]["Close"].dropna().iloc[-1]), 3)
            return ticker, round(float(data["Close"].dropna().iloc[-1]), 3)
            
    except Exception as e:
        logger.debug(f"Brak danych Yahoo dla {ticker} (async): {e}")
        
    # Stooq fallback
    try:
        logger.debug(f"Pobieranie stooq (async) dla {ticker}")
        start = (pd.Timestamp.today() - pd.DateOffset(days=10)).strftime('%Y-%m-%d')
        data = await asyncio.to_thread(_fetch_from_stooq_sync, [ticker], start=start)
        if not data.empty:
            # dla 1 tickera pandas_datareader zwraca kolumny 'Close', 'High'
            val = round(float(data["Close"].dropna().iloc[-1]), 3)
            return ticker, val
    except Exception as e:
        logger.debug(f"Brak danych STOOQ dla {ticker} (async): {e}")
        
    return ticker, None

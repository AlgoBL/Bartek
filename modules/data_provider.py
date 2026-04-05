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
        if start:
            kw["start"] = start
            if end:
                kw["end"] = end
        elif period:
            kw["period"] = period
        else:
            kw["period"] = "1y"
        return tkr.history(**kw)
        
    kwargs = {"progress": False, "auto_adjust": auto_adjust, "threads": True}
    if start:
        kwargs["start"] = start
        if end:
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
    
    try:
        # pandas_datareader data
        data = web.DataReader(stooq_tickers, 'stooq', start=start, end=end)
        
        if data is None or data.empty:
            return pd.DataFrame()
            
        # Stooq zwraca (Attributes, Symbols). Zamieniamy Symbols z powrotem na Yahoo
        if isinstance(data.columns, pd.MultiIndex):
            # Translate symbols back
            yahoo_stooq_map = {translate_ticker_for_stooq(t): t for t in tickers}
            
            # Jeśli mamy wiele tickerów: (Attribute, Symbol) np. ('Close', 'AAPL.US')
            new_columns = []
            valid_cols = False
            for attr, sym in data.columns:
                original_sym = yahoo_stooq_map.get(sym, sym)
                new_columns.append((attr, original_sym))
                if original_sym in tickers:
                    valid_cols = True
                    
            if not valid_cols:
                return pd.DataFrame()

            data.columns = pd.MultiIndex.from_tuples(new_columns)
            
            # Do not swap levels here to match yfinance default format: (Price, Ticker)
            data.sort_index(axis=1, level=0, inplace=True)
        else:
            # Jeden ticker, struktura:  Open High Low Close Volume
            pass
            
        # Posortuj chronologicznie, gdyż Stooq zwraca z reguły "od najnowszego"
        data = data.sort_index(ascending=True)
        return data
        
    except Exception as e:
        from modules.logger import setup_logger
        _log = setup_logger(__name__)
        _log.warning(f"Błąd we wbudowanym czytniku Stooq: {e}")
        return pd.DataFrame()

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


async def fetch_ticker_async(ticker: str, period: str = "5d") -> tuple[str, float | None, float | None]:
    """Asynchroniczne pobieranie ceny zamknięcia. Yahoo -> StooqFallback"""
    try:
        data = await asyncio.to_thread(_fetch_from_yfinance_sync, [ticker], period=period)
        if not data.empty:
            # yfinance shape handle  
            if isinstance(data.columns, pd.MultiIndex):
                series = data["Close"][ticker].dropna()
            else:
                series = data["Close"].dropna()
                
            val = round(float(series.iloc[-1]), 3)
            pct = 0.0
            if len(series) > 1:
                pct = round(float((series.iloc[-1] / series.iloc[-2]) - 1) * 100, 2)
            return ticker, val, pct
            
    except Exception as e:
        logger.debug(f"Brak danych Yahoo dla {ticker} (async): {e}")
        
    # Stooq fallback
    try:
        logger.debug(f"Pobieranie stooq (async) dla {ticker}")
        start = (pd.Timestamp.today() - pd.DateOffset(days=10)).strftime('%Y-%m-%d')
        data = await asyncio.to_thread(_fetch_from_stooq_sync, [ticker], start=start)
        if not data.empty:
            # dla 1 tickera pandas_datareader zwraca kolumny 'Close', 'High'
            series = data["Close"].dropna()
            val = round(float(series.iloc[-1]), 3)
            pct = 0.0
            if len(series) > 1:
                pct = round(float((series.iloc[-1] / series.iloc[-2]) - 1) * 100, 2)
            return ticker, val, pct
    except Exception as e:
        logger.debug(f"Brak danych STOOQ dla {ticker} (async): {e}")
        
    return ticker, None, None

def fetch_usdpln_data(start: str = None, end: str = None, period: str = None) -> pd.Series:
    """Pobiera historyczny kurs USD/PLN z Yahoo Finance."""
    try:
        ticker = "USDPLN=X"
        data = fetch_data([ticker], start=start, end=end, period=period)
        if data.empty:
            return pd.Series()
        
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close'][ticker]
        else:
            return data['Close']
    except Exception as e:
        logger.error(f"Błąd pobierania USD/PLN: {e}")
        return pd.Series()

def fetch_currency_adjusted_data(tickers: List[str], start: str = None, end: str = None, period: str = None) -> pd.DataFrame:
    """
    Pobiera dane dla tickerów oraz kurs USD/PLN, a następnie przelicza ceny na PLN.
    Zakłada, że tickery bez sufiksu .PL lub .WA są denominowane w USD.
    """
    data = fetch_data(tickers, start=start, end=end, period=period)
    if data.empty:
        return data
        
    # Pobierz USDPLN dla tego samego okresu
    usdpln = fetch_usdpln_data(start=start, end=end, period=period)
    if usdpln.empty:
        logger.warning("Nie udało się pobrać kursu USD/PLN. Zwracam dane bez przeliczenia.")
        return data
        
    # Ujednolicenie indeksów (częsty problem z dniami wolnymi w różnych krajach)
    combined = pd.concat([data, usdpln.rename("USDPLN")], axis=1).ffill().dropna()
    
    # Kopia fragmentu odpowiadającego wyjściowym tickerom
    if isinstance(data.columns, pd.MultiIndex):
        adjusted_data = combined[data.columns.levels[0]].copy()
        usdpln_aligned = combined["USDPLN"]
        
        for t in tickers:
            # Zakładamy że wszystko co nie jest .PL/.WA jest w USD
            if not (t.endswith(".PL") or t.endswith(".WA")):
                for price_col in data.columns.levels[0]:
                    if (price_col, t) in data.columns:
                        adjusted_data.loc[:, (price_col, t)] *= usdpln_aligned
    else:
        adjusted_data = combined[data.columns].copy()
        usdpln_aligned = combined["USDPLN"]
        t = tickers[0]
        if not (t.endswith(".PL") or t.endswith(".WA")):
            for col in adjusted_data.columns:
                if col in ['Open', 'High', 'Low', 'Close']:
                    adjusted_data[col] *= usdpln_aligned
                    
    return adjusted_data

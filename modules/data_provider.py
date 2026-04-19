import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import asyncio
import streamlit as st
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
        hist = tkr.history(**kw)
        if not hist.empty:
            # Force MultiIndex (Attribute, Ticker) Nawet dla pojedynczego tickera
            hist.columns = pd.MultiIndex.from_product([hist.columns, [tickers[0]]])
        return hist
        
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

@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_data_cached(tickers_tuple: tuple, start: str, end: str, period: str, auto_adjust: bool) -> pd.DataFrame:
    tickers = list(tickers_tuple)
    
    # Transparent ISIN resolution
    from modules.isin_resolver import ISINResolver
    mapped_tickers = []
    isin_map = {}
    for t in tickers:
        mapped = ISINResolver.resolve(t)
        mapped_tickers.append(mapped)
        if mapped != t.strip().upper():
            isin_map[mapped] = t.strip().upper()
    
    try:
        logger.info(f"Pobieranie danych rynkowych (Yahoo) | Tickers: {mapped_tickers} | Period: {period} | Start: {start} | End: {end}")
        data = _fetch_from_yfinance_sync(mapped_tickers, start, end, period, auto_adjust)
        
        if not data.empty:
            # Sprawdź czy wszystkie tickery mają dane
            if isinstance(data.columns, pd.MultiIndex):
                # Dla MultiIndex (np. Adj Close, SPY) bierzemy level 1 (Tickers)
                fetched_tickers = data.columns.get_level_values(1).unique() if data.columns.nlevels > 1 else data.columns
            else:
                fetched_tickers = data.columns.unique()
            
            missing = [t for t in mapped_tickers if t not in fetched_tickers]
            if missing:
                logger.warning(f"yfinance nie zwróciło żadnych kolumn dla: {missing}")
            
            logger.info(f"Pobrano pomyślnie {len(data)} wierszy.")
            
            # Ensure index is standardized (no timezone, sorted)
            if data.index.tz is not None:
                data.index = data.index.tz_convert(None)
            data.sort_index(inplace=True)

            if isin_map and isinstance(data.columns, pd.MultiIndex):
                new_cols = []
                for attr, sym in data.columns:
                    new_cols.append((attr, isin_map.get(sym, sym)))
                data.columns = pd.MultiIndex.from_tuples(new_cols)
                data.sort_index(axis=1, level=0, inplace=True)
            return data
        else:
            logger.warning(f"yfinance zwrocilo puste dane dla {mapped_tickers}. Sprawdz czy symbole sa poprawne.")
            logger.info("Przelaczanie na alternatywne zrodla (Stooq)...")

    except Exception as e:
        logger.warning(f"Błąd yfinance ({e}). Przełączanie na stooq...")

    # Fallback to Stooq
    try:
        if not start:
            if period and period.endswith("y"):
                years = int(period.replace("y", ""))
                start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
            elif period and period.endswith("mo"):
                months = int(period.replace("mo", ""))
                start = (pd.Timestamp.today() - pd.DateOffset(months=months)).strftime('%Y-%m-%d')
            else:
                start = "2000-01-01"

        logger.info(f"Pobieranie stooq dla: {mapped_tickers[:3]}...")
        data = _fetch_from_stooq_sync(mapped_tickers, start, end)
        
        if isin_map and not data.empty and isinstance(data.columns, pd.MultiIndex):
            new_cols = []
            for attr, sym in data.columns:
                new_cols.append((attr, isin_map.get(sym, sym)))
            data.columns = pd.MultiIndex.from_tuples(new_cols)
            data.sort_index(axis=1, level=0, inplace=True)
            
        return data
    except Exception as e:
        logger.error(f"Krytyczny błąd pobierania danych (Yahoo+Stooq ostatecznie zawiodły): {e}")
        return pd.DataFrame()


def fetch_data(tickers: Union[str, List[str]], start: str = None, end: str = None, period: str = None, auto_adjust: bool = True) -> pd.DataFrame:
    """
    Kluczowy system pobierania danych. Próbuje pobrać poprzez YFinance, 
    a następnie w wypadku błędu za pośrednictwem Stooq.pl.
    Zwraca ujednolicony pd.DataFrame.
    """
    if isinstance(tickers, str):
        tickers_tuple = (tickers,)
    else:
        tickers_tuple = tuple(tickers)
        
    return _fetch_data_cached(tickers_tuple, start, end, period, auto_adjust)


async def fetch_ticker_async(ticker: str, period: str = "5d") -> tuple[str, float | None, float | None]:
    """Asynchroniczne pobieranie ceny zamknięcia. Yahoo -> StooqFallback"""
    from modules.isin_resolver import ISINResolver
    mapped_ticker = ISINResolver.resolve(ticker)
    
    try:
        data = await asyncio.to_thread(_fetch_from_yfinance_sync, [mapped_ticker], period=period)
        if not data.empty:
            # yfinance shape handle  
            if isinstance(data.columns, pd.MultiIndex):
                series = data["Close"][mapped_ticker].dropna()
            else:
                series = data["Close"].dropna()
                
            val = round(float(series.iloc[-1]), 3)
            pct = 0.0
            if len(series) > 1:
                pct = round(float((series.iloc[-1] / series.iloc[-2]) - 1) * 100, 2)
            return ticker, val, pct
            
    except Exception as e:
        logger.debug(f"Brak danych Yahoo dla {ticker} ({mapped_ticker}) (async): {e}")
        
    # Stooq fallback
    try:
        logger.debug(f"Pobieranie stooq (async) dla {ticker}")
        start = (pd.Timestamp.today() - pd.DateOffset(days=10)).strftime('%Y-%m-%d')
        data = await asyncio.to_thread(_fetch_from_stooq_sync, [mapped_ticker], start=start)
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
        
    # Ujednolicenie indeksów (Timezone normalization to naive)
    if data.index.tz is not None:
        data.index = data.index.tz_convert(None)
    if usdpln.index.tz is not None:
        usdpln.index = usdpln.index.tz_convert(None)
    
    # Przemianowanie USDPLN tak, aby pasowało do struktury MultiIndex (jeśli dotyczy)
    if isinstance(data.columns, pd.MultiIndex):
        usdpln_col = ("USDPLN", "")
        combined = pd.concat([data, usdpln.rename(usdpln_col)], axis=1)
    else:
        usdpln_col = "USDPLN"
        combined = pd.concat([data, usdpln.rename(usdpln_col)], axis=1)

    combined = combined.ffill()
    
    # Drop rows where we have no FX data (cannot adjust)
    combined = combined.dropna(subset=[usdpln_col])
    
    if combined.empty:
        logger.warning("Brak wspólnych danych po połączeniu z kursem USD/PLN. Zwracam oryginalne dane.")
        return data

    # Kopia fragmentu odpowiadającego wyjściowym tickerom
    if isinstance(data.columns, pd.MultiIndex):
        adjusted_data = combined[data.columns].copy()
        usdpln_aligned = combined[usdpln_col] # Series
        
        # Valid labels handle
        level0 = data.columns.get_level_values(0).unique()
        
        for t in tickers:
            # Zakładamy, że wszystko co nie jest .PL/.WA jest w USD
            if not (t.endswith(".PL") or t.endswith(".WA")):
                for price_col in level0:
                    if (price_col, t) in data.columns:
                        adjusted_data.loc[:, (price_col, t)] *= usdpln_aligned
    else:
        adjusted_data = combined[data.columns].copy()
        usdpln_aligned = combined[usdpln_col]
        t = tickers[0]
        if not (t.endswith(".PL") or t.endswith(".WA")):
            for col in adjusted_data.columns:
                if col in ['Open', 'High', 'Low', 'Close']:
                    adjusted_data[col] *= usdpln_aligned
                    
    return adjusted_data

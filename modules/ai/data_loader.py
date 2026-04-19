
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
    
    # --- 1. Automatyczne czyszczenie tickerów (Sanitization) ---
    if isinstance(tickers, str):
        tickers = [x.strip().upper() for x in tickers.split(",") if x.strip()]
    else:
        tickers = [str(x).strip().upper() for x in tickers if str(x).strip()]
    
    # Usuwanie duplikatów przy zachowaniu kolejności
    tickers = list(dict.fromkeys(tickers))
    
    if not tickers:
        logger.warning("Przekazano pustą listę tickerów.")
        return pd.DataFrame()
        
    try:
        if base_currency == "PLN":
            data = fetch_currency_adjusted_data(tickers, start=start_date, end=end_date, period=period)
        else:
            data = fetch_data(tickers, start=start_date, end=end_date, period=period, auto_adjust=False)
        
        if data.empty:
            logger.warning(f"Zaladowano pusty DataFrame dla tickerow: {tickers}")
            return pd.DataFrame()

        # ─── Obsługa struktury danych ──────────────────────────────────────────
        # Normalizacja indeksu na naive (usuwa UTC jezeli jest)
        if data.index.tz is not None:
             data.index = data.index.tz_convert(None)

        # Wybór odpowiedniej kolumny cenowej (standard: Adj Close > Close)
        # Dzięki poprawce w data_provider, zazwyczaj mamy MultiIndex (Attribute, Ticker)
        if isinstance(data.columns, pd.MultiIndex):
            available_attrs = data.columns.get_level_values(0).unique()
            if 'Adj Close' in available_attrs:
                data = data['Adj Close']
            elif 'Close' in available_attrs:
                data = data['Close']
            else:
                logger.error(f"Nie znaleziono kolumny 'Close'/'Adj Close' w MultiIndex. Dostępne: {list(available_attrs)}")
                # Ostatnia szansa: jeśli jest tylko jeden atrybut w L0, spróbuj go wziąć
                if len(available_attrs) == 1:
                    data = data[available_attrs[0]]
                else:
                    return pd.DataFrame()
        else:
             # Fallback dla nieoczekiwanego SingleIndex
             if 'Close' in data.columns:
                 data = data['Close'].to_frame()
                 data.columns = tickers
             else:
                logger.error(f"Nieobsługiwana struktura danych (SingleIndex) dla {tickers}")
                return pd.DataFrame()

        # ─── Walidacja i Czyszczenie Tickerów ─────────
        if isinstance(data, pd.Series):
             data = data.to_frame()

        found_tickers = list(data.columns)
        missing_tickers = [t for t in tickers if t not in found_tickers]
        
        if missing_tickers:
            st.warning(f"⚠️ Nie znaleziono danych dla: {missing_tickers}. Sprawdź ich poprawność.")
            logger.warning(f"Brak danych w pobranym obiekcie dla: {missing_tickers}")

        # Usuwamy kolumny, które są same w sobie całkowicie puste
        empty_cols = [col for col in data.columns if data[col].dropna().empty]
        if empty_cols:
            st.warning(f"❌ Aktywa bez historii: {empty_cols}. Pominięto.")
            data = data.drop(columns=empty_cols)

        if data.empty:
            logger.error(f"DataFrame stał się pusty po usunięciu pustych kolumn dla {tickers}")
            return pd.DataFrame()

        # --- 3. Robust Alignment (Alignment rynków akcji i krypto) ---
        # Używamy forward-fill aby uzupełnić luki (np. weekendy dla akcji gdy krypto ma dane)
        final_data = data.ffill()
        
        # Zamiast agresywnego dropna() na całych wierszach, sprawdzamy czy mamy wystarczająco dużo danych
        # Obcinamy tylko te dni na początku, gdzie ŻADEN asset nie ma danych
        final_data = final_data.dropna(how='all')
        
        # Jeśli nadal są NaNs na początku dla niektórych assetów (bo np. IPO było później), 
        # zostawiamy je jako NaNs lub bfill() jeśli to krótkie luki.
        # Większość algorytmów (HRP/Kelly) poradzi sobie z NaNs na początku przez dynamiczne okna.
        final_data = final_data.bfill(limit=5) # wypełnij tylko krótkie luki na początku

        if final_data.empty:
            logger.warning(f"Brak danych po procesie czyszczenia dla {tickers}")
            return pd.DataFrame()

        logger.info(f"Pomyślnie załadowano dane: {list(final_data.columns)} ({len(final_data)} dni)")
        return final_data
        
    except Exception as e:
        logger.error(f"Błąd krytyczny w load_data dla {tickers}: {str(e)}", exc_info=True)
        return pd.DataFrame()


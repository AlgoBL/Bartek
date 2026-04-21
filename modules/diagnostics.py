import time
import os
import json
import yfinance as yf
import pandas_datareader.data as web
from functools import wraps
from modules.logger import setup_logger, AWARIE_JSON_PATH

log = setup_logger(__name__)

def get_recent_errors(limit=50):
    """Odczytuje najnowsze błędy zapisane w systemie."""
    if not os.path.exists(AWARIE_JSON_PATH):
        return []
    try:
        with open(AWARIE_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return (data[:limit] if isinstance(data, list) else [])
    except Exception:
        return []

def mark_error_resolved(error_id):
    """Zaznacza błąd o danym ID jako rozwiązany."""
    if not os.path.exists(AWARIE_JSON_PATH):
        return
    try:
        with open(AWARIE_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            if d.get("id") == error_id:
                d["resolved"] = True
                d["resolved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(AWARIE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Nie udało się zaktualizować statusu błędu w kwarantannie: {e}")

def run_health_check():
    """Wysyła "żółte karteczki" do serwerów zewnętrznych by zbadać gdzie leży wina przyczyna zerwania połączeń."""
    results = {}
    
    # 1. Yahoo Ping
    start = time.time()
    try:
        df = yf.download("SPY", period="1d", progress=False)
        if df.empty: raise ValueError("Pusty pakiet danych od Yahoo")
        results["yahoo"] = {"status": "ok", "time_ms": int((time.time() - start) * 1000)}
    except Exception as e:
        results["yahoo"] = {"status": "fail", "time_ms": int((time.time() - start) * 1000), "error": str(e)}
        log.warning(f"Health Check: Brak połączenia z Yahoo Finance: {e}")

    # Usunięto wadliwy radar Stooq. Pozostawiono tylko główny Yahoo.
        
    return results

def safe_background_task(func):
    """
    Dekorator opakowujący funkcje asynchroniczne i wielowątkowe.
    Jeśli funkcja rzuci błąd, zostanie on cicho przechwycony za kulisami by nie psuć widoku w przeglądarce.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            l = setup_logger(func.__module__)
            l.error(f"Złamany Wątek w Tle [{func.__name__}]: {e}", exc_info=True)
            return None
    return wrapper

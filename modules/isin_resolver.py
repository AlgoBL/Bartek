import re
import json
import os
import requests
from modules.logger import setup_logger

logger = setup_logger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "isin_cache.json")

class ISINResolver:
    @staticmethod
    def is_isin(identifier: str) -> bool:
        """Sprawdza, czy ciąg znaków pasuje do uniwersalnego formatu ISIN."""
        if not identifier or not isinstance(identifier, str):
            return False
        identifier = identifier.strip().upper()
        # 2 litery kodu kraju + 9 znaków alfanumerycznych + 1 cyfra kontrolna
        pattern = r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$"
        return bool(re.match(pattern, identifier))

    @classmethod
    def load_cache(cls) -> dict:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Błąd odczytu cache ISIN: {e}")
        return {}

    @classmethod
    def save_cache(cls, cache: dict):
        os.makedirs(CACHE_DIR, exist_ok=True)
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=4)
        except Exception as e:
            logger.warning(f"Błąd zapisu cache ISIN: {e}")

    @classmethod
    def search_isin(cls, isin: str) -> dict:
        """
        Wyszukuje ISIN w Yahoo Finance by znaleźć powiązany Ticker.
        Zwraca pełen słownik wyników dla UI lub wyszukiwarki.
        """
        isin = isin.strip().upper()
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={isin}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            quotes = data.get("quotes", [])
            
            # Priorytetyzujemy wyniki, najchętniej ETF
            best_match = None
            for q in quotes:
                if q.get("quoteType", "").upper() in ["ETF", "MUTUALFUND", "EQUITY"]:
                    best_match = q
                    break
                    
            if best_match is None and quotes:
                best_match = quotes[0]
                
            return best_match if best_match else {}
            
        except Exception as e:
            logger.error(f"Błąd wyszukiwania ISIN {isin} w Yahoo Finance: {e}")
            return {}

    @classmethod
    def resolve(cls, identifier: str) -> str:
        """
        Główna metoda dla warstwy Providera Danych. Zwraca ticker Yahoo Finance
        lub wyjściowy ticker (nie będący ISIN), dzięki czemu aplikacja nie widzi różnicy.
        """
        if not cls.is_isin(identifier):
            return identifier.strip().upper()
            
        isin = identifier.strip().upper()
        cache = cls.load_cache()
        
        if isin in cache and "symbol" in cache[isin]:
            return cache[isin]["symbol"]
            
        logger.info(f"Tłumaczenie ISIN {isin} na Ticker...")
        best_match = cls.search_isin(isin)
        if best_match and "symbol" in best_match:
            symbol = best_match["symbol"]
            cache[isin] = {
                "symbol": symbol,
                "name": best_match.get("longname") or best_match.get("shortname", ""),
                "exchange": best_match.get("exchange", "")
            }
            cls.save_cache(cache)
            logger.info(f"Odnaleziono ISIN {isin} -> {symbol} ({cache[isin]['name']})")
            return symbol
            
        logger.warning(f"Nie udało się odnaleźć Tickera dla ISIN: {isin}")
        return isin

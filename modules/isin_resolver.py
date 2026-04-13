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

    _MEMORY_CACHE: dict = {}

    @classmethod
    def load_cache(cls) -> dict:
        if cls._MEMORY_CACHE:
            return cls._MEMORY_CACHE
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    cls._MEMORY_CACHE = json.load(f)
                    return cls._MEMORY_CACHE
            except Exception as e:
                logger.warning(f"Błąd odczytu cache ISIN: {e}")
        return {}

    @classmethod
    def save_cache(cls, cache: dict):
        cls._MEMORY_CACHE = cache
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
    def _fallback_openfigi(cls, isin: str) -> dict:
        """
        Zapasowa metoda korzystająca z darmowego publicznego endpointu OpenFIGI.
        Zwraca najlepsze dopasowanie w formacie słownika (symbol, name, exchange).
        """
        url = "https://api.openfigi.com/v3/mapping"
        headers = {"Content-Type": "application/json"}
        payload = [{"idType": "ID_ISIN", "idValue": isin}]
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list) and "data" in data[0]:
                    mappings = data[0]["data"]
                    
                    yahoo_suffixes = {
                        "GY": ".DE", "LN": ".L", "L": ".L", "NA": ".AS", 
                        "FP": ".PA", "IM": ".MI", "PW": ".WA", "SW": ".SW", 
                        "VX": ".VX", "CN": ".TO", "AT": ".AX", 
                        "US": "", "UQ": "", "UW": "", "UR": ""
                    }
                    
                    priority = ["US", "UQ", "UW", "UR", "LN", "L", "GY", "NA", "PW", "FP", "IM"]
                    
                    best_match = None
                    best_priority = 999
                    name = mappings[0].get("name", "")
                    
                    for item in mappings:
                        exch = item.get("exchCode", "")
                        ticker = item.get("ticker", "")
                        if not ticker: continue
                        
                        p = priority.index(exch) if exch in priority else 999
                        if p < best_priority:
                            best_priority = p
                            suffix = yahoo_suffixes.get(exch, None)
                            name = item.get("name", name)
                            if suffix is not None:
                                best_match = f"{ticker}{suffix}"
                            else:
                                best_match = ticker
                                
                    if best_match:
                        return {"symbol": best_match, "longname": name, "exchange": "OpenFIGI"}
        except Exception as e:
            logger.error(f"Błąd OpenFIGI fallback dla {isin}: {e}")
            
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
        
        # Jeśli Yahoo zawiedzie, próbujemy OpenFIGI
        if not best_match or "symbol" not in best_match:
            logger.info(f"Yahoo Finance nie znalazło ISIN {isin}, testuję OpenFIGI...")
            best_match = cls._fallback_openfigi(isin)
            
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

    @classmethod
    def replace_isins_in_text(cls, text: str) -> str:
        """
        Wyszukuje wzorce ISIN w tekście i podmienia je na dostępne tickery,
        korzystając z metody resolve().
        """
        if not text:
            return text
            
        # Wzorzec dopasowujący słowo wyglądające na ISIN (2 litery + 9 alfanum + 1 cyfra)
        pattern = r"\b[A-Za-z]{2}[A-Za-z0-9]{9}[0-9]\b"
        
        def replacer(match):
            isin = match.group(0).upper()
            resolved = cls.resolve(isin)
            if resolved == isin:
                # Brak zmian (isin nie przetłumaczony)
                return match.group(0) # Zostaw oryginalną wielkość liter jeśli nie udalo się
            return resolved
            
        return re.sub(pattern, replacer, text)

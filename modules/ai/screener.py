import yfinance as yf
import pandas as pd
from modules.ai.asset_universe import get_sp500_tickers, get_global_etfs

class FundamentalScreener:
    """
    Warstwa 3: Mikro-Skaner Finansowy
    Odpowiada za pobieranie szerokich list rynkowych (w oparciu np. o decyzje CIO)
    i odrzucanie aktywów niepłynnych.
    """
    
    def __init__(self, min_volume: int = 500000):
        # Minimalny dzienny obrót aby odrzucić wydmuszki (domyślnie 500k sztuk)
        self.min_volume = min_volume

    def fetch_broad_universe(self, focus: str = "global") -> list:
        """
        Zwraca szeroką listę potencjalnych tickerów do przeskanowania.
        """
        if focus == "sp500":
            return get_sp500_tickers()[:150] # Limit dla przyspieszenia obliczeń
        elif focus == "global":
            # Reprezentanci globalnych rynków, wehikułów dłużnych i surowców
            return get_global_etfs()
        else:
            return get_global_etfs()
            
    def filter_liquid_assets(self, tickers: list, lookback_days: int = 252) -> list:
        """
        Filtruje listę tickerów, pobierając je hurtem za pomocą YFinance 
        i zostawiając tylko te wysoce płynne (duży Average Volume).
        """
        if not tickers:
            return []
            
        print(f"Pobieranie historii dla {len(tickers)} aktywów...")
        
        # Pobieranie "hurtowe" jest o wiele szybsze
        data = yf.download(tickers, period="1mo", threads=True, progress=False)
        
        liquid_tickers = []
        if 'Volume' in data:
            vol_data = data['Volume']
            for ticker in tickers:
                if ticker in vol_data:
                    # Bierzemy średni wolumen z ostatnich 20 dni giełdowych (1 miesiąc)
                    avg_vol = vol_data[ticker].mean()
                    if pd.notna(avg_vol) and avg_vol >= self.min_volume:
                        liquid_tickers.append(ticker)
        else:
            # Fallback jeśli YFinance zwrócił dane w innej formie dla 1 tickera
            pass

        return liquid_tickers
        
    def get_screened_universe(self, focus: str = "global") -> list:
        """Metoda fasadowa."""
        raw_tickers = self.fetch_broad_universe(focus=focus)
        return self.filter_liquid_assets(raw_tickers)

if __name__ == "__main__":
    screener = FundamentalScreener(min_volume=1000000) # Ostrzejszy filtr
    print("Screener szuka globalnych ETFów...")
    good_assets = screener.get_screened_universe("global")
    print(f"Znalazłem {len(good_assets)} bardzo płynnych ETFów z globalnej puli.")
    print("Przykłady:", good_assets[:5])

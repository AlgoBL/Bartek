import pandas as pd
from modules.ai.asset_universe import get_sp500_tickers, get_global_etfs
from modules.data_provider import fetch_data

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
            
        from modules.logger import setup_logger
        _log = setup_logger(__name__)
        _log.info(f"Filtrowanie płynności dla {len(tickers)} aktywów...")
        
        data = fetch_data(tickers, period="1mo", auto_adjust=True)
        
        if data.empty:
            _log.warning("Brak danych płynności — zwracam całą listę bez filtracji.")
            return tickers
        
        # ── Wyodrębnij Volume niezależnie od formatu MultiIndex ────────────────
        vol_df = None
        if isinstance(data.columns, pd.MultiIndex):
            lvl0 = data.columns.get_level_values(0).unique().tolist()
            lvl1 = data.columns.get_level_values(1).unique().tolist()
            if "Volume" in lvl0:
                # Format (Price, Ticker) — nowy yfinance
                vol_df = data["Volume"]
            elif "Volume" in lvl1:
                # Format (Ticker, Price) — stary yfinance
                vol_df = data.xs("Volume", axis=1, level=1)
        else:
            # Jeden ticker — flat DataFrame
            if "Volume" in data.columns:
                vol_df = data[["Volume"]]
                vol_df.columns = [tickers[0]]

        if vol_df is None or vol_df.empty:
            _log.warning("Nie udało się wyodrębnić Volume — zwracam całą listę.")
            return tickers

        liquid_tickers = []
        for ticker in tickers:
            if ticker in vol_df.columns:
                avg_vol = vol_df[ticker].mean()
                if pd.notna(avg_vol) and avg_vol >= self.min_volume:
                    liquid_tickers.append(ticker)

        if not liquid_tickers:
            _log.warning("Filtr płynności odrzucił wszystkie aktywa — zwracam oryginalną listę.")
            return tickers

        _log.info(f"Płynne aktywa: {len(liquid_tickers)}/{len(tickers)}")
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

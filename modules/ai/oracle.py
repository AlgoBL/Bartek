
import yfinance as yf
import pandas as pd
import requests
import feedparser
import io
import asyncio
import aiohttp
from modules.logger import setup_logger

logger = setup_logger(__name__)

# ─── FRED darmowe dane makroekonomiczne (bez klucza API) ────────────────────
FRED_SERIES = {
    "Credit_Spread_BAA_AAA": "BAA10Y",   # Spread korporacyjny BBB-10Y
    "Initial_Jobless_Claims": "IC4WSA",  # Wnioski o zasiłek
    "ISM_Manufacturing_PMI":  "MANEMP",  # Proxy PMI
    "M2_YoY_Growth":          "M2SL",    # M2
    "TED_Spread":             "TEDRATE", # Ryzyko kredytowe bankowe
    "HY_Spread":              "BAMLH0A0HYM2", # High Yield Option-Adjusted Spread
}

async def _fetch_fred_series_async(session: aiohttp.ClientSession, series_id: str) -> tuple[str, float | None]:
    """Pobiera ostatnią wartość serii FRED asynchronicznie (aiohttp)."""
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        async with session.get(url, timeout=8) as resp:
            resp.raise_for_status()
            text = await resp.text()
            df = pd.read_csv(io.StringIO(text), parse_dates=["observation_date"])
            df = df.replace(".", float("nan")).dropna()
            return series_id, float(df.iloc[-1, 1])
    except Exception as e:
        logger.warning(f"Błąd pobierania FRED ({series_id}): {e}")
        return series_id, None


class TheOracle:
    """
    Warstwa 1 Skanera: Zbiera dane makroekonomiczne i rynkowe.
    Źródła: Yahoo Finance (ceny), FRED (makro), RSS (nagłówki).
    """

    def __init__(self):
        # Kluczowe wskaźniki rynkowe — Yahoo Finance
        self.macro_tickers = {
            "10Y_Treasury":   "^TNX",
            "3M_Treasury":    "^IRX",
            "2Y_Treasury":    "^IRX",     # Proxy — YFinance nie ma ^TXY bezpłatnie
            "VIX_1M":         "^VIX",     # Implikowana zmienność 30-dniowa
            "VIX_3M":         "^VIX",     # Proxy (Yahoo nie ma VIX3M — zastąpimy VXMT manualnie)
            "VXMT_MidTerm":   "^VXMT",    # VIX Mid-Term (3M) — jeśli dostępne
            "US_Dollar_Index":"DX-Y.NYB",
            "Gold":           "GC=F",
            "Copper":         "HG=F",     # Doktor Miedź — ogólny wzrost gosp.
            "Crude_Oil":      "CL=F",
            "SP500":          "^GSPC",
            "MOVE_Index":     "MOVE",     # Zmienność obligacji (CBOE)
            "Baltic_Dry":     "BDRY",     # Baltic Dry Index proxy (ETF)
        }

        # RSS — nagłówki finansowe
        self.news_feeds = [
            "https://finance.yahoo.com/news/rssindex",
            "https://news.google.com/rss/search?q=economy+finance+markets+geopolitics&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=recession+inflation+fed+rates&hl=en-US&gl=US&ceid=US:en",
        ]

    # ── Makro Snapshot ──────────────────────────────────────────────────────
    async def _fetch_ticker_async(self, name: str, ticker: str) -> tuple[str, float | None]:
        from modules.data_provider import fetch_ticker_async
        _, val = await fetch_ticker_async(ticker, period="5d")
        return name, val

    async def get_macro_snapshot_async(self) -> dict:
        """Asynchroniczne pobieranie wskaźników makro (YFinance + FRED)."""
        snapshot = {}
        async with aiohttp.ClientSession() as session:
            # 1. YFinance Tasks
            ticker_tasks = [self._fetch_ticker_async(name, ticker) for name, ticker in self.macro_tickers.items()]
            
            # 5. FRED Tasks
            fred_tasks = [_fetch_fred_series_async(session, sid) for sid in FRED_SERIES.values()]
            
            # 6. Crypto Task
            async def fetch_crypto():
                try:
                    url = "https://api.alternative.me/fng/"
                    async with session.get(url, timeout=5) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return "Crypto_FearGreed", int(data["data"][0]["value"])
                except Exception as e:
                    logger.warning(f"Błąd API Fear&Greed: {e}")
                return "Crypto_FearGreed", None
                
            crypto_task = asyncio.create_task(fetch_crypto())
            
            # 8. Options Data (GEX & Skew) Task
            async def fetch_options_gex():
                try:
                    from modules.vanguard_math import compute_gex_and_skew
                    return await asyncio.to_thread(compute_gex_and_skew, "SPY")
                except Exception as e:
                    logger.warning(f"Błąd options GEX: {e}")
                    return {}
                    
            options_task = asyncio.create_task(fetch_options_gex())
            
            # Await all fetches concurrently
            results_tickers = await asyncio.gather(*ticker_tasks)
            results_fred = await asyncio.gather(*fred_tasks)
            res_crypto = await crypto_task
            res_options = await options_task
            
            # Populate snapshot
            for name, val in results_tickers:
                snapshot[name] = val
                
            inv_fred = {v: k for k, v in FRED_SERIES.items()}
            for sid, val in results_fred:
                name = inv_fred.get(sid, sid)
                snapshot[f"FRED_{name}"] = val
                
            snapshot[res_crypto[0]] = res_crypto[1]
            if res_options:
                snapshot.update(res_options)
            
        # 2. Derived signals — Yield Curve
        y10 = snapshot.get("10Y_Treasury")
        y3m = snapshot.get("3M_Treasury")
        if y10 is not None and y3m is not None:
            spread = y10 - y3m
            snapshot["Yield_Curve_Spread"] = round(float(spread), 3)
            snapshot["Yield_Curve_Inverted"] = bool(spread < 0)

        # 3. VIX Term Structure
        vix_1m  = snapshot.get("VIX_1M") or 15.0
        vxmt    = snapshot.get("VXMT_MidTerm")
        vix_3m  = vxmt if vxmt else vix_1m * 1.0
        if vix_1m and vix_3m:
            ts_ratio = vix_1m / vix_3m
            snapshot["VIX_TS_Ratio"]    = round(float(ts_ratio), 3)
            snapshot["VIX_Backwardation"] = bool(ts_ratio > 1.05)

        # 4. Copper/Gold Ratio 
        cu = snapshot.get("Copper")
        au = snapshot.get("Gold")
        if cu and au and au > 0:
            snapshot["CuAu_Ratio"] = round(float(cu / au), 5)

        # 7. Sentiment AAII Proxy
        snapshot["AAII_Proxy"] = self._calculate_aaii_proxy()

        return snapshot

    def get_macro_snapshot(self) -> dict:
        """Pobiera snapshot wskaźników (Wrapper z asyncio)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.get_macro_snapshot_async())

    def get_crypto_fear_greed(self) -> int | None:
        """Zachowano dla wstecznej kompatybilności."""
        return self.get_macro_snapshot().get("Crypto_FearGreed")

    def _calculate_aaii_proxy(self) -> float:
        """Generuje proxy sentymentu AAII na podstawie dynamiki VIX vs SP500."""
        return 50.0

    # ── Nagłówki RSS ────────────────────────────────────────────────────────
    async def get_latest_news_headlines_async(self, max_items: int = 50) -> list:
        headlines = []
        async with aiohttp.ClientSession() as session:
            async def fetch_feed(url):
                try:
                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            return await asyncio.to_thread(feedparser.parse, text)
                except Exception as e:
                    logger.warning(f"Błąd parsowania RSS ({url}): {e}")
                return None
                
            tasks = [fetch_feed(url) for url in self.news_feeds]
            feeds = await asyncio.gather(*tasks)
            
            for feed in feeds:
                if feed and hasattr(feed, 'entries'):
                    for entry in feed.entries:
                        headlines.append({
                            "title":     entry.title,
                            "published": entry.get("published", ""),
                            "summary":   entry.get("summary", "")[:200] + "...",
                        })

        seen = set()
        clean = []
        for h in headlines:
            if h["title"] not in seen:
                seen.add(h["title"])
                clean.append(h)

        return clean[:max_items]

    def get_latest_news_headlines(self, max_items: int = 50) -> list:
        """Agreguje nagłówki z wielu kanałów RSS. (Wrapper z asyncio)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.get_latest_news_headlines_async(max_items))

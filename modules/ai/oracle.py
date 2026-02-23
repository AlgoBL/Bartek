
import yfinance as yf
import pandas as pd
import requests
import feedparser
import io

# ─── FRED darmowe dane makroekonomiczne (bez klucza API) ────────────────────
FRED_SERIES = {
    "Credit_Spread_BAA_AAA": "BAA10Y",   # Spread korporacyjny BBB-10Y (stres kredytowy)
    "Initial_Jobless_Claims": "IC4WSA",  # Tygodniowe wnioski o zasiłek (leading indicator recesji)
    "ISM_Manufacturing_PMI":  "MANEMP",  # Zatrudnienie w przemyśle (proxy PMI)
    "M2_YoY_Growth":          "M2SL",    # Podaż pieniądza M2 (inflacja monetarna)
}

def _fetch_fred_series(series_id: str) -> float | None:
    """Pobiera ostatnią wartość serii FRED przez publiczny endpoint CSV (bez klucza API)."""
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"])
        df = df.replace(".", float("nan")).dropna()
        return float(df.iloc[-1, 1])
    except Exception:
        return None


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
            "SP500":          "^GSPC",    # Rynek akcji (momentum makro)
        }

        # RSS — nagłówki finansowe
        self.news_feeds = [
            "https://finance.yahoo.com/news/rssindex",
            "https://news.google.com/rss/search?q=economy+finance+markets+geopolitics&hl=en-US&gl=US&ceid=US:en",
            "https://news.google.com/rss/search?q=recession+inflation+fed+rates&hl=en-US&gl=US&ceid=US:en",
        ]

    # ── Makro Snapshot ──────────────────────────────────────────────────────
    def get_macro_snapshot(self) -> dict:
        """Pobiera snapshot wskaźników makro (YFinance + FRED)."""
        snapshot = {}

        # 1. YFinance — ceny rynkowe
        for name, ticker in self.macro_tickers.items():
            try:
                data = yf.Ticker(ticker).history(period="5d")
                if not data.empty:
                    snapshot[name] = round(float(data["Close"].iloc[-1]), 3)
                else:
                    snapshot[name] = None
            except Exception:
                snapshot[name] = None

        # 2. Derived signals — Yield Curve
        y10 = snapshot.get("10Y_Treasury")
        y3m = snapshot.get("3M_Treasury")
        if y10 is not None and y3m is not None:
            spread = y10 - y3m
            snapshot["Yield_Curve_Spread"] = round(float(spread), 3)
            snapshot["Yield_Curve_Inverted"] = bool(spread < 0)

        # 3. VIX Term Structure (Contango vs. Backwardation)
        # Używamy VXMT jeśli dostępne, wpp VIX jako proxy obu
        vix_1m  = snapshot.get("VIX_1M")  or 15.0
        vxmt    = snapshot.get("VXMT_MidTerm")
        vix_3m  = vxmt if vxmt else vix_1m * 1.0  # fallback = flat TS
        if vix_1m and vix_3m:
            ts_ratio = vix_1m / vix_3m
            snapshot["VIX_TS_Ratio"]    = round(float(ts_ratio), 3)
            snapshot["VIX_Backwardation"] = bool(ts_ratio > 1.05)  # Panika gdy spot > futures

        # 4. Copper/Gold Ratio (ryzyko makro ekonomiczne)
        cu = snapshot.get("Copper")
        au = snapshot.get("Gold")
        if cu and au and au > 0:
            snapshot["CuAu_Ratio"] = round(float(cu / au), 5)  # Rosnący = ryzyko ON

        # 5. FRED — dane fundamentalne (makro leading)
        for name, series_id in FRED_SERIES.items():
            val = _fetch_fred_series(series_id)
            snapshot[f"FRED_{name}"] = val

        return snapshot

    # ── Nagłówki RSS ────────────────────────────────────────────────────────
    def get_latest_news_headlines(self, max_items: int = 50) -> list:
        """Agreguje nagłówki z wielu kanałów RSS do analizy NLP."""
        headlines = []
        for feed_url in self.news_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    headlines.append({
                        "title":     entry.title,
                        "published": entry.get("published", ""),
                        "summary":   entry.get("summary", "")[:200] + "...",
                    })
            except Exception:
                continue

        # Deduplikacja po tytule
        seen = set()
        clean = []
        for h in headlines:
            if h["title"] not in seen:
                seen.add(h["title"])
                clean.append(h)

        return clean[:max_items]

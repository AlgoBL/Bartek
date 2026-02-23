
"""
scanner_engine.py — Orchestracja V5.2 (Barbell Science Grade)

Pipeline:
  Oracle  →  Economist + Geopolitik + CIO  →  Screener  →  EVT Engine  →  Composite Barbell Score
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

from modules.scanner import (
    calculate_convecity_metrics,
    score_asset,
    score_asset_composite,
)
from modules.ai.oracle import TheOracle
from modules.ai.agents import LocalEconomist, LocalGeopolitics, LocalCIO
from modules.ai.screener import FundamentalScreener


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bulk_data_cached(tickers: tuple, period: str = "2y") -> pd.DataFrame:
    """
    Pobiera dane OHLCV dla wielu tickerów naraz (cache 1h).
    Zwraca MultiIndex DataFrame z poziomami (PriceType, Ticker).
    """
    try:
        data = yf.download(
            list(tickers), period=period,
            group_by="ticker", progress=False, auto_adjust=True
        )
        return data
    except Exception:
        return pd.DataFrame()


class ScannerEngine:
    """
    Główny orskiestrator Skanera V5.2.
    Zapewnia spójność wyników z misją Barbella:
      • Risk-On  → faworyzuje aktywa z grubym prawym ogonem (Krypto, Tech, Commodities)
      • Risk-Off → ostrzega przed Risky Sleeve; sklania ku ochronie kapitału
    """

    def __init__(self):
        pass

    def fetch_bulk_data(self, tickers: list, period: str = "2y") -> pd.DataFrame:
        return fetch_bulk_data_cached(tuple(tickers), period)

    # ── Mikro Skan (EVT) ────────────────────────────────────────────────────
    def scan_markets(self, tickers: list, progress_callback=None) -> pd.DataFrame:
        """
        Oblicza metryki EVT dla listy tickerów.
        Przekazuje dane Volume do Amihud Ratio.
        """
        if not tickers:
            return pd.DataFrame()

        data_bulk = self.fetch_bulk_data(tickers)
        results   = []

        for i, ticker in enumerate(tickers):
            try:
                # Wydobywamy DataFrame dla tickera z MultiIndex
                if isinstance(data_bulk.columns, pd.MultiIndex):
                    if ticker not in data_bulk.columns.get_level_values(0):
                        continue
                    df = data_bulk[ticker]
                else:
                    df = data_bulk

                if "Close" not in df.columns or df["Close"].dropna().empty:
                    continue

                price_series  = df["Close"].dropna()
                volume_series = df["Volume"].dropna() if "Volume" in df.columns else None

                metrics = calculate_convecity_metrics(ticker, price_series, volume_series)
                if metrics:
                    metrics["Score"] = score_asset(metrics)
                    results.append(metrics)

            except Exception:
                pass

            if progress_callback and i % 5 == 0:
                pct = (i + 1) / max(len(tickers), 1)
                progress_callback(min(pct, 0.99), f"EVT: {ticker} ({i+1}/{len(tickers)})")

        if not results:
            return pd.DataFrame()

        # Composite Barbell Score (ważony Z-Score — AQR-style)
        df_results = pd.DataFrame(results)
        df_results["Barbell Score"] = score_asset_composite(df_results).values
        return df_results

    # ── Selekcja Kandydatów ──────────────────────────────────────────────────
    def select_best_candidates(self, candidates_df: pd.DataFrame, max_count: int = 10,
                               cio_regime: str = "risk_on") -> list:
        """
        Wybiera top N kandydatów wg Barbell Score.
        W trybie Risk-On → preferuje aktywa z dużym prawym ogonem (xi_right wysoki).
        W trybie Risk-Off → zwraca pustą listę (nie wchodzimy w ryzyko).
        """
        if candidates_df.empty:
            return []

        if cio_regime == "risk_off":
            # W czasie paniki systomowej nie szukamy Risky Sleeve
            return []

        sort_col = "Barbell Score" if "Barbell Score" in candidates_df.columns else "Score"
        top = candidates_df.sort_values(sort_col, ascending=False)

        # Filtr bezpieczeństwa: wykluczamy aktywa z ujemną skewnością i zerowym Kelly
        if "Skewness" in top.columns:
            top = top[top["Skewness"] >= -0.5]  # Lekka tolerancja
        if "Kelly Full" in top.columns:
            top = top[top["Kelly Full"] > 0]

        return top.head(max_count)["Ticker"].tolist()

    # ── Główna Orkiestracja V5.2 ─────────────────────────────────────────────
    def run_v5_autonomous_scan(self, horizon_years: int, progress_callback=None) -> dict:
        """
        Pipeline: Oracle → Agenci → Screener → EVT → Composite Barbell Score.
        Spójność z Barbell Strategy na każdym etapie.
        """
        cb = progress_callback or (lambda p, m: None)

        # WARSTWA 1: Oracle — dane makro + RSS
        cb(0.05, "Wyrocznia: Pobieranie danych makro (YFinance + FRED)...")
        oracle     = TheOracle()
        macro_snap = oracle.get_macro_snapshot()
        news       = oracle.get_latest_news_headlines(40)

        # WARSTWA 2: Agenci — Nowcast + Reżim Barbella
        cb(0.20, "Ekonomista: Obliczanie Nowcast Ryzyka Makro (7 czynników)...")
        economist  = LocalEconomist()
        geo        = LocalGeopolitics()
        cio        = LocalCIO()

        econ_report = economist.analyze_macro(macro_snap)
        geo_report  = geo.analyze_news(news)
        cio_thesis  = cio.synthesize_thesis(econ_report, geo_report, horizon_years)

        cio_regime  = cio_thesis.get("regime", "risk_on")

        # WARSTWA 3: Screener — tylko płynne aktywa
        cb(0.40, f"Mikro-Skaner: Filtracja aktywów (tryb CIO: {cio_thesis['mode']})...")
        screener    = FundamentalScreener(min_volume=500_000)
        raw_universe = screener.fetch_broad_universe("global")
        liquid_assets = screener.filter_liquid_assets(raw_universe[:60])

        # WARSTWA 4: EVT — Matematyka Ogonów + Composite Score
        n_assets = len(liquid_assets)
        cb(0.60, f"EVT Engine: Ocenianie {n_assets} wyselekcjonowanych aktywów...")

        def evt_cb(p, m):
            cb(0.60 + p * 0.30, m)

        metrics_df = self.scan_markets(liquid_assets, progress_callback=evt_cb)

        # WARSTWA 5: Selekcja (z uwzględnieniem reżimu CIO)
        cb(0.92, "Barbell Score: Klasyfikacja kandydatów do Risky Sleeve...")
        top_picks = self.select_best_candidates(
            metrics_df, max_count=10, cio_regime=cio_regime
        )

        cb(1.00, "✅ Skan zakończony. Wyniki gotowe.")

        return {
            "cio_thesis":           cio_thesis,
            "econ_report":          econ_report,
            "geo_report":           geo_report,
            "macro_snapshot":       macro_snap,
            "scanned_universe_size":n_assets,
            "top_picks":            top_picks,
            "metrics_df":           metrics_df,
        }

"""
async_data_loader.py — Asynchroniczne pobieranie danych makroekonomicznych.

Zamienia synchroniczne calls do FRED/external APIs na współbieżne
(asyncio + aiohttp), co redukuje czas ładowania Control Center
z sekwencyjnego T_total = sum(T_i) do T_total = max(T_i).

Referencje:
  - aiohttp: https://docs.aiohttp.org/
  - asyncio.gather: PEP 3156
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ─── Soft dependency —————————————————————————————————————————————————————————
try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False
    logger.warning(
        "aiohttp not installed — async fetching unavailable. "
        "Run: pip install aiohttp"
    )


# ─── FRED async fetch ─────────────────────────────────────────────────────────

_DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=10) if _AIOHTTP_AVAILABLE else None
_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


async def _fetch_fred_series(
    session: "aiohttp.ClientSession",
    series_id: str,
    api_key: str,
    limit: int = 5,
) -> tuple[str, list[dict]]:
    """
    Pobiera asynchronicznie dane szeregu FRED.

    Returns
    -------
    (series_id, observations_list) — lista słowników {"date": ..., "value": ...}
    """
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": limit,
    }
    try:
        async with session.get(_FRED_BASE, params=params) as resp:
            if resp.status != 200:
                logger.warning(f"FRED {series_id}: HTTP {resp.status}")
                return series_id, []
            data = await resp.json(content_type=None)
            return series_id, data.get("observations", [])
    except Exception as exc:
        logger.warning(f"FRED {series_id} fetch error: {exc}")
        return series_id, []


async def _fetch_url_json(
    session: "aiohttp.ClientSession",
    url: str,
    label: str,
    params: dict | None = None,
) -> tuple[str, Any]:
    """Generyczny async GET zwracający (label, json_payload)."""
    try:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return label, None
            return label, await resp.json(content_type=None)
    except Exception as exc:
        logger.warning(f"Async fetch [{label}] error: {exc}")
        return label, None


# ─── PUBLIC API ──────────────────────────────────────────────────────────────

async def fetch_fred_batch_async(
    series_ids: list[str],
    api_key: str,
    limit: int = 5,
) -> dict[str, list[dict]]:
    """
    Współbieżne pobieranie wielu szeregów FRED.

    Parameters
    ----------
    series_ids : lista ID szeregów FRED (np. ['T10Y2Y', 'STLFSI4', 'BAMLH0A0HYM2'])
    api_key    : FRED API key
    limit      : liczba ostatnich obserwacji

    Returns
    -------
    dict {series_id: [{"date": ..., "value": ...}, ...]}
    """
    if not _AIOHTTP_AVAILABLE:
        logger.error("aiohttp not available — returning empty dict")
        return {}

    async with aiohttp.ClientSession(timeout=_DEFAULT_TIMEOUT) as session:
        tasks = [
            _fetch_fred_series(session, sid, api_key, limit)
            for sid in series_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    output: dict[str, list[dict]] = {}
    for res in results:
        if isinstance(res, Exception):
            logger.warning(f"FRED batch exception: {res}")
            continue
        sid, observations = res
        output[sid] = observations
    return output


async def fetch_multiple_urls_async(
    url_configs: list[dict],
) -> dict[str, Any]:
    """
    Współbieżne pobieranie dowolnych URL-i.

    Parameters
    ----------
    url_configs : lista słowników {"label": str, "url": str, "params": dict|None}

    Returns
    -------
    dict {label: json_payload}
    """
    if not _AIOHTTP_AVAILABLE:
        return {}

    async with aiohttp.ClientSession(timeout=_DEFAULT_TIMEOUT) as session:
        tasks = [
            _fetch_url_json(session, cfg["url"], cfg["label"], cfg.get("params"))
            for cfg in url_configs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    output: dict[str, Any] = {}
    for res in results:
        if isinstance(res, Exception):
            continue
        label, payload = res
        output[label] = payload
    return output


# ─── SYNCHRONOUS WRAPPERS (dla Streamlit / non-async contexts) ────────────────

def run_fred_batch(series_ids: list[str], api_key: str, limit: int = 5) -> dict:
    """
    Synchroniczny wrapper do użycia w Streamlit / Oracle.

    Przykład:
        results = run_fred_batch(
            ["T10Y2Y", "STLFSI4", "BAMLH0A0HYM2"],
            api_key=os.environ["FRED_API_KEY"]
        )
        stlfsi_val = float(results["STLFSI4"][0]["value"]) if results.get("STLFSI4") else None
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # W środowisku Jupyter/async — użyj nest_asyncio lub create_task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, fetch_fred_batch_async(series_ids, api_key, limit)
                )
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(
                fetch_fred_batch_async(series_ids, api_key, limit)
            )
    except Exception as exc:
        logger.error(f"run_fred_batch failed: {exc}")
        return {}


def run_multiple_urls(url_configs: list[dict]) -> dict:
    """Synchroniczny wrapper dla fetch_multiple_urls_async."""
    try:
        return asyncio.run(fetch_multiple_urls_async(url_configs))
    except RuntimeError:
        # Event loop już działa
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(
                asyncio.run, fetch_multiple_urls_async(url_configs)
            )
            return future.result(timeout=30)


# ─── ORACLE INTEGRATION HELPER ────────────────────────────────────────────────

def fetch_macro_async_batch(fred_api_key: str) -> dict[str, float | None]:
    """
    Pobiera kompletny zestaw danych makroekonomicznych współbieżnie.
    Drop-in replacement dla sekwencyjnych wywołań w oracle.py.

    Returns
    -------
    dict {metric_name: float|None}
    """
    SERIES_MAP = {
        "T10Y2Y":      "Yield_Curve_Spread_2Y",
        "T10Y3M":      "Yield_Curve_Spread_3M",
        "STLFSI4":     "FRED_Financial_Stress_Index",
        "BAMLH0A0HYM2": "FRED_HY_Spread_bps",    # bps
        "BAA10Y":      "FRED_Credit_Spread_BAA",
        "M2SL":        "FRED_M2",
        "ICSA":        "FRED_Initial_Jobless_Claims",
        "DFII10":      "FRED_Real_Yield_10Y",
        "TEDRATE":     "FRED_TED_Spread",
    }

    raw = run_fred_batch(list(SERIES_MAP.keys()), fred_api_key, limit=3)

    results: dict[str, float | None] = {}
    for series_id, metric_name in SERIES_MAP.items():
        observations = raw.get(series_id, [])
        val = None
        for obs in observations:
            try:
                v = float(obs["value"])
                val = v
                break
            except (ValueError, KeyError):
                continue
        results[metric_name] = val

    # Konwersja HY Spread z % do bps
    hy_raw = results.get("FRED_HY_Spread_bps")
    if hy_raw is not None and hy_raw < 30:  # wartość w %, nie bps
        results["FRED_HY_Spread_bps"] = hy_raw * 100

    return results

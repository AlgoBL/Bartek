"""
modules/diagnostics.py
======================
Centrum diagnostyki i health-check aplikacji Barbell Dashboard.

Funkcje:
- get_recent_errors()          — lista błędów z JSON logu
- get_error_stats()            — statystyki: łącznie, per severity, per category
- get_error_timeline()         — błędy per godzina (24h) dla wykresu
- mark_error_resolved()        — oznacz błąd jako rozwiązany
- clear_all_resolved()         — usuń wszystkie resolved
- deduplicate_errors()         — scala identyczne błędy w krótkim oknie
- run_health_check()           — Yahoo Finance ping
- run_health_check_extended()  — Yahoo + FRED + cache file staleness
- safe_background_task()       — dekorator opakowujący wątki tła
"""
import time
import os
import json
from datetime import datetime, timedelta
from functools import wraps

import yfinance as yf

from modules.logger import setup_logger, AWARIE_JSON_PATH

log = setup_logger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_errors(limit: int = 500) -> list:
    """Load errors from JSON file safely."""
    if not os.path.exists(AWARIE_JSON_PATH):
        return []
    try:
        with open(AWARIE_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                return []
            import hashlib
            for i, d in enumerate(data):
                if "id" not in d:
                    # Stabilne ID dla bardzo starych wpisów JSON, żeby UI nie uderzyło w KeyError
                    stable_str = str(d.get("timestamp", "")) + str(d.get("message", "")) + str(i)
                    d["id"] = hashlib.md5(stable_str.encode()).hexdigest()
            return data[:limit]
    except Exception:
        return []


def _save_errors(errors: list) -> None:
    """Save errors list to JSON file safely."""
    try:
        with open(AWARIE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"Nie udało się zapisać listy błędów: {e}")


# ── Publiczne API ──────────────────────────────────────────────────────────────

def get_recent_errors(limit: int = 50) -> list:
    """Odczytuje najnowsze błędy zapisane w systemie."""
    return _load_errors(limit)


def get_error_stats() -> dict:
    """
    Zwraca statystyki błędów.
    
    Returns:
        dict z kluczami: total, active, resolved, by_severity, by_category,
                         trend_1h (zmiana liczby aktywnych w ciągu godziny)
    """
    errors = _load_errors(500)
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=1)
    twenty_four_h_ago = now - timedelta(hours=24)

    total = len(errors)
    active = sum(1 for e in errors if not e.get("resolved"))
    resolved = total - active

    by_severity: dict[str, int] = {}
    by_category: dict[str, int] = {}
    errors_last_hour = 0
    errors_24h = 0

    for e in errors:
        if e.get("resolved"):
            continue
        sev = e.get("severity", "UNKNOWN")
        cat = e.get("category", "Wewnętrzny")
        by_severity[sev] = by_severity.get(sev, 0) + 1
        by_category[cat] = by_category.get(cat, 0) + 1

        try:
            ts = datetime.strptime(e["timestamp"], "%Y-%m-%d %H:%M:%S")
            if ts >= one_hour_ago:
                errors_last_hour += 1
            if ts >= twenty_four_h_ago:
                errors_24h += 1
        except Exception:
            pass

    return {
        "total": total,
        "active": active,
        "resolved": resolved,
        "by_severity": by_severity,
        "by_category": by_category,
        "errors_last_hour": errors_last_hour,
        "errors_24h": errors_24h,
    }


def get_error_timeline(hours: int = 24) -> list:
    """
    Zwraca błędy per godzina dla wykresu słupkowego.
    
    Returns:
        list of dicts: {"hour": "10:00", "critical": N, "warning": N, "total": N}
    """
    errors = _load_errors(500)
    now = datetime.now()
    buckets: dict[str, dict] = {}

    for i in range(hours, -1, -1):
        slot = now - timedelta(hours=i)
        label = slot.strftime("%H:00")
        buckets[label] = {"hour": label, "CRITICAL": 0, "ERROR": 0, "WARNING": 0, "total": 0}

    for e in errors:
        try:
            ts = datetime.strptime(e["timestamp"], "%Y-%m-%d %H:%M:%S")
            if ts < now - timedelta(hours=hours):
                continue
            label = ts.strftime("%H:00")
            if label not in buckets:
                continue
            sev = e.get("severity", "WARNING")
            if sev in buckets[label]:
                buckets[label][sev] = buckets[label].get(sev, 0) + 1
            buckets[label]["total"] += 1
        except Exception:
            pass

    return list(buckets.values())


def mark_error_resolved(error_id: str) -> None:
    """Zaznacza błąd o danym ID jako rozwiązany."""
    errors = _load_errors(500)
    for d in errors:
        if d.get("id") == error_id:
            d["resolved"] = True
            d["resolved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_errors(errors)


def clear_all_resolved() -> int:
    """Usuwa wszystkie błędy oznaczone jako resolved. Zwraca liczbę usuniętych."""
    errors = _load_errors(500)
    before = len(errors)
    errors = [e for e in errors if not e.get("resolved")]
    _save_errors(errors)
    return before - len(errors)


def deduplicate_errors() -> int:
    """
    Scala identyczne błędy (ten sam message+module) w oknie 10 minut.
    Zwraca liczbę scalonych duplikatów.
    """
    errors = _load_errors(500)
    dedup_window = 600  # 10 minut
    merged_count = 0
    keep = []
    removed_ids: set[str] = set()

    for i, e in enumerate(errors):
        if e.get("id") in removed_ids:
            continue
        keep.append(e)
        try:
            ts_i = datetime.strptime(e["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            continue

        for j, e2 in enumerate(errors):
            if i == j or e2.get("id") in removed_ids:
                continue
            if e2.get("id") == e.get("id"):
                continue
            if (
                e2.get("message") == e.get("message")
                and e2.get("module") == e.get("module")
                and not e2.get("resolved")
                and not e.get("resolved")
            ):
                try:
                    ts_j = datetime.strptime(e2["timestamp"], "%Y-%m-%d %H:%M:%S").timestamp()
                    if abs(ts_i - ts_j) < dedup_window:
                        e["count"] = e.get("count", 1) + e2.get("count", 1)
                        e["last_seen"] = max(e.get("timestamp", ""), e2.get("timestamp", ""))
                        removed_ids.add(e2.get("id", ""))
                        merged_count += 1
                except Exception:
                    pass

    _save_errors(keep)
    return merged_count


def run_health_check() -> dict:
    """Wysyła zapytania testowe do Yahoo Finance."""
    results = {}
    start = time.time()
    try:
        df = yf.download("SPY", period="1d", progress=False)
        if df.empty:
            raise ValueError("Pusty pakiet danych od Yahoo")
        results["yahoo"] = {"status": "ok", "time_ms": int((time.time() - start) * 1000)}
    except Exception as e:
        results["yahoo"] = {
            "status": "fail",
            "time_ms": int((time.time() - start) * 1000),
            "error": str(e),
        }
        log.warning(f"Health Check: Brak połączenia z Yahoo Finance: {e}")
    return results


def run_health_check_extended() -> dict:
    """
    Rozszerzony health check:
    - Yahoo Finance (pobierz SPY 1d)
    - FRED API (ping przez requests lub fredapi)
    - Cache file (staleness — ile minut temu zapisano)
    
    Returns:
        dict: {"yahoo": {...}, "fred": {...}, "cache": {...}}
    """
    results = {}

    # 1. Yahoo Finance
    start = time.time()
    try:
        df = yf.download("SPY", period="1d", progress=False)
        if df.empty:
            raise ValueError("Pusty pakiet danych od Yahoo")
        results["yahoo"] = {
            "status": "ok",
            "time_ms": int((time.time() - start) * 1000),
            "label": "Yahoo Finance",
        }
    except Exception as e:
        results["yahoo"] = {
            "status": "fail",
            "time_ms": int((time.time() - start) * 1000),
            "error": str(e)[:80],
            "label": "Yahoo Finance",
        }

    # 2. FRED API (ping via requests — sprawdź dostępność)
    start = time.time()
    try:
        import requests
        r = requests.get(
            "https://api.stlouisfed.org/fred/series?series_id=DFF&api_key=test&file_type=json",
            timeout=5,
        )
        # 400 = zły klucz ale FRED odpowiada = online
        ok = r.status_code in (200, 400)
        results["fred"] = {
            "status": "ok" if ok else "fail",
            "time_ms": int((time.time() - start) * 1000),
            "label": "FRED API (St. Louis Fed)",
        }
    except Exception as e:
        results["fred"] = {
            "status": "fail",
            "time_ms": int((time.time() - start) * 1000),
            "error": str(e)[:80],
            "label": "FRED API (St. Louis Fed)",
        }

    # 3. Cache file — sprawdź świeżość
    cache_candidates = [
        os.path.join("data", "market_cache.json"),
        "heartbeat_cache.json",
    ]
    cache_info = {"status": "fail", "label": "Heartbeat Cache", "time_ms": 0}
    for path in cache_candidates:
        if os.path.exists(path):
            try:
                mtime = os.path.getmtime(path)
                age_minutes = (time.time() - mtime) / 60
                size_kb = os.path.getsize(path) // 1024
                cache_info = {
                    "status": "ok" if age_minutes < 120 else "warn",
                    "label": "Heartbeat Cache",
                    "time_ms": 0,
                    "age_minutes": round(age_minutes, 1),
                    "size_kb": size_kb,
                    "path": path,
                }
                break
            except Exception as e:
                cache_info["error"] = str(e)[:80]
        else:
            cache_info["error"] = "Plik cache nie istnieje"

    results["cache"] = cache_info
    return results


def safe_background_task(func):
    """
    Dekorator opakowujący funkcje asynchroniczne i wielowątkowe.
    Jeśli funkcja rzuci błąd, zostanie on cicho przechwycony za kulisami
    by nie psuć widoku w przeglądarce.
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

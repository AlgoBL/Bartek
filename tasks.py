"""
tasks.py — Celery + Redis Task Queue dla Intelligent Barbell

Uruchamia długotrwałe operacje (skan V6, backtest) asynchronicznie.
Streamlit polling sprawdza status przez task_id.

═══════════════════════════════════════════════════════════════
WYMAGANIA:
  pip install celery redis

START BROKERA (Windows — jeden z opcji):
  Opcja A — Redis via WSL2 (zalecane):
    wsl -d Ubuntu
    sudo service redis-server start

  Opcja B — Memurai (natywny Redis dla Windows):
    https://www.memurai.com/

  Opcja C — Docker:
    docker run -d -p 6379:6379 redis:alpine

START WORKERA (z katalogu projektu):
  celery -A tasks worker --loglevel=info --pool=solo

UŻYCIE w Streamlit:
  from tasks import scan_task, get_task_result
  task = scan_task.delay(horizon_years=5)
  result = get_task_result(task.id)
═══════════════════════════════════════════════════════════════
"""

from celery import Celery
from celery.result import AsyncResult
import os

# ─── Konfiguracja Celery ──────────────────────────────────────────────────────

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "barbell_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

app.conf.update(
    # Timeouty
    task_soft_time_limit=300,   # 5 min soft limit (ScannerTimeout warning)
    task_time_limit=360,        # 6 min hard kill

    # Serializacja
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Retries
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Wyniki
    result_expires=3600,        # Wyniki w Redis przez 1h
    result_compression="gzip",

    # Worker
    worker_prefetch_multiplier=1,  # Jeden task na raz (long tasks)
    worker_max_tasks_per_child=50, # Reset workera po 50 taskach (memory leak prevention)
)

# ─── Taski ────────────────────────────────────────────────────────────────────

@app.task(bind=True, name="barbell.scan")
def scan_task(self, horizon_years: int = 5) -> dict:
    """
    Celery task: Uruchamia pełny pipeline skanera V6.0 asynchronicznie.

    Parametry
    ---------
    horizon_years : horyzont inwestycyjny w latach (int)

    Zwraca
    ------
    dict z kluczami: cio_thesis, econ_report, geo_report,
                     macro_snapshot, top_picks, metrics_df (JSON),
                     scanned_universe_size, data_backend, sentiment_backend
    """
    from modules.ai.scanner_engine import ScannerEngine

    engine = ScannerEngine()

    def progress_cb(pct: float, msg: str):
        """Aktualizuje meta danych Celery — widoczne przez task.info."""
        self.update_state(
            state="PROGRESS",
            meta={"progress": round(pct * 100), "message": msg},
        )

    result = engine.run_v5_autonomous_scan(
        horizon_years=horizon_years,
        progress_callback=progress_cb,
    )

    # metrics_df → JSON (nie można serializować DataFrame przez Celery)
    if "metrics_df" in result and hasattr(result["metrics_df"], "to_json"):
        result["metrics_df_json"] = result["metrics_df"].to_json(orient="records")
        del result["metrics_df"]

    return result


@app.task(bind=True, name="barbell.backtest")
def backtest_task(
    self,
    safe_tickers: list,
    risky_tickers: list,
    initial_capital: float = 100_000,
    safe_type: str = "Ticker",
    safe_fixed_rate: float = RISK_FREE_RATE_PL,
    allocation_mode: str = "AI Dynamic",
    alloc_safe_fixed: float = 0.85,
    kelly_params: dict = None,
    rebalance_strategy: str = "Monthly",
    threshold_percent: float = 0.20,
    risky_weights_dict: dict = None,
) -> dict:
    """
    Celery task: Uruchamia backtest Intelligent Barbell asynchronicznie.
    """
    from modules.simulation import run_ai_backtest
    import yfinance as yf

    def progress_cb(pct: float, msg: str):
        self.update_state(
            state="PROGRESS",
            meta={"progress": round(pct * 100), "message": msg},
        )

    # Pobierz dane
    progress_cb(0.05, "Pobieranie danych historycznych...")
    safe_data  = yf.download(safe_tickers,  period="3y", auto_adjust=True, progress=False)
    risky_data = yf.download(risky_tickers, period="3y", auto_adjust=True, progress=False)

    progress_cb(0.20, "Uruchamianie backtestu...")

    result = run_ai_backtest(
        safe_data=safe_data,
        risky_data=risky_data,
        initial_capital=initial_capital,
        safe_type=safe_type,
        safe_fixed_rate=safe_fixed_rate,
        allocation_mode=allocation_mode,
        alloc_safe_fixed=alloc_safe_fixed,
        kelly_params=kelly_params,
        rebalance_strategy=rebalance_strategy,
        threshold_percent=threshold_percent,
        progress_callback=progress_cb,
        risky_weights_dict=risky_weights_dict,
    )

    # Serializacja equity_curve
    if "equity_curve" in result and hasattr(result["equity_curve"], "to_json"):
        result["equity_curve_json"] = result["equity_curve"].to_json()
        del result["equity_curve"]

    return result


# ─── Helper functions dla Streamlit ──────────────────────────────────────────

def get_task_status(task_id: str) -> dict:
    """
    Sprawdza status zadania Celery.

    Zwraca dict:
    {
      "state":    "PENDING" | "PROGRESS" | "SUCCESS" | "FAILURE",
      "progress": 0–100,
      "message":  str,
      "result":   dict | None,
      "error":    str | None,
    }
    """
    async_result = AsyncResult(task_id, app=app)
    state = async_result.state

    if state == "PENDING":
        return {"state": "PENDING", "progress": 0, "message": "Oczekiwanie w kolejce...", "result": None, "error": None}

    elif state == "PROGRESS":
        info = async_result.info or {}
        return {
            "state":    "PROGRESS",
            "progress": info.get("progress", 0),
            "message":  info.get("message", ""),
            "result":   None,
            "error":    None,
        }

    elif state == "SUCCESS":
        return {
            "state":    "SUCCESS",
            "progress": 100,
            "message":  "✅ Zakończono",
            "result":   async_result.result,
            "error":    None,
        }

    else:  # FAILURE / REVOKED
        return {
            "state":    state,
            "progress": 0,
            "message":  "Błąd zadania",
            "result":   None,
            "error":    str(async_result.info),
        }


def is_celery_available() -> bool:
    """
    Sprawdza czy Redis broker jest dostępny.
    Używaj w Streamlit do warunkowego wyświetlania UI async vs sync.
    """
    try:
        app.control.inspect(timeout=2.0).ping()
        return True
    except Exception as e:
        from modules.logger import setup_logger
        logger = setup_logger(__name__)
        logger.error(f"Redis backend niedostępny: {e}")
        return False

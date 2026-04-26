import logging
import sys
import os
import json
import traceback
from datetime import datetime
from threading import Lock
from config import LOG_LEVEL, LOG_FORMAT

# Ensure data/logs exists
LOGS_DIR = os.path.join("data", "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
AWARIE_JSON_PATH = os.path.join(LOGS_DIR, "app_errors.json")

_json_lock = Lock()
_DEDUP_WINDOW_SECONDS = 120  # Ignoruj duplikaty w oknie 2 minut


def _detect_category(message: str, module: str) -> str:
    """Auto-detect error category from message and module name."""
    msg_lower = (message or "").lower()
    mod_lower = (module or "").lower()

    if any(k in msg_lower for k in ["yahoo", "yfinance", "remotedata", "download"]):
        return "Yahoo Finance"
    if any(k in msg_lower for k in ["fred", "federal reserve", "stlfsi", "m2sl"]):
        return "FRED API"
    if any(k in msg_lower for k in ["multiindex", "index", "keyerror", "column"]):
        return "Dane / Struktura"
    if any(k in msg_lower for k in ["asyncio", "thread", "executor", "concurrent"]):
        return "Asynchroniczność"
    if any(k in msg_lower for k in ["celery", "worker", "task"]):
        return "Celery Worker"
    if any(k in msg_lower for k in ["transformers", "finbert", "torch", "cuda"]):
        return "AI / ML"
    if any(k in msg_lower for k in ["connection", "timeout", "ssl", "network"]):
        return "Sieć"
    if "excepthook" in mod_lower or "root" in mod_lower:
        return "Krytyczny Crash"
    return "Wewnętrzny"


class JsonAwarieHandler(logging.Handler):
    """Saves WARN/ERROR/FATAL to a JSON file for the diagnostic dashboard.
    
    Enhanced with:
    - Deduplication (same message+module within 2 min → increment count)
    - Auto-category detection
    - Rotating archive at 500 entries
    """

    def emit(self, record):
        if record.levelno < logging.WARNING:
            return

        try:
            category = _detect_category(record.getMessage(), record.module)
            exc_text = None
            if record.exc_info:
                exc_text = logging.Formatter().formatException(record.exc_info)

            new_entry = {
                "id": str(int(record.created * 1000)) + str(record.lineno),
                "timestamp": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S"),
                "severity": record.levelname,
                "module": record.module,
                "file": record.filename,
                "line": record.lineno,
                "message": record.getMessage(),
                "category": category,
                "count": 1,
                "resolved": False,
                "exc_text": exc_text,
            }

            with _json_lock:
                errors = []
                if os.path.exists(AWARIE_JSON_PATH):
                    try:
                        with open(AWARIE_JSON_PATH, "r", encoding="utf-8") as f:
                            errors = json.load(f)
                        if not isinstance(errors, list):
                            errors = []
                    except (json.JSONDecodeError, Exception):
                        errors = []

                # Deduplication: check last 20 entries for same message+module
                now_ts = record.created
                deduped = False
                for existing in errors[:20]:
                    if (
                        existing.get("message") == new_entry["message"]
                        and existing.get("module") == new_entry["module"]
                        and not existing.get("resolved", False)
                    ):
                        try:
                            ex_ts = datetime.strptime(
                                existing["timestamp"], "%Y-%m-%d %H:%M:%S"
                            ).timestamp()
                            if abs(now_ts - ex_ts) < _DEDUP_WINDOW_SECONDS:
                                existing["count"] = existing.get("count", 1) + 1
                                existing["last_seen"] = new_entry["timestamp"]
                                deduped = True
                                break
                        except Exception:
                            pass

                if not deduped:
                    errors.insert(0, new_entry)

                # Rotate: archive if > 500
                if len(errors) > 500:
                    archive_path = AWARIE_JSON_PATH.replace(".json", "_archive.json")
                    try:
                        arch = []
                        if os.path.exists(archive_path):
                            with open(archive_path, "r", encoding="utf-8") as f:
                                arch = json.load(f)
                        arch = (errors[400:] + arch)[:2000]
                        with open(archive_path, "w", encoding="utf-8") as f:
                            json.dump(arch, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                    errors = errors[:400]

                with open(AWARIE_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(errors, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def setup_logger(name: str) -> logging.Logger:
    """Configures and returns a centralized logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Nasz nowy system ratunkowy zrzucający błędy
        json_handler = JsonAwarieHandler()
        logger.addHandler(json_handler)

        logger.setLevel(LOG_LEVEL)
        logger.propagate = False
    return logger


def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Catches unhandled exceptions that crash the app and logs them."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger = setup_logger("root_excepthook")
    logger.critical(
        f"Krytyczny błąd nieobsłużony (Zatrzymanie działania strony): {exc_value}",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


# Instalacja agenta szpiegującego krytyczne crashe pythona
sys.excepthook = global_exception_handler

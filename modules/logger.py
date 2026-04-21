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

class JsonAwarieHandler(logging.Handler):
    """Saves WARN, ERROR, FATAL to a JSON file for the diagnostic dashboard."""
    def emit(self, record):
        if record.levelno < logging.WARNING:
            return  # Tylko od Warningów w górę
            
        try:
            log_entry = {
                "id": str(int(record.created * 1000)) + str(record.lineno),
                "timestamp": datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S"),
                "severity": record.levelname,
                "module": record.module,
                "file": record.filename,
                "line": record.lineno,
                "message": record.getMessage(),
                "resolved": False
            }
            if record.exc_info:
                log_entry["exc_text"] = logging.Formatter().formatException(record.exc_info)
                
            with _json_lock:
                errors = []
                if os.path.exists(AWARIE_JSON_PATH):
                    try:
                        with open(AWARIE_JSON_PATH, "r", encoding="utf-8") as f:
                            errors = json.load(f)
                    except json.JSONDecodeError:
                        errors = []
                        
                errors.insert(0, log_entry)  # Od najnowszego
                if len(errors) > 500:  # Trzymaj max 500 logów
                    errors = errors[:500]
                    
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
    logger.critical(f"Krytyczny błąd nieobsłużony (Zatrzymanie działania strony): {exc_value}", exc_info=(exc_type, exc_value, exc_traceback))

# Instalacja agenta szpiegującego krytyczne crashe pythona
sys.excepthook = global_exception_handler

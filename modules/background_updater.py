import os
import time
import json
import logging
import threading
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger("BackgroundEngine")
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

CACHE_DIR = os.path.join(BASE_DIR, "data")
CACHE_FILE = os.path.join(CACHE_DIR, "market_cache.json")
TMP_CACHE_FILE = os.path.join(CACHE_DIR, "market_cache.json.tmp")

# Zabezpiecz utworzenie katalogu data
os.makedirs(CACHE_DIR, exist_ok=True)

class BackgroundDataEngine:
    """Silnik Singletona odświeżający dane rynkowe i makro w tle (The Heartbeat)."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(BackgroundDataEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._thread = None
        self._stop_event = threading.Event()
        self._last_refresh_time = None
        self._refresh_interval_minutes = 15
        self._enabled = True
        self._is_running = False
        self._initialized = True
        logger.info("BackgroundDataEngine zainicjalizowany.")

    def set_config(self, enabled: bool, interval_minutes: int):
        """Aktualizacja konfiguracji silnika bez restartu wątku."""
        self._enabled = enabled
        self._refresh_interval_minutes = max(1, interval_minutes)
        logger.info(f"Konfiguracja silnika zaaktualizowana: enabled={enabled}, interval={interval_minutes}m")

    def start(self):
        """Uruchamia wątek w tle jeśli jeszcze nie działa."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._run, name="HeartbeatThread", daemon=True)
                self._thread.start()
                self._is_running = True
                logger.info("Wątek HeartbeatThread wystartował.")

    def stop(self):
        """Zatrzymuje wątek."""
        self._stop_event.set()
        self._is_running = False
        logger.info("Zatrzymywanie wątku HeartbeatThread...")

    def fetch_now_sync(self):
        """Wymusza synchroniczne (natychmiastowe) pobranie i zapis ciche."""
        logger.info("Wymuszono natychmiastowe odświeżenie danych (Sync).")
        self._fetch_and_save()

    def _fetch_and_save(self):
        """Główna logika pobierania i atomowego zapisu do JSON."""
        try:
            from modules.ai.oracle import TheOracle
            from modules.ai.agents import LocalGeopolitics
            
            # 1. Pobieranie ciężkich danych (wcześniej zawieszało app.py)
            oracle = TheOracle()
            macro = oracle.get_macro_snapshot()
            
            geo = LocalGeopolitics()
            news = oracle.get_latest_news_headlines(30)
            geo_report = geo.analyze_news(news)

            # 2. Składanie pakietu
            now_str = datetime.now().isoformat(timespec="seconds")
            packet = {
                "timestamp": now_str,
                "status": "success",
                "macro": macro,
                "geo_report": geo_report
            }

            # 3. Zapis do TMP
            with open(TMP_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(packet, f, ensure_ascii=False)

            # 4. Atomowa podmiana - Thread Safe / Process Safe update dla Streamlit
            if os.path.exists(TMP_CACHE_FILE):
                os.replace(TMP_CACHE_FILE, CACHE_FILE)
            
            self._last_refresh_time = datetime.now()
            logger.info(f"Świeże dane rynkowe zapisane atomowo do {CACHE_FILE}")

        except Exception as e:
            logger.error(f"Błąd podczas pobierania/zapisu danych w tle: {e}")
            # Zapis awaryjny statusu
            try:
                error_packet = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "status": "error",
                    "error_message": str(e)
                }
                # Podmiana przy błędzie, by UI wiedziało, że źródło leży (bądź zostawienie starych danych)
                # Lepsze: Nie nadpisujemy dobrych starych danych, tylko zaktualizujemy log/status
                logger.warning("Nie nadpisuję głównego cache – zachowuję wersję 'stale' przed awarią.")
            except Exception as e2:
                logger.error(f"Krytyczny błąd obsługi błędu w background_updater: {e2}")

    def _run(self):
        """Pętla wątku. Wybudza się co kilkanaście sekund, by sprawdzić warunki."""
        while not self._stop_event.is_set():
            # Sprawdzenie, czy minęło wystarczająco dużo czasu od ostatniego odświeżenia
            now = datetime.now()
            
            # Wymuszenie pierwszego pobrania po starcie serwera, jeśli brak zcacheowanego pliku LUB ostatni czas jest None
            needs_update = False
            if self._last_refresh_time is None:
                needs_update = True
            elif self._enabled:
                elapsed_minutes = (now - self._last_refresh_time).total_seconds() / 60.0
                if elapsed_minutes >= self._refresh_interval_minutes:
                    needs_update = True

            if needs_update and self._enabled:
                logger.info("Wybudzanie HeartbeatThread. Pobieram nowe dane z zewnętrznych API...")
                self._fetch_and_save()
            
            # Sleep in chunks allows quick reaction to stop_event
            for _ in range(10): # 10 * 1s
                if self._stop_event.is_set():
                    break
                time.sleep(1)

# Globalna instancja dostarczana na start
bg_engine = BackgroundDataEngine()

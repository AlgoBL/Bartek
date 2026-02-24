# config.py
# Scentralizowana konfiguracja projektu Intelligent Barbell

# --- USTAWIENIA RYNKOWE I PODATKOWE ---
TAX_BELKA = 0.19
RISK_FREE_RATE_PL = 0.0551  # Stopa wolna od ryzyka w PL (np. obligacje TOS)
INFLATION_RATE_PL = 0.030   # Domyślna stopa inflacji w PL

# --- USTAWIENIA SKANERA (EVT, Płynność) ---
SCANNER_HORIZON_YEARS = 5
SCANNER_MIN_VOLUME = 500_000

# --- USTAWIENIA SYMULATORA ---
DEFAULT_INITIAL_CAPITAL = 100_000.0

# --- USTAWIENIA LOGOWANIA ---
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

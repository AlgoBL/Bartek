# config.py
# Scentralizowana konfiguracja projektu Intelligent Barbell

import os

# --- USTAWIENIA RYNKOWE I PODATKOWE ---
TAX_BELKA = 0.19
RISK_FREE_RATE_PL = 0.0551  # Stopa wolna od ryzyka w PL (np. obligacje TOS)
INFLATION_RATE_PL = 0.030   # Domyślna stopa inflacji w PL

# --- USTAWIENIA SKANERA (EVT, Płynność) ---
SCANNER_HORIZON_YEARS = 5
SCANNER_MIN_VOLUME = 500_000

# --- USTAWIENIA SYMULATORA ---
DEFAULT_INITIAL_CAPITAL = 100_000.0

# --- GLOBALNE USTAWIENIA PORTFELA (domyślne wartości fabryczne) ---
DEFAULT_SAFE_ALLOCATION = 0.85          # 85% bezpieczna, 15% ryzykowna
DEFAULT_SAFE_RATE = RISK_FREE_RATE_PL   # stopa obligacji
DEFAULT_SAFE_TYPE = "fixed"             # "fixed" = TOS, "tickers" = własny koszyk
DEFAULT_SAFE_TICKERS = ["TLT", "GLD"]  # koszyk bezpieczny (gdy safe_type=tickers)
DEFAULT_RISKY_ASSETS = [               # koszyk ryzykowny (wagi w % sumują się do 100%)
    {"ticker": "SPY",     "weight": 40.0, "asset_class": "ETF US"},
    {"ticker": "QQQ",     "weight": 30.0, "asset_class": "ETF Tech"},
    {"ticker": "NVDA",    "weight": 20.0, "asset_class": "Akcja"},
    {"ticker": "BTC-USD", "weight": 10.0, "asset_class": "Krypto"},
]

# Ścieżka do pliku z globalnymi ustawieniami (katalog główny projektu)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GLOBAL_SETTINGS_PATH = os.path.join(_BASE_DIR, "global_settings.json")

# --- USTAWIENIA LOGOWANIA ---
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

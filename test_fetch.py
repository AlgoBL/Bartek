import os
import sys

# Dodanie ścieżki
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import logging
logging.basicConfig(level=logging.INFO)

from modules.background_updater import bg_engine

try:
    print("Rozpoczynam testowe pobieranie danych...")
    bg_engine.fetch_now_sync()
    print("Zakończono sukcesem!")
except Exception as e:
    import traceback
    traceback.print_exc()

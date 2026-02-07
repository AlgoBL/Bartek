import json
import os

SECRETS_FILE = "secrets.json"

def load_api_key():
    """Wczytuje klucz API z pliku secrets.json, jeśli istnieje."""
    if os.path.exists(SECRETS_FILE):
        try:
            with open(SECRETS_FILE, "r") as f:
                data = json.load(f)
                return data.get("GEMINI_API_KEY", "")
        except Exception:
            return ""
    return ""

def save_api_key(key):
    """Zapisuje klucz API do pliku secrets.json."""
    if not key:
        return
        
    data = {}
    if os.path.exists(SECRETS_FILE):
        try:
            with open(SECRETS_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}
            
    data["GEMINI_API_KEY"] = key
    
    with open(SECRETS_FILE, "w") as f:
        json.dump(data, f)

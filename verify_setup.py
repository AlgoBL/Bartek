import sys
import importlib

packages = [
    "streamlit", "pandas", "numpy", "plotly", "scipy", "yfinance", "pygad", 
    "sklearn", "matplotlib", "lxml", "requests", "tabulate", "google.generativeai",
    "networkx", "arch", "numba", "transformers", "torch", "polars", "celery", 
    "redis", "vaderSentiment", "feedparser", "aiohttp", "pandas_datareader"
]

missing = []
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"OK: {pkg}")
    except ImportError as e:
        missing.append(pkg)
        print(f"FAIL: {pkg} ({e})")

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    sys.exit(1)
else:
    print("\nAll core dependencies verified.")
    sys.exit(0)

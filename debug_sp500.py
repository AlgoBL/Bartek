
import pandas as pd
import requests
import io
import traceback

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
print(f"Attempting to read S&P 500 from: {url}")

try:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    print("Request successful (Status 200)")
    
    tables = pd.read_html(io.StringIO(response.text))
    print(f"Success! Found {len(tables)} tables.")
    if len(tables) > 0:
        print("First table columns:", tables[0].columns)
        print("First few tickers:", tables[0]['Symbol'].head())
except Exception as e:
    print("!!! Error occurred:")
    print(e)
    traceback.print_exc()

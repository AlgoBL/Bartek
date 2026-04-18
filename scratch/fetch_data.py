import requests, json
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

session = requests.Session()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Language': 'pl-PL,pl;q=0.9,en-US;q=0.8,en;q=0.7',
}

# 1. Get main page to establish session
print("Getting main page...")
r_main = session.get("https://atlasetf.pl/", headers=headers, verify=False)
print(f"Main page status: {r_main.status_code}")
print(f"Cookies: {session.cookies.get_dict()}")

# 2. Try API with session
api_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://atlasetf.pl/etf-list/eu/',
    'Origin': 'https://atlasetf.pl',
}

print("\nTrying API list...")
r_api = session.get("https://api.atlasetf.pl/v2/etfs", headers=api_headers, verify=False)
print(f"API status: {r_api.status_code}")
print(f"Content-Type: {r_api.headers.get('Content-Type')}")
print(f"Preview: {r_api.text[:100]}")

if r_api.status_code == 200 and 'json' in r_api.headers.get('Content-Type', '').lower():
    with open("atlas_etfs.json", "w", encoding="utf-8") as f:
        json.dump(r_api.json(), f, indent=2, ensure_ascii=False)
    print("Success: atlas_etfs.json saved")
else:
    print("Failed to get JSON")

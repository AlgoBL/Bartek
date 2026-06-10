import sys, re
sys.path.insert(0, '.')
import requests

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'pl-PL,pl;q=0.9',
}

# Poszczegolne strony obligacji z aktualnym list emisyjnym
URLS = {
    'OTS': 'https://www.obligacjeskarbowe.pl/oferta-obligacji/obligacje-3-miesieczne-ots/ots0926/',
    'ROR': 'https://www.obligacjeskarbowe.pl/oferta-obligacji/obligacje-roczne-ror/ror0627/',
    'DOR': 'https://www.obligacjeskarbowe.pl/oferta-obligacji/obligacje-2-letnie-dor/dor0628/',
    'TOS': 'https://www.obligacjeskarbowe.pl/oferta-obligacji/obligacje-3-letnie-tos/tos0629/',
    'COI': 'https://www.obligacjeskarbowe.pl/oferta-obligacji/obligacje-4-letnie-coi/coi0630/',
    'EDO': 'https://www.obligacjeskarbowe.pl/oferta-obligacji/obligacje-10-letnie-edo/edo0636/',
}

for name, url in URLS.items():
    try:
        r = requests.get(url, timeout=8, headers=HEADERS)
        # Szukamy stawki: np. "4,40%" lub "4.40%" lub "oprocentowanie: 4,40"
        # Typowy pattern na tych stronach: .product-interest lub duza liczba %
        text = r.text
        
        # Wzorzec: duza liczba z procentem w naglowku strony
        m = re.search(r'(\d+)[,.](\d+)\s*%', text)
        rate_raw = f"{m.group(1)},{m.group(2)}%" if m else "NIE ZNALEZIONO"
        
        # Szukamy tez "Oprocentowanie" w poblizu
        idx = text.lower().find('oprocentowanie')
        snippet = text[idx:idx+300] if idx >= 0 else ''
        rate_near = re.search(r'(\d+)[,.](\d+)\s*%', snippet)
        rate_near_str = f"{rate_near.group(1)},{rate_near.group(2)}%" if rate_near else "brak"
        
        print(f'{name}: status={r.status_code} | pierwsza_stawka={rate_raw} | przy_oprocentowaniu={rate_near_str}')
    except Exception as e:
        print(f'{name}: ERROR {e}')

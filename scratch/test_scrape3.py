import sys, re
sys.path.insert(0, '.')
import requests
from html.parser import HTMLParser

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'pl-PL,pl;q=0.9',
}

# Strona oferty zawiera linki do aktualnych serii
r = requests.get('https://www.obligacjeskarbowe.pl/oferta-obligacji/', timeout=10, headers=HEADERS)
text = r.text

# Szukamy linkow do poszczegolnych serii, np:
# /oferta-obligacji/obligacje-3-letnie-tos/tos0629/
bond_link_pattern = r'href="(/oferta-obligacji/[^"]+/((?:ots|ror|dor|tos|coi|edo|ros|rod)\d+)/)"'
links = re.findall(bond_link_pattern, text, re.IGNORECASE)
print("Znalezione linki do serii:")
for path, series in links:
    print(f"  {series.upper()[:3]}: https://www.obligacjeskarbowe.pl{path}")

# Sprawdzamy tez czy jest bezposrednia strona z oprocentowaniem
print()
print("Fragment HTML wokol 'product-card':")
idx = text.find('product-card')
print(repr(text[idx:idx+500]))

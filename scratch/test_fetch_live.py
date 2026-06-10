import sys, re, requests, datetime
sys.path.insert(0, '.')

_OFFERS_URL = 'https://www.obligacjeskarbowe.pl/oferta-obligacji/'
_FETCH_HEADERS = {'User-Agent': 'Mozilla/5.0', 'Accept': 'text/html', 'Accept-Language': 'pl-PL'}
_BASE_URL = 'https://www.obligacjeskarbowe.pl'
BOND_SYMBOLS = ['OTS','ROR','DOR','TOS','COI','EDO']

r = requests.get(_OFFERS_URL, timeout=10, headers=_FETCH_HEADERS)
html = r.text

link_pattern = re.compile(r'href="(/oferta-obligacji/[^"]+/((?:ots|ror|dor|tos|coi|edo)\d+)/)"', re.IGNORECASE)
series_urls = {}
for path, series_code in link_pattern.findall(html):
    symbol = series_code[:3].upper()
    if symbol in BOND_SYMBOLS and symbol not in series_urls:
        series_urls[symbol] = _BASE_URL + path

print('Serie wykryte:', series_urls)

card_pattern = re.compile(
    r'product-card[^>]*>.*?product-card__promo-value[^>]*>.*?(\d+),(\d+)\s*<sub>%</sub>'
    r'.*?product-card__title[^>]*>([^<]+)<', re.DOTALL)

rates = {}
for m in card_pattern.finditer(html):
    int_part, dec_part, title = m.group(1), m.group(2), m.group(3).strip()
    rate = float(int_part + '.' + dec_part) / 100
    for sym in BOND_SYMBOLS:
        if sym.lower() in title.lower():
            rates[sym] = rate
            break

print('Stawki z kart:', rates)

missing = [s for s in BOND_SYMBOLS if s not in rates]
print('Backup dla:', missing)
for sym in missing:
    url = series_urls.get(sym)
    if url:
        rp = requests.get(url, timeout=6, headers=_FETCH_HEADERS)
        m2 = re.search(r'(\d+),(\d+)\s*%', rp.text)
        if m2:
            rates[sym] = float(m2.group(1) + '.' + m2.group(2)) / 100
            print('  ' + sym + ' backup: ' + str(rates[sym]*100) + '%')

print()
print('=== WYNIK FINALNY ===')
for sym, rate in rates.items():
    print('  ' + sym + ': ' + str(round(rate*100, 2)) + '%')
print('Pobrano ' + str(len(rates)) + '/6 obligacji')

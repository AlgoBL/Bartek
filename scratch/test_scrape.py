import sys
sys.path.insert(0, '.')
import requests, re

r = requests.get(
    'https://www.obligacjeskarbowe.pl/oferta-obligacji/',
    timeout=10,
    headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'pl-PL,pl;q=0.9',
    }
)
print('Status:', r.status_code)
print('Content length:', len(r.text))

# Search for rates in different patterns
patterns = [
    r'(\d+,\d+)%',
    r'(\d+\.\d+)%',
    r'Oprocentowanie.*?(\d)',
    r'OTS.*?(\d)',
    r'TOS.*?(\d)',
]
for pat in patterns:
    matches = re.findall(pat, r.text[:20000])
    print(f'Pattern {pat!r}: {matches[:10]}')

# Look for specific bond names
for bond in ['OTS', 'ROR', 'DOR', 'TOS', 'COI', 'EDO']:
    idx = r.text.find(bond)
    if idx >= 0:
        snippet = r.text[max(0,idx-50):idx+200]
        print(f'\n--- {bond} ---')
        print(repr(snippet))

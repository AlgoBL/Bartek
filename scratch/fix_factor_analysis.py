import os
import re

file_path = r'c:\Users\bartl\OneDrive\KOd\Bartek\pages\22_Factor_Analysis.py'

with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

replacements = {
    'Î›': 'Λ',
    'Váµ€': 'Vᵀ',
    'â†’': '→',
    'rozkÅ‚ada': 'rozkłada',
    'wyjaÅ›nionÄ…': 'wyjaśnioną',
    'wariancjÄ™': 'wariancję',
    'gÅ‚ówny': 'główny',
    'JeÅ›li': 'Jeśli',
    'zachowuje siÄ™': 'zachowuje się',
    'aktyw.': 'aktyw.',
    'WÅ‚Ä…cz': 'Włącz',
    'Nie udaÅ‚o siÄ™': 'Nie udało się',
    'pobraÄ‡': 'pobrać',
    'poÅ‚Ä…czenie': 'połączenie',
    'wyjaÅ›niona': 'wyjaśniona',
    'â€”': '—',
    'ekspozycjÄ™': 'ekspozycję',
    'wyjaÅ›nienia': 'wyjaśnienia',
    'umiejÄ™tnoÅ›Ä‡': 'umiejętność',
    'maÅ‚Ä…': 'małą',
    'alfÄ…': 'alfą',
    'zmiennoÅ›Ä‡': 'zmienność',
    'czÄ™Å›Ä‡': 'część',
    'dÅ‚ugoterminowÄ…': 'długoterminową',
    'bazujÄ…c': 'bazując',
    'roÅ›nie': 'rośnie',
    'sprzedawaÄ‡': 'sprzedać',
    'Å›': 'ś',
    'Ä…': 'ą',
    'Ä™': 'ę',
    'Å‚': 'ł',
    'Å„': 'Ń',
    'Å„': 'ń', # wait
    'Å›': 'ś',
    'Å¼': 'ż',
    'Åº': 'ź',
    'Ä‡': 'ć',
    'Ã³': 'ó',
}

for old, new in replacements.items():
    text = text.replace(old, new)

# Also fix the 4E Cognition stuff if present
text = text.replace('Å›ci', 'ści')
text = text.replace('Ã³w', 'ów')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)
print("File cleaned.")

import re
with open('pages/48_Stochastic_Errors.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Szukamy stringów wyglądających jak "#" i 8 znaków a-f0-9
matches = re.findall(r'"#[0-9a-fA-F]{8}"', text)
print("Pozostale hex8:", matches)

import re

path = 'pages/48_Stochastic_Errors.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

helper = '''
def hex_to_rgba(hex_code, alpha):
    h = hex_code.lstrip('#')
    if len(h) != 6: return hex_code
    return f"rgba({int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"
'''

if 'def hex_to_rgba' not in content:
    content = content.replace('CLR_ME    = "#ff1744"', 'CLR_ME    = "#ff1744"\\n' + helper)

content = content.replace('fillcolor=f"{color_hex}18"', 'fillcolor=hex_to_rgba(color_hex, 0.10)')
content = content.replace('fillcolor=f"{color_hex}30"', 'fillcolor=hex_to_rgba(color_hex, 0.20)')
content = content.replace('fillcolor=f"{color}20"', 'fillcolor=hex_to_rgba(color, 0.15)')
content = content.replace('fillcolor=f"{color}15"', 'fillcolor=hex_to_rgba(color, 0.10)')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Colors fixed successfully!")

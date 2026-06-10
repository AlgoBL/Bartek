import sys
sys.path.insert(0, '.')
from modules.obligacje_skarbowe import BONDS, calculate_bond_returns

tos = BONDS['TOS']
r = calculate_bond_returns(10000, tos, 0.03)
print('=== TOS 3 lata / 10 000 zl ===')
print('Brutto:', r['brutto'], '  (oczekiwane: ~1368)')
print('Belka:', r['belka'], '  (oczekiwane: ~-260)')
print('Netto:', r['netto'], '  (oczekiwane: ~1108)')
print('Lacznie:', r['lacznie'], '  (oczekiwane: ~11108)')
print('CAGR netto:', r['cagr_netto'], '%  (oczekiwane: ~3.56)')
print('Laczna pct:', r['lacznie_pct'], '%  (oczekiwane: ~11.08)')
print()
for row in r['yearly_breakdown']:
    print(row)
print()
for row in r['early_redemption']:
    print(row)

edo = BONDS['EDO']
r2 = calculate_bond_returns(10000, edo, 0.03)
print()
print('=== EDO ===')
print('Brutto:', r2['brutto'])
print('Netto:', r2['netto'])
print('CAGR:', r2['cagr_netto'], '%')
for row in r2['yearly_breakdown']:
    print(row)

ots = BONDS['OTS']
r3 = calculate_bond_returns(10000, ots, 0.03)
print()
print('=== OTS ===')
print('Brutto:', r3['brutto'], 'Netto:', r3['netto'], 'Lacznie:', r3['lacznie'])

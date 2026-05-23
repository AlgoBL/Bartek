with open('pages/48_Stochastic_Errors.py', 'r', encoding='utf-8') as f:
    text = f.read()

count = text.count('np.trapz(')
text = text.replace('np.trapz(', 'np.trapezoid(')

with open('pages/48_Stochastic_Errors.py', 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Replaced {count} occurrences of np.trapz -> np.trapezoid")

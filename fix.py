import os

with open("pages/22_Factor_Analysis.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# The error starts at line 525 (index 524) up to line 533 (index 532).
# Let's just find the bad else: and remove the duplicated block.
new_lines = []
skip = False
for i, line in enumerate(lines):
    if "elif tau_ratio > 0.4:" in line:
        new_lines.append(line)
        new_lines.append('            advice = (\n')
        new_lines.append('                f"🟠 **Podwyższona makro-zmienność** — σ_MIDAS={sigma_now_pct*100:.1f}%. "\n')
        new_lines.append('                "Rozważ użycie tej wartości jako wejścia do Symulatora Monte Carlo."\n')
        new_lines.append('            )\n')
        skip = True
    elif skip and "🟢 **Niska makro-zmienność**" in line:
        # We reached the correct else block's content
        new_lines.append('        else:\n')
        new_lines.append('            advice = (\n')
        new_lines.append(line)
        skip = False
    elif not skip:
        new_lines.append(line)

with open("pages/22_Factor_Analysis.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

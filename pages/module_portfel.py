import streamlit as st
import runpy
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown(module_header(
    title="Zarządzanie Portfelem",
    subtitle="Optymalizacja alokacji, rebalansowanie i ochrona majątku według nowoczesnych metod ilościowych.",
    icon="⚖️",
    badge="Portfolio Management"
), unsafe_allow_html=True)

options = ["Portfolio Health Monitor", "Smart Rebalancing", "Tax Optimizer PL", "Wealth Optimizer"]
sel = st.pills("Wybierz moduł:", options, default=options[0])
st.divider()

_MAP = {
    "Portfolio Health Monitor": "pages/8_Health_Monitor.py",
    "Smart Rebalancing":        "pages/16_Rebalancing.py",
    "Tax Optimizer PL":         "pages/15_Tax_Optimizer.py",
    "Wealth Optimizer":         "pages/19_Wealth_Optimizer.py",
}
if sel in _MAP:
    runpy.run_path(_MAP[sel], run_name="__main__")

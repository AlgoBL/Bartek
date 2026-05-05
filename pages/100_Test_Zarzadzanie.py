import streamlit as st
import runpy

st.title("🛠️ Zarządzanie Portfelem i Optymalizacja")
st.markdown("---")

# Zamiast tabs używamy nawigacji poziomej by uniknąć ładowania wszystkich skryptów naraz
# (co spowodowałoby konflikty w st.sidebar)
module = st.radio(
    "Wybierz moduł roboczy:",
    ["Portfolio Health Monitor", "Smart Rebalancing", "Tax Optimizer PL", "Wealth Optimizer"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

if module == "Portfolio Health Monitor":
    runpy.run_path("pages/8_Health_Monitor.py", run_name="__main__")
elif module == "Smart Rebalancing":
    runpy.run_path("pages/16_Rebalancing.py", run_name="__main__")
elif module == "Tax Optimizer PL":
    runpy.run_path("pages/15_Tax_Optimizer.py", run_name="__main__")
elif module == "Wealth Optimizer":
    runpy.run_path("pages/19_Wealth_Optimizer.py", run_name="__main__")

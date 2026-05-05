import streamlit as st
import runpy

options = ["Portfolio Health Monitor", "Smart Rebalancing", "Tax Optimizer PL", "Wealth Optimizer"]
sel = st.pills("Wybierz moduł:", options, default=options[0])
st.divider()

if sel == "Portfolio Health Monitor":
    runpy.run_path("pages/8_Health_Monitor.py", run_name="__main__")
elif sel == "Smart Rebalancing":
    runpy.run_path("pages/16_Rebalancing.py", run_name="__main__")
elif sel == "Tax Optimizer PL":
    runpy.run_path("pages/15_Tax_Optimizer.py", run_name="__main__")
elif sel == "Wealth Optimizer":
    runpy.run_path("pages/19_Wealth_Optimizer.py", run_name="__main__")

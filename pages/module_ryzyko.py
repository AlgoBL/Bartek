import streamlit as st
import runpy

options = [
    "Stress Test", "Concentration Risk", "Liquidity Risk",
    "EVT — Tail Risk", "Systemic Risk & CoVaR", "Drawdown Recovery",
    "Tail Risk Hedging", "Inżynieria Opcji"
]
sel = st.pills("Wybierz analizę ryzyka:", options, default=options[0])
st.divider()

if sel == "Stress Test":
    runpy.run_path("pages/3_Stress_Test.py", run_name="__main__")
elif sel == "Concentration Risk":
    runpy.run_path("pages/9_Concentration_Risk.py", run_name="__main__")
elif sel == "Liquidity Risk":
    runpy.run_path("pages/13_Liquidity_Risk.py", run_name="__main__")
elif sel == "EVT — Tail Risk":
    runpy.run_path("pages/5_EVT_Analysis.py", run_name="__main__")
elif sel == "Systemic Risk & CoVaR":
    runpy.run_path("pages/27_Systemic_Risk.py", run_name="__main__")
elif sel == "Drawdown Recovery":
    runpy.run_path("pages/10_Drawdown_Recovery.py", run_name="__main__")
elif sel == "Tail Risk Hedging":
    runpy.run_path("pages/14_Tail_Hedging.py", run_name="__main__")
elif sel == "Inżynieria Opcji":
    runpy.run_path("pages/32_Inzynieria_Opcji.py", run_name="__main__")

import streamlit as st
import runpy
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown(module_header(
    title="Centrum Ryzyka",
    subtitle="Kompleksowa analiza ryzyka ogona, koncentracji, płynności i scenariuszowego stress-testingu.",
    icon="🛡️",
    badge="Risk Management"
), unsafe_allow_html=True)

options = [
    "Stress Test", "Concentration Risk", "Liquidity Risk",
    "EVT — Tail Risk", "Systemic Risk & CoVaR", "Drawdown Recovery",
    "Tail Risk Hedging", "Inżynieria Opcji"
]
sel = st.pills("Wybierz analizę ryzyka:", options, default=options[0])
st.divider()

_MAP = {
    "Stress Test":           "pages/3_Stress_Test.py",
    "Concentration Risk":    "pages/9_Concentration_Risk.py",
    "Liquidity Risk":        "pages/13_Liquidity_Risk.py",
    "EVT — Tail Risk":       "pages/5_EVT_Analysis.py",
    "Systemic Risk & CoVaR": "pages/27_Systemic_Risk.py",
    "Drawdown Recovery":     "pages/10_Drawdown_Recovery.py",
    "Tail Risk Hedging":     "pages/14_Tail_Hedging.py",
    "Inżynieria Opcji":      "pages/32_Inzynieria_Opcji.py",
}
if sel in _MAP:
    runpy.run_path(_MAP[sel], run_name="__main__")

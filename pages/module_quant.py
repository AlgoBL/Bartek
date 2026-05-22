import streamlit as st
import runpy
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown(module_header(
    title="Laboratorium Quant i AI",
    subtitle="Zaawansowane modele ilościowe: Black-Litterman, HERC, DCC-GARCH, Factor Zoo i sieci przyczynowe.",
    icon="🧬",
    badge="Quant Research"
), unsafe_allow_html=True)

options = ["Black-Litterman AI", "HERC Portfolio", "DCC — Korelacje", "Factor Zoo & PCA", "Sieci Przyczynowe"]
sel = st.pills("Wybierz model:", options, default=options[0])
st.divider()

_MAP = {
    "Black-Litterman AI": "pages/6_BL_Dashboard.py",
    "HERC Portfolio":     "pages/24_HERC_Portfolio.py",
    "DCC — Korelacje":    "pages/7_DCC_Dashboard.py",
    "Factor Zoo & PCA":   "pages/22_Factor_Analysis.py",
    "Sieci Przyczynowe":  "pages/33_Sieci_Przyczynowe.py",
}
if sel in _MAP:
    runpy.run_path(_MAP[sel], run_name="__main__")

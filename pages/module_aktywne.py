import streamlit as st
import runpy
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown(module_header(
    title="Moduły Aktywne i Trening",
    subtitle="Aktywne strategie, day trading, symulacje i system zarządzania życiem — tryb treningowy.",
    icon="🎯",
    badge="Active Trading"
), unsafe_allow_html=True)

options = ["Symulator Barbell", "Day Trading", "Sentiment & Flow", "Walk-Forward CPCV", "Life OS — Łowca"]
sel = st.pills("Wybierz moduł treningowy:", options, default=options[0])
st.divider()

_MAP = {
    "Symulator Barbell":  "pages/1_Symulator.py",
    "Day Trading":        "pages/21_Day_Trading.py",
    "Sentiment & Flow":   "pages/17_Sentiment_Flow.py",
    "Walk-Forward CPCV":  "pages/23_Walk_Forward.py",
    "Life OS — Łowca":    "pages/20_Life_OS.py",
}
if sel in _MAP:
    runpy.run_path(_MAP[sel], run_name="__main__")

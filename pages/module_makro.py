import streamlit as st
import runpy
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown(module_header(
    title="Środowisko Makro i Reżimy",
    subtitle="Analiza cyklu koniunkturalnego, reżimów rynkowych i prawdopodobieństwa recesji.",
    icon="📉",
    badge="Makro Analytics"
), unsafe_allow_html=True)

options = ["Skaner Rynku", "Zegar Inwestycyjny", "Alokacja Reżimowa", "Recession Nowcasting"]
sel = st.pills("Wybierz moduł:", options, default=options[0])
st.divider()

_MAP = {
    "Skaner Rynku":        "pages/2_Skaner.py",
    "Zegar Inwestycyjny":  "pages/11_Regime_Clock.py",
    "Alokacja Reżimowa":   "pages/12_Regime_Allocation.py",
    "Recession Nowcasting":"pages/26_Recession_Nowcasting.py",
}
if sel in _MAP:
    runpy.run_path(_MAP[sel], run_name="__main__")

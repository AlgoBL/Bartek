import streamlit as st
import runpy
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown(module_header(
    title="Meta-Decyzje i Teoria",
    subtitle="Teoria gier, Bayesowskie wnioskowanie, asymetria informacji, chaos deterministyczny — fundamenty decyzji.",
    icon="♟️",
    badge="Decision Theory"
), unsafe_allow_html=True)

options = ["Przewaga Informacyjna", "Kalkulator Bayesa", "Asymetria Informacji", "Teoria Gier", "🌀 Chaos Deterministyczny"]
sel = st.pills("Wybierz narzędzie:", options, default=options[0])
st.divider()

_MAP = {
    "Przewaga Informacyjna":     "pages/34_Przewaga_Informacyjna.py",
    "Kalkulator Bayesa":         "pages/29_Kalkulator_Bayesa.py",
    "Asymetria Informacji":      "pages/31_Asymetria_Informacji.py",
    "Teoria Gier":               "pages/30_Teoria_Gier.py",
    "🌀 Chaos Deterministyczny": "pages/49_Chaos_Deterministyczny.py",
}
if sel in _MAP:
    runpy.run_path(_MAP[sel], run_name="__main__")

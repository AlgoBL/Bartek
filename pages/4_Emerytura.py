
import streamlit as st
from modules.styling import apply_styling
from modules.emerytura import render_emerytura_module
from modules.global_settings import get_gs, apply_gs_to_session, gs_sidebar_badge

# 2. Apply Custom Styling
st.markdown(apply_styling(), unsafe_allow_html=True)

# Globalne ustawienia portfela — wczytaj i wstrzyknij jako domyślne
_gs = get_gs()
apply_gs_to_session(_gs)

# Klucze pomocnicze dla modułu Emerytura — odczyt z globalnych ustawień
if "rem_initial_capital" not in st.session_state:
    st.session_state["rem_initial_capital"] = _gs.initial_capital
if "rem_expected_return" not in st.session_state:
    st.session_state["rem_expected_return"] = 0.07
if "rem_volatility" not in st.session_state:
    st.session_state["rem_volatility"] = 0.15

if "custom_stress_scenarios" not in st.session_state:
    st.session_state["custom_stress_scenarios"] = {}

render_emerytura_module()

# Badge globalnych ustawień na dole sidebara
gs_sidebar_badge()

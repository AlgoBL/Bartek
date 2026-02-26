
import streamlit as st
from modules.styling import apply_styling
from modules.emerytura import render_emerytura_module

# 2. Apply Custom Styling
st.markdown(apply_styling(), unsafe_allow_html=True)

# Klucze pomocnicze dla modułu Emerytura
if "rem_initial_capital" not in st.session_state:
    st.session_state["rem_initial_capital"] = 1000000.0
if "rem_expected_return" not in st.session_state:
    st.session_state["rem_expected_return"] = 0.07
if "rem_volatility" not in st.session_state:
    st.session_state["rem_volatility"] = 0.15

if "custom_stress_scenarios" not in st.session_state:
    st.session_state["custom_stress_scenarios"] = {}

render_emerytura_module()

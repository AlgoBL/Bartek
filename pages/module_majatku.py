import streamlit as st
from modules.styling import apply_styling
from modules.emerytura import render_emerytura_module
from modules.decumulation_ui import render_decumulation_module
from modules.global_settings import get_gs, apply_gs_to_session, gs_sidebar_badge

# Apply styling once at the top level to avoid DOM duplication/removeChild errors
st.markdown(apply_styling(), unsafe_allow_html=True)

options = ["Emerytura / FIRE", "Decumulation / SWR"]
sel = st.pills("Wybierz moduł:", options, default=options[0])
st.divider()

if sel == "Emerytura / FIRE":
    with st.container(key="emerytura_wrapper"):
        # Ensure session state is initialized
        from modules.global_settings import get_gs, apply_gs_to_session
        _gs = get_gs()
        apply_gs_to_session(_gs)
        
        if "rem_initial_capital" not in st.session_state:
            st.session_state["rem_initial_capital"] = _gs.initial_capital
        if "rem_expected_return" not in st.session_state:
            st.session_state["rem_expected_return"] = 0.07
        if "rem_volatility" not in st.session_state:
            st.session_state["rem_volatility"] = 0.15
        if "custom_stress_scenarios" not in st.session_state:
            st.session_state["custom_stress_scenarios"] = {}
            
        render_emerytura_module()
elif sel == "Decumulation / SWR":
    with st.container(key="decumulation_wrapper"):
        render_decumulation_module()

gs_sidebar_badge()

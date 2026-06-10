import streamlit as st
from modules.styling import apply_styling
from modules.obligacje_skarbowe import render_obligacje_module
from modules.global_settings import get_gs, apply_gs_to_session, gs_sidebar_badge
from modules.i18n import t

def render_page():
    _gs = get_gs()
    apply_gs_to_session(_gs)
    st.markdown(apply_styling(), unsafe_allow_html=True)
    render_obligacje_module()
    gs_sidebar_badge()

if __name__ == "__main__":
    render_page()

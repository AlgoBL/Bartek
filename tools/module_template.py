"""
module_template.py
==================
Generuje pliki boilerplate (szablony) dla nowych modułów
zapewniając ich jednolite formatowanie.

Użycie wewnętrzne w tools/project_updater.py
"""
import os

PAGE_TEMPLATE = '''import streamlit as st
from modules.styling import apply_styling
from modules.{module_name} import render_{module_name}_module
from modules.global_settings import get_gs, apply_gs_to_session, gs_sidebar_badge

def render_page():
    _gs = get_gs()
    apply_gs_to_session(_gs)
    
    st.markdown(apply_styling(), unsafe_allow_html=True)
    
    render_{module_name}_module()
    
    gs_sidebar_badge()

if __name__ == "__main__":
    render_page()
'''

MODULE_TEMPLATE = '''"""
{module_name}.py — {title_pl}
Barbell Strategy Dashboard

Kategoria: {category}
Ikona: {icon}
"""
import streamlit as st

def render_{module_name}_module():
    """Główna funkcja renderująca moduł w interfejsie."""
    st.markdown(f"## {icon} {title_pl}")
    st.info("🚧 Moduł w budowie.")
    
    # Dodaj logikę poniżej
    pass
'''

def generate_templates(module_name: str, title_pl: str, category: str, icon: str):
    """Zwraca gotowe stringi szablonów jako (page_content, module_content)"""
    page_content = PAGE_TEMPLATE.format(module_name=module_name)
    module_content = MODULE_TEMPLATE.format(
        module_name=module_name,
        title_pl=title_pl,
        category=category,
        icon=icon
    )
    return page_content, module_content

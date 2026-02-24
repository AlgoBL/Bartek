import streamlit as st
from modules.styling import apply_styling

st.set_page_config(
    page_title="Barbell Strategy Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def home():
    st.markdown(apply_styling(), unsafe_allow_html=True)
    
    st.title("⚖️ Intelligent Barbell Strategy")
    st.markdown("""
    ### Zautomatyzowana fuzja finansów, AI i nauki
    Witaj w rdzeniu analitycznym! System wspomaga podejmowanie obiektywnych decyzji w oparciu o koncepcję strategii Barbell.
    Aplikacja została oparta o zaawansowaną architekturę agentową V6.
    
    **Wybierz moduł z nowoczesnego menu (pasek boczny), aby przejść dalej:**
    
    *   **📉 Symulator**: Weryfikacja kwantowa i backtesty wytypowanych portfeli
    *   **🔍 Skaner**: Detekcja antykruchych aktywów rynkowych w skali globalnej
    *   **⚡ Stress Test**: Badanie odporności portfeli na potężne historyczne kryzysy
    *   **🏖️ Emerytura**: Optymalizacja i planowanie strategii FIRE
    """)
    
    # Handle Legacy Navigation (force_navigate)
    if "force_navigate" in st.session_state:
        target = st.session_state.pop("force_navigate")
        if target == "📉 Symulator":
            st.switch_page("pages/1_Symulator.py")
        elif target == "⚡ Stress Test":
            st.switch_page("pages/3_Stress_Test.py")

pages = {
    "Start": [
        st.Page(home, title="Strona główna", icon="🏠", default=True),
    ],
    "Narzędzia Analityczne": [
        st.Page("pages/1_Symulator.py", title="Symulator", icon="📉"),
        st.Page("pages/2_Skaner.py", title="Skaner", icon="🔍"),
        st.Page("pages/3_Stress_Test.py", title="Stress Test", icon="⚡"),
    ],
    "Planowanie": [
        st.Page("pages/4_Emerytura.py", title="Emerytura", icon="🏖️"),
    ]
}

pg = st.navigation(pages)
pg.run()


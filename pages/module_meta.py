import streamlit as st
import runpy

options = ["Przewaga Informacyjna", "Kalkulator Bayesa", "Asymetria Informacji", "Teoria Gier"]
sel = st.pills("Wybierz narzędzie:", options, default=options[0])
st.divider()

if sel == "Przewaga Informacyjna":
    runpy.run_path("pages/34_Przewaga_Informacyjna.py", run_name="__main__")
elif sel == "Kalkulator Bayesa":
    runpy.run_path("pages/29_Kalkulator_Bayesa.py", run_name="__main__")
elif sel == "Asymetria Informacji":
    runpy.run_path("pages/31_Asymetria_Informacji.py", run_name="__main__")
elif sel == "Teoria Gier":
    runpy.run_path("pages/30_Teoria_Gier.py", run_name="__main__")

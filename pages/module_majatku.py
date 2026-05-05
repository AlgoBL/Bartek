import streamlit as st
import runpy

options = ["Emerytura / FIRE", "Decumulation / SWR"]
sel = st.pills("Wybierz moduł:", options, default=options[0])
st.divider()

if sel == "Emerytura / FIRE":
    runpy.run_path("pages/4_Emerytura.py", run_name="__main__")
elif sel == "Decumulation / SWR":
    runpy.run_path("pages/25_Decumulation.py", run_name="__main__")

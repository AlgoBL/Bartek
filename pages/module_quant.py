import streamlit as st
import runpy

options = ["Black-Litterman AI", "HERC Portfolio", "DCC — Korelacje", "Factor Zoo & PCA", "Sieci Przyczynowe"]
sel = st.pills("Wybierz model:", options, default=options[0])
st.divider()

if sel == "Black-Litterman AI":
    runpy.run_path("pages/6_BL_Dashboard.py", run_name="__main__")
elif sel == "HERC Portfolio":
    runpy.run_path("pages/24_HERC_Portfolio.py", run_name="__main__")
elif sel == "DCC — Korelacje":
    runpy.run_path("pages/7_DCC_Dashboard.py", run_name="__main__")
elif sel == "Factor Zoo & PCA":
    runpy.run_path("pages/22_Factor_Analysis.py", run_name="__main__")
elif sel == "Sieci Przyczynowe":
    runpy.run_path("pages/33_Sieci_Przyczynowe.py", run_name="__main__")

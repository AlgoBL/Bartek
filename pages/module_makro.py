import streamlit as st
import runpy

options = ["Skaner Rynku", "Zegar Inwestycyjny", "Alokacja Reżimowa", "Recession Nowcasting"]
sel = st.pills("Wybierz moduł:", options, default=options[0])
st.divider()

if sel == "Skaner Rynku":
    runpy.run_path("pages/2_Skaner.py", run_name="__main__")
elif sel == "Zegar Inwestycyjny":
    runpy.run_path("pages/11_Regime_Clock.py", run_name="__main__")
elif sel == "Alokacja Reżimowa":
    runpy.run_path("pages/12_Regime_Allocation.py", run_name="__main__")
elif sel == "Recession Nowcasting":
    runpy.run_path("pages/26_Recession_Nowcasting.py", run_name="__main__")

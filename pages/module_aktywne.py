import streamlit as st
import runpy

options = ["Symulator Barbell", "Day Trading", "Sentiment & Flow", "Walk-Forward CPCV", "Life OS — Łowca"]
sel = st.pills("Wybierz moduł treningowy:", options, default=options[0])
st.divider()

if sel == "Symulator Barbell":
    runpy.run_path("pages/1_Symulator.py", run_name="__main__")
elif sel == "Day Trading":
    runpy.run_path("pages/21_Day_Trading.py", run_name="__main__")
elif sel == "Sentiment & Flow":
    runpy.run_path("pages/17_Sentiment_Flow.py", run_name="__main__")
elif sel == "Walk-Forward CPCV":
    runpy.run_path("pages/23_Walk_Forward.py", run_name="__main__")
elif sel == "Life OS — Łowca":
    runpy.run_path("pages/20_Life_OS.py", run_name="__main__")

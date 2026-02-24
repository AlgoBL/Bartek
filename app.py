import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from modules.styling import apply_styling

st.set_page_config(
    page_title="Barbell Strategy Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_control_center_data():
    from modules.ai.oracle import TheOracle
    from modules.ai.agents import LocalGeopolitics
    
    oracle = TheOracle()
    macro = oracle.get_macro_snapshot()
    
    # NLP Sentiment
    geo = LocalGeopolitics()
    news = oracle.get_latest_news_headlines(30)
    geo_report = geo.analyze_news(news)
    
    return macro, geo_report

def calculate_regime_score(macro, geo_report):
    score = 50.0
    
    # 1. Yield Curve
    if macro.get("Yield_Curve_Inverted", False):
        score += 15.0
    elif macro.get("Yield_Curve_Spread", 0) > 1.5:
        score -= 10.0
        
    # 2. VIX & Option
    vix_ts = macro.get("VIX_TS_Ratio", 1.0)
    if vix_ts > 1.05: score += 20.0
    elif vix_ts < 0.9: score -= 10.0
    
    gex = macro.get("total_gex_billions", 0)
    if gex < 0: score += 15.0
    elif gex > 5: score -= 10.0
    
    # 3. Sentiment
    sent = geo_report.get("compound_sentiment", 0.0)
    score -= sent * 15.0
    
    # 4. Economy
    jobless = macro.get("FRED_Initial_Jobless_Claims", 250000)
    if jobless > 300000: score += 10.0
    
    hy = macro.get("FRED_HY_Spread", 4.0)
    if hy > 6.0: score += 15.0
    elif hy < 4.0: score -= 5.0
    
    return max(1.0, min(100.0, score))

def determine_business_cycle(macro):
    yc = macro.get("Yield_Curve_Spread", 0)
    claims = macro.get("FRED_Initial_Jobless_Claims", 250000)
    pmi = macro.get("FRED_ISM_Manufacturing_PMI", 50.0)
    
    if yc < 0:
        return "Spowolnienie (Slowdown)", "Zacieśnianie polityki przez bank centralny. Inwersja krzywej rentowności.", "📉", "#f39c12"
    elif claims > 300000 and yc >= 0:
        return "Recesja (Recession)", "Kryzys gospodarczy. Rosnące bezrobocie, dno rynkowe.", "💀", "#e74c3c"
    elif pmi < 50 and yc > 0.5:
        return "Odrodzenie (Recovery)", "Dno za nami. Stymulacja systemowa, hossa na rynkach dyskontuje poprawę.", "🌱", "#3498db"
    else:
        return "Ekspansja (Expansion)", "Silny wzrost gospodarczy. Zyski rosną, optymizm na rynkach.", "🚀", "#2ecc71"

def draw_regime_radar(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Regime Radar (Poziom Paniki)", 'font': {'size': 24, 'color': 'white'}},
        number={'font': {'size': 48, 'color': 'white'}, 'suffix': " / 100"},
        gauge={
            'axis': {'range': [1, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "black",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [1, 30], 'color': "rgba(46, 204, 113, 0.8)", 'name': "Hossa"},
                {'range': [30, 70], 'color': "rgba(243, 156, 18, 0.8)", 'name': "Ostrożnie"},
                {'range': [70, 100], 'color': "rgba(231, 76, 60, 0.8)", 'name': "Panika / Krach"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 6},
                'thickness': 0.85,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )
    return fig

def home():
    st.markdown(apply_styling(), unsafe_allow_html=True)
    
    # Handle Legacy Navigation (force_navigate)
    if "force_navigate" in st.session_state:
        target = st.session_state.pop("force_navigate")
        if target == "📉 Symulator":
            st.switch_page("pages/1_Symulator.py")
        elif target == "⚡ Stress Test":
            st.switch_page("pages/3_Stress_Test.py")

    st.title("⚖️ Control Center V8")
    st.markdown("### Autonomiczny Pulpit Nawigacyjny Sztangi")
    
    with st.spinner("Synchronizacja z sensorami globalnymi (TheOracle)..."):
        try:
            macro, geo_report = fetch_control_center_data()
        except Exception as e:
            st.error(f"Błąd synchronizacji sensorów: {e}")
            macro, geo_report = {}, {}

    if not macro:
        st.warning("Brak danych makro. Sprawdź połączenie internetowe.")
        return

    score = calculate_regime_score(macro, geo_report)
    
    # --- RADAR REŻIMU ---
    st.plotly_chart(draw_regime_radar(score), use_container_width=True)
    
    st.divider()

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('### ☣️ Zegar Zagłady (Doomsday Matrix)')
        
        c1, c2, c3 = st.columns(3)
        
        # 1. VIX Term Structure
        bkwd = macro.get("VIX_Backwardation", False)
        vix_ts = macro.get("VIX_TS_Ratio", 0.0)
        vix_icon = "🔥 Panika (Backwardation)" if bkwd else "🛡️ Spokój (Contango)"
        vix_color = "#e74c3c" if bkwd else "#2ecc71"
        
        with c1:
            st.markdown(f"<div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid {vix_color}; text-align: center;'>"
                        f"<h4 style='margin-bottom: 5px;'>Krzywa VIX</h4>"
                        f"<h2 style='margin: 0; color: {vix_color};'>{vix_ts:.2f}</h2>"
                        f"<p style='margin-top: 5px; color: #aaaaaa;'>{vix_icon}</p>"
                        f"</div>", unsafe_allow_html=True)

        # 2. Yield Curve
        yc_spread = macro.get("Yield_Curve_Spread", 0.0)
        yc_inv = macro.get("Yield_Curve_Inverted", False)
        yc_icon = "⚠️ Odwrócona" if yc_inv else "✅ Rosnąca"
        yc_color = "#e74c3c" if yc_inv else "#2ecc71"
        
        with c2:
            st.markdown(f"<div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid {yc_color}; text-align: center;'>"
                        f"<h4 style='margin-bottom: 5px;'>Spread Rentowności (US)</h4>"
                        f"<h2 style='margin: 0; color: {yc_color};'>{yc_spread:.2f}%</h2>"
                        f"<p style='margin-top: 5px; color: #aaaaaa;'>{yc_icon}</p>"
                        f"</div>", unsafe_allow_html=True)

        # 3. NLP Sentiment
        sent = geo_report.get("compound_sentiment", 0.0)
        sent_icon = "📰 Negatywny" if sent < -0.15 else ("📰 Neutralny" if sent < 0.15 else "📰 Pozytywny")
        sent_color = "#e74c3c" if sent < -0.15 else ("#f39c12" if sent < 0.15 else "#2ecc71")
        
        with c3:
            st.markdown(f"<div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid {sent_color}; text-align: center;'>"
                        f"<h4 style='margin-bottom: 5px;'>Global Sentyment</h4>"
                        f"<h2 style='margin: 0; color: {sent_color};'>{sent:.2f}</h2>"
                        f"<p style='margin-top: 5px; color: #aaaaaa;'>{sent_icon}</p>"
                        f"</div>", unsafe_allow_html=True)
                        
    with col2:
        st.markdown('### 🏭 Zegar Biznesowy')
        phase, desc, icon, color = determine_business_cycle(macro)
        
        st.markdown(f"""
        <div style='background-color: #1e1e1e; padding: 25px; border-radius: 15px; text-align: center; border: 2px solid {color}'>
            <h1 style='font-size: 60px; margin: 0;'>{icon}</h1>
            <h3 style='color: {color}; margin-top: 10px;'>{phase}</h3>
            <p style='color: #dddddd; font-size: 14px; margin-top: 15px;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

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


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
    
    # 1. Stress indicators (Negative Gamma, High VIX, High TED)
    vix_ts = macro.get("VIX_TS_Ratio", 1.0)
    if vix_ts > 1.05: score += 15.0
    
    gex = macro.get("total_gex_billions", 0)
    if gex < 0: score += 10.0
    
    ted = macro.get("FRED_TED_Spread")
    if ted and ted > 0.5: score += 10.0
    
    fci = macro.get("FRED_Financial_Stress_Index")
    if fci and fci > 0: score += 15.0 # Positive means stress above normal
    
    # 2. Macro (Real Yields, Yield Curve)
    if macro.get("Yield_Curve_Inverted", False): score += 10.0
    
    ry = macro.get("FRED_Real_Yield_10Y")
    if ry and ry > 2.0: score += 5.0
    
    # 3. Sentiment & Breadth
    sent = geo_report.get("compound_sentiment", 0.0)
    score -= sent * 15.0
    
    breadth = macro.get("Breadth_Momentum")
    if breadth and breadth < -0.02: score += 10.0
    
    return max(1.0, min(100.0, score))

def determine_business_cycle(macro):
    yc = macro.get("Yield_Curve_Spread", 0)
    claims = macro.get("FRED_Initial_Jobless_Claims", 250000)
    pmi = macro.get("FRED_ISM_Manufacturing_PMI", 50.0)
    
    if yc < 0:
        return "Spowolnienie (Slowdown)", "Zacieśnianie polityki przez bank centralny. Inwersja krzywej.", "📉", "#f39c12"
    elif claims > 300000 and yc >= 0:
        return "Recesja (Recession)", "Kryzys gospodarczy. Rosnące bezrobocie, dno rynkowe.", "💀", "#e74c3c"
    elif pmi < 50 and yc > 0.5:
        return "Odrodzenie (Recovery)", "Dno za nami. Stymulacja systemowa dyskontuje poprawę.", "🌱", "#3498db"
    else:
        return "Ekspansja (Expansion)", "Silny wzrost gospodarczy. Zyski rosną, optymizm na rynkach.", "🚀", "#2ecc71"

def draw_regime_radar(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Regime Radar (Poziom Paniki)", 'font': {'size': 20, 'color': 'white'}},
        number={'font': {'size': 40, 'color': 'white'}, 'suffix': " / 100"},
        gauge={
            'axis': {'range': [1, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "black",
            'borderwidth': 1,
            'bordercolor': "gray",
            'steps': [
                {'range': [1, 30], 'color': "rgba(46, 204, 113, 0.6)", 'name': "Hossa"},
                {'range': [30, 70], 'color': "rgba(243, 156, 18, 0.6)", 'name': "Ostrożnie"},
                {'range': [70, 100], 'color': "rgba(231, 76, 60, 0.6)", 'name': "Panika / Krach"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 5},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

def make_sensor_card(title, value, icon, color, desc):
    return f"""
    <div style='background-color: #1a1c23; padding: 15px; border-radius: 10px; border-top: 3px solid {color}; margin-bottom: 10px; height: 110px;'>
        <div style='display: flex; justify-content: space-between; align-items: start;'>
            <h5 style='margin: 0; color: #888; font-size: 13px;'>{title}</h5>
            <span style='font-size: 18px;'>{icon}</span>
        </div>
        <h3 style='margin: 5px 0; color: {color}; font-size: 20px; font-weight: bold;'>{value}</h3>
        <p style='color: #666; font-size: 10px; margin: 0; line-height: 1.1;'>{desc}</p>
    </div>
    """

def get_vanguard_report(score, macro, geo_report):
    sent = geo_report.get("compound_sentiment", 0)
    cycle, _, _, _ = determine_business_cycle(macro)
    
    if score > 70:
        return "⚠️ ALARM: Wysokie ryzyko systemowe. Dark Pools i VIX wskazują na kaskadową zmienność. Rekomendacja: Obrona kapitału.", "#e74c3c"
    elif score < 35 and sent > 0.1:
        return "✅ STATUS: Rynek w silnym reżimie Risk-On. Płynność wspiera wzrosty. Rekomendacja: Ekspansja w Risky Sleeve.", "#2ecc71"
    else:
        return f"⚖️ STATUS: Reżim mieszany. Faza {cycle}. Rynek szuka kierunku przy stabilnych warunkach finansowych.", "#3498db"

def home():
    st.markdown(apply_styling(), unsafe_allow_html=True)
    
    # Handle Legacy Navigation (force_navigate)
    if "force_navigate" in st.session_state:
        target = st.session_state.pop("force_navigate")
        if target == "📉 Symulator":
            st.switch_page("pages/1_Symulator.py")
        elif target == "⚡ Stress Test":
            st.switch_page("pages/3_Stress_Test.py")

    with st.spinner("Inicjalizacja Terminala Vanguard V9.0..."):
        try:
            macro, geo_report = fetch_control_center_data()
        except Exception as e:
            st.error(f"Błąd synchronizacji terminala: {e}")
            macro, geo_report = {}, {}

    if not macro:
        st.warning("Brak połączenia z siecią sensorów.")
        return

    score = calculate_regime_score(macro, geo_report)
    report_text, report_color = get_vanguard_report(score, macro, geo_report)
    
    # --- TOP HEADER: INTELLIGENCE REPORT ---
    st.markdown(f"""
    <div style='background-color: #0d0e12; padding: 20px; border-radius: 12px; border-left: 10px solid {report_color}; margin-bottom: 25px;'>
        <h4 style='margin: 0; color: #888; font-size: 14px; letter-spacing: 1px;'>VANGUARD INTELLIGENCE REPORT</h4>
        <h2 style='margin: 5px 0 0 0; color: white; font-size: 22px;'>{report_text}</h2>
    </div>
    """, unsafe_allow_html=True)

    # --- MAIN GAUGES ---
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.plotly_chart(draw_regime_radar(score), use_container_width=True)
    with col_t2:
        phase, desc, icon, color = determine_business_cycle(macro)
        st.markdown(f"""
        <div style='background-color: #1a1c23; padding: 30px; border-radius: 15px; text-align: center; border: 1px solid #333; height: 280px; display: flex; flex-direction: column; justify-content: center;'>
            <h1 style='font-size: 50px; margin: 0;'>{icon}</h1>
            <h2 style='color: {color}; margin-top: 10px; font-size: 20px;'>{phase}</h2>
            <p style='color: #888; font-size: 12px; margin-top: 10px;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    
    # --- 4-PILLAR GRID ---
    p1, p2, p3, p4 = st.columns(4)
    
    with p1:
        st.markdown("<h4 style='text-align: center; color: #e74c3c;'>🚨 Stress & Volatility</h4>", unsafe_allow_html=True)
        # 1. Bond Vol (MOVE Proxy)
        bv = macro.get("Bond_Vol_Proxy")
        bv_val = f"{bv:.1f}%" if bv else "N/A"
        st.markdown(make_sensor_card("Bond Vol (MOVE Proxy)", bv_val, "📉", "#f39c12", "Zmienność rynku długu. Kluczowa dla płynności."), unsafe_allow_html=True)
        # 2. TED Spread
        ted = macro.get("FRED_TED_Spread")
        ted_val = f"{ted:.3f}" if ted else "N/A"
        ted_col = "#e74c3c" if ted and ted > 0.4 else "#2ecc71"
        st.markdown(make_sensor_card("TED Spread", ted_val, "🏦", ted_col, "Zaufanie międzybankowe. Wzrost = kryzys kredytowy."), unsafe_allow_html=True)
        # 3. GEX
        gex = macro.get("total_gex_billions")
        gex_val = f"${gex:.1f}B" if gex is not None else "N/A"
        st.markdown(make_sensor_card("Dark Pool GEX", gex_val, "🎰", "#3498db", "Gamma Exposure Market Makerów na SPY."), unsafe_allow_html=True)

    with p2:
        st.markdown("<h4 style='text-align: center; color: #3498db;'>🏛️ Macro & Policy</h4>", unsafe_allow_html=True)
        # 1. Financial Stress Index
        fci = macro.get("FRED_Financial_Stress_Index")
        fci_val = f"{fci:.2f}" if fci is not None else "N/A"
        fci_col = "#e74c3c" if fci and fci > 0 else "#2ecc71"
        st.markdown(make_sensor_card("Fed Financial Stress", fci_val, "🏛️", fci_col, "Agregat stresu rynkowego. >0 to gorzej niż norma."), unsafe_allow_html=True)
        # 2. Yield Curve
        yc = macro.get("Yield_Curve_Spread", 0)
        st.markdown(make_sensor_card("Yield Curve (10Y-2Y)", f"{yc:.2f}%", "⏳", "#3498db", "Inwersja krzywej rentowności w USA."), unsafe_allow_html=True)
        # 3. Real Yield
        ry = macro.get("FRED_Real_Yield_10Y")
        st.markdown(make_sensor_card("Real 10Y Yield", f"{ry:.2f}%", "⚓", "#f1c40f", "Rentowność po odjęciu oczekiwanej inflacji."), unsafe_allow_html=True)

    with p3:
        st.markdown("<h4 style='text-align: center; color: #2ecc71;'>🚚 Real Economy</h4>", unsafe_allow_html=True)
        # 1. Baltic Dry Index
        bdry = macro.get("Baltic_Dry")
        bdry_val = f"{bdry:.1f}" if bdry else "N/A"
        st.markdown(make_sensor_card("Baltic Dry (Freight)", bdry_val, "🚢", "#2ecc71", "Koszt transportu morskiego. Puls handlu."), unsafe_allow_html=True)
        # 2. Dr. Copper
        cu = macro.get("Copper")
        cu_val = f"${cu:.2f}" if cu else "N/A"
        st.markdown(make_sensor_card("Dr. Copper", cu_val, "🏗️", "#e67e22", "Miedź jako wskaźnik globalnej aktywności."), unsafe_allow_html=True)
        # 3. Jobless Claims
        claims = macro.get("FRED_Initial_Jobless_Claims")
        claims_val = f"{claims/1000:.0f}k" if claims else "N/A"
        st.markdown(make_sensor_card("Jobless Claims", claims_val, "👥", "#3498db", "Nowe wnioski o zasiłek w USA co tydzień."), unsafe_allow_html=True)

    with p4:
        st.markdown("<h4 style='text-align: center; color: #f1c40f;'>🧠 Sent. & Breadth</h4>", unsafe_allow_html=True)
        # 1. News Sentiment
        sent = geo_report.get("compound_sentiment", 0)
        st.markdown(make_sensor_card("Global News NLP", f"{sent:.2f}", "📰", "#f1c40f", "Sentyment z 30 ostatnich nagłówków globalnych."), unsafe_allow_html=True)
        # 2. Market Breadth
        breadth = macro.get("Breadth_Momentum")
        br_val = f"{breadth*100:.1f}bp" if breadth else "N/A"
        st.markdown(make_sensor_card("Market Breadth (1mo)", br_val, "🩺", "#9b59b6", "Różnica w sile między RSP a SPY."), unsafe_allow_html=True)
        # 3. Fear & Greed
        fng = macro.get("Crypto_FearGreed")
        st.markdown(make_sensor_card("Crypto Fear & Greed", f"{fng}", "🪙", "#27ae60", "Nastroje na rynku aktywów cyfrowych."), unsafe_allow_html=True)

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


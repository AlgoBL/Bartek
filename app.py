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

def draw_mini_gauge(title, value, min_val, max_val, invert=False, suffix=""):
    # Invert means: high value is GOOD (e.g. Breadth, Sentiment)
    # Default: high value is BAD (e.g. VIX, Stress, Yield Curve)
    
    if value is None:
        return st.empty()
    
    # Calculate color based on threat level
    norm_val = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    
    # Premium Palette
    P_GREEN = "#00e676"
    P_YELLOW = "#ffea00"
    P_RED = "#ff1744"
    
    if invert:
        # High is GOOD
        if norm_val > 0.65: c = P_GREEN
        elif norm_val < 0.35: c = P_RED
        else: c = P_YELLOW
    else:
        # High is BAD
        if norm_val > 0.65: c = P_RED
        elif norm_val < 0.35: c = P_GREEN
        else: c = P_YELLOW

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': suffix, 'font': {'size': 20, 'color': 'white', 'family': "Inter"}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "#444", 'tickfont': {'size': 8}},
            'bar': {'color': c},
            'bgcolor': "#0d0e12",
            'borderwidth': 1,
            'bordercolor': "#333",
            'steps': [
                {'range': [min_val, (max_val-min_val)*0.35 + min_val], 'color': "rgba(0, 230, 118, 0.05)" if not invert else "rgba(255, 23, 68, 0.05)"},
                {'range': [(max_val-min_val)*0.65 + min_val, max_val], 'color': "rgba(255, 23, 68, 0.05)" if not invert else "rgba(0, 230, 118, 0.05)"}
            ]
        }
    ))
    fig.update_layout(height=130, margin=dict(l=15, r=15, t=15, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig

def get_vanguard_report(score, macro, geo_report):
    sent = geo_report.get("compound_sentiment", 0)
    cycle, _, _, _ = determine_business_cycle(macro)
    
    if score > 70:
        return "ALARM: Wysokie ryzyko systemowe. Dark Pools i VIX wskazują na kaskadową zmienność. Rekomendacja: Obrona kapitału.", "#e74c3c"
    elif score < 35 and sent > 0.1:
        return "STATUS: Rynek w silnym reżimie Risk-On. Płynność wspiera wzrosty. Rekomendacja: Ekspansja w Risky Sleeve.", "#2ecc71"
    else:
        return f"STATUS: Reżim mieszany. Faza {cycle}. Rynek szuka kierunku przy stabilnych warunkach finansowych.", "#3498db"

def home():
    st.markdown(apply_styling(), unsafe_allow_html=True)
    
    # Moderate top padding (lowered content)
    st.markdown("<style>div.block-container{padding-top:5rem;}</style>", unsafe_allow_html=True)

    # Handle Legacy Navigation (force_navigate)
    if "force_navigate" in st.session_state:
        target = st.session_state.pop("force_navigate")
        if target == "📉 Symulator":
            st.switch_page("pages/1_Symulator.py")
        elif target == "⚡ Stress Test":
            st.switch_page("pages/3_Stress_Test.py")

    with st.spinner("Synchronizacja terminala V9.4..."):
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
    
    # --- 1. MAIN GAUGES (AT THE TOP) ---
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.plotly_chart(draw_regime_radar(score), use_container_width=True)
    with col_t2:
        phase, desc, icon, color = determine_business_cycle(macro)
        st.markdown(f"""
        <div style='background-color: #1a1c23; padding: 25px; border-radius: 15px; text-align: center; border: 1px solid #333; height: 280px; display: flex; flex-direction: column; justify-content: center;'>
            <h1 style='font-size: 50px; margin: 0;'>{icon}</h1>
            <h2 style='color: {color}; margin-top: -5px; font-size: 20px;'>{phase}</h2>
            <p style='color: #888; font-size: 13px; margin-top: 5px; line-height: 1.2;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    
    # --- 2. 4-PILLAR GRID ---
    p1, p2, p3, p4 = st.columns(4)
    
    with p1:
        st.markdown("<h4 style='text-align: center; color: #e74c3c; font-size: 16px; margin-bottom: 20px;'>Stress & Volatility</h4>", unsafe_allow_html=True)
        # Bond Vol
        bv = macro.get("Bond_Vol_Proxy")
        if bv:
            st.markdown("", help="MOVE Proxy: Zmienność rynku obligacji skarbowych. Skoki zmienności długu wyprzedzają panikę na EQ.")
            st.plotly_chart(draw_mini_gauge("Bond Vol (MOVE)", bv, 5, 30, invert=False, suffix="%"), use_container_width=True)
        # TED
        ted = macro.get("FRED_TED_Spread")
        if ted:
            st.markdown("", help="TED Spread: Zaufanie międzybankowe. Wzrost powyżej 0.5 sugeruje kryzys płynności dolara.")
            st.plotly_chart(draw_mini_gauge("TED Spread", ted, 0, 1.0, invert=False), use_container_width=True)
        # GEX
        gex = macro.get("total_gex_billions")
        if gex is not None:
            st.markdown("", help="GEX: Pozycjonowanie dealerów opcyjnych. Ujemny GEX = wysoka zmienność (Short Gamma).")
            st.plotly_chart(draw_mini_gauge("Dark Pool GEX", gex, -10, 10, invert=True, suffix="B"), use_container_width=True)

    with p2:
        st.markdown("<h4 style='text-align: center; color: #3498db; font-size: 16px; margin-bottom: 20px;'>Macro & Policy</h4>", unsafe_allow_html=True)
        # FCI
        fci = macro.get("FRED_Financial_Stress_Index")
        if fci is not None:
            st.markdown("", help="Financial Stress Index: Agregat stresu rynkowego od Fed. Powyżej 0 oznacza restrykcyjne warunki.")
            st.plotly_chart(draw_mini_gauge("Financial Stress", fci, -2, 5, invert=False), use_container_width=True)
        # YC
        yc = macro.get("Yield_Curve_Spread", 0)
        st.markdown("", help="Yield Curve: Różnica 10Y-2Y. Inwersja (<0) historycznie zwiastuje recesję.")
        st.plotly_chart(draw_mini_gauge("Yield Curve", yc, -1.0, 3.0, invert=True, suffix="%"), use_container_width=True)
        # Real Yield
        ry = macro.get("FRED_Real_Yield_10Y")
        if ry is not None:
            st.markdown("", help="Real 10Y Yield: Rentowność po inflacji. Wysoki realny koszt pieniądza uderza w aktywa ryzykowne.")
            st.plotly_chart(draw_mini_gauge("Real 10Y Yield", ry, -1.0, 3.0, invert=False, suffix="%"), use_container_width=True)

    with p3:
        st.markdown("<h4 style='text-align: center; color: #2ecc71; font-size: 16px; margin-bottom: 20px;'>Real Economy</h4>", unsafe_allow_html=True)
        # BDRY
        bdry = macro.get("Baltic_Dry")
        if bdry:
            st.markdown("", help="Baltic Dry Index: Koszt frachtu morskiego. Puls globalnego handlu surowcami.")
            st.plotly_chart(draw_mini_gauge("Baltic Dry", bdry, 0, 4000, invert=True), use_container_width=True)
        # Copper
        cu = macro.get("Copper")
        if cu:
            st.markdown("", help="Dr. Copper: Miedź jako barometr przemysłu. Rosnące ceny sugerują ekspansję gospodarczą.")
            st.plotly_chart(draw_mini_gauge("Dr. Copper", cu, 2.0, 6.0, invert=True, suffix="$"), use_container_width=True)
        # Claims
        claims = macro.get("FRED_Initial_Jobless_Claims")
        if claims:
            st.markdown("", help="Jobless Claims: Nowe wnioski o zasiłek. Nagły skok to wczesny sygnał pękania rynku pracy.")
            st.plotly_chart(draw_mini_gauge("Jobless Claims", claims/1000, 150, 400, invert=False, suffix="k"), use_container_width=True)

    with p4:
        st.markdown("<h4 style='text-align: center; color: #f1c40f; font-size: 16px; margin-bottom: 20px;'>Sent. & Breadth</h4>", unsafe_allow_html=True)
        # Sentiment
        sent = geo_report.get("compound_sentiment", 0)
        st.markdown("", help="News Sentiment: Analiza AI nagłówków globalnych. Negatywny wynik sugeruje rosnący niepokój.")
        st.plotly_chart(draw_mini_gauge("News NLP", sent, -1.0, 1.0, invert=True), use_container_width=True)
        # Breadth
        breadth = macro.get("Breadth_Momentum")
        if breadth:
            st.markdown("", help="Market Breadth: Siła RSP (Equal Weight) vs SPY. Wysokie wartości = szeroka, zdrowa hossa.")
            st.plotly_chart(draw_mini_gauge("Breadth (bp)", breadth*10000, -200, 200, invert=True), use_container_width=True)
        # F&G
        fng = macro.get("Crypto_FearGreed")
        if fng:
            st.markdown("", help="Crypto Fear & Greed Index: Sentyment rynku krypto jako proxy dla spekulacyjnego risk-on.")
            st.plotly_chart(draw_mini_gauge("Fear & Greed", fng, 0, 100, invert=True), use_container_width=True)

    st.write("")
    st.divider()

    # --- 3. BOTTOM: INTELLIGENCE REPORT ---
    st.markdown(f"""
    <div style='background-color: #0d0e12; padding: 18px; border-radius: 12px; border-left: 6px solid {report_color}; margin-top: 10px; border: 1px solid #333;'>
        <p style='margin: 0; color: white; font-size: 17px; line-height: 1.5;'><b>Raport:</b> {report_text}</p>
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


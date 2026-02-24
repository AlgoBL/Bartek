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
    
    # 3. Sentiment & Defcon
    sent = geo_report.get("compound_sentiment", 0.0)
    score -= sent * 15.0
    
    # 4. Economy
    jobless = macro.get("FRED_Initial_Jobless_Claims", 250000)
    if jobless > 300000: score += 10.0
    
    hy = macro.get("FRED_HY_Spread", 4.0)
    if hy > 6.0: score += 15.0
    elif hy < 4.0: score -= 5.0
    
    # 5. Płynność i Grawitacja (V8.5)
    m2_yoy = macro.get("FRED_M2_YoY_Growth")
    if m2_yoy is not None:
        if m2_yoy < 0: score += 15.0
        elif m2_yoy > 4.0: score -= 10.0
        
    real_yield = macro.get("FRED_Real_Yield_10Y")
    if real_yield is not None:
        if real_yield > 2.0: score += 10.0
        elif real_yield < 0.5: score -= 10.0
        
    breadth = macro.get("Breadth_Momentum")
    if breadth is not None:
        if breadth < -0.03: score += 10.0 # Rynek ciągnie tylko promil spółek (niezdrowy trend)
        elif breadth > 0.01: score -= 5.0
    
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
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )
    return fig

def make_sensor_card(title, value, icon, color, desc):
    return f"""
    <div style='background-color: #1a1c23; padding: 20px; border-radius: 12px; border-top: 4px solid {color}; border-bottom: 1px solid #333; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.3);'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
            <h4 style='margin: 0; color: #e0e0e0; font-size: 16px;'>{title}</h4>
            <span style='font-size: 24px;'>{icon}</span>
        </div>
        <h2 style='margin: 0; color: {color}; font-size: 28px; font-weight: bold;'>{value}</h2>
        <p style='color: #888; font-size: 12px; margin-top: 8px; line-height: 1.3;'>{desc}</p>
    </div>
    """

def home():
    st.markdown(apply_styling(), unsafe_allow_html=True)
    
    # Handle Legacy Navigation (force_navigate)
    if "force_navigate" in st.session_state:
        target = st.session_state.pop("force_navigate")
        if target == "📉 Symulator":
            st.switch_page("pages/1_Symulator.py")
        elif target == "⚡ Stress Test":
            st.switch_page("pages/3_Stress_Test.py")

    with st.spinner("Oko Saurona kalibruje sensory globalne (TheOracle)..."):
        try:
            macro, geo_report = fetch_control_center_data()
        except Exception as e:
            st.error(f"Błąd synchronizacji sensorów: {e}")
            macro, geo_report = {}, {}

    if not macro:
        st.warning("Brak danych makro. Sprawdź połączenie internetowe.")
        return

    score = calculate_regime_score(macro, geo_report)
    
    # --- GÓRNY PANEL ---
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.plotly_chart(draw_regime_radar(score), use_container_width=True)
    with col_t2:
        st.markdown('### 🏭 Zegar Biznesowy')
        phase, desc, icon, color = determine_business_cycle(macro)
        st.markdown(f"""
        <div style='background-color: #1e1e1e; padding: 25px; border-radius: 15px; text-align: center; border: 2px solid {color}'>
            <h1 style='font-size: 60px; margin: 0;'>{icon}</h1>
            <h3 style='color: {color}; margin-top: 10px;'>{phase}</h3>
            <p style='color: #dddddd; font-size: 14px; margin-top: 15px;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    
    # --- ODCZYTY SENSORÓW ---
    st.markdown('### 🎛️ Sensory Taktyczne: Oko Saurona (V8.5)')
    
    # 1. Hydraulika (M2)
    m2_yoy = macro.get("FRED_M2_YoY_Growth")
    if m2_yoy is not None:
        m2_color = "#2ecc71" if m2_yoy > 0 else "#e74c3c"
        m2_desc = "Drukarki włączone (Zalew gotówki, Hossa)" if m2_yoy > 2.0 else ("Zacieśnianie ilościowe (Wysychanie)" if m2_yoy < 0 else "Neutralna Płynność")
        m2_val = f"{m2_yoy:.2f}% r/r"
    else:
        m2_color, m2_desc, m2_val = "#7f8c8d", "Brak danych z FRED", "N/A"

    # 2. Grawitacja (Real Yield)
    ry = macro.get("FRED_Real_Yield_10Y")
    if ry is not None:
        ry_color = "#e74c3c" if ry > 2.0 else ("#f1c40f" if ry > 0.5 else "#2ecc71")
        ry_desc = "Tani pieniądz napędza spekulację (Risk-On)." if ry <= 0.5 else "Koszt pieniądza ściąga w dół wyceny tech."
        ry_val = f"{ry:.2f}%"
    else:
        ry_color, ry_desc, ry_val = "#7f8c8d", "Brak danych z FRED", "N/A"

    # 3. Defcon Geopolityczny
    sent = geo_report.get("compound_sentiment", 0.0)
    if sent < -0.4: defcon, d_col, d_desc = 1, "#e74c3c", "Globalna destabilizacja. Szoki podażowe. Uciekaj do bezpiecznej bazy sztangi!"
    elif sent < -0.15: defcon, d_col, d_desc = 2, "#e67e22", "Podwyższone ryzyko konfliktów. Ostrzeżenie przed eskalacją."
    elif sent < 0.1: defcon, d_col, d_desc = 3, "#f1c40f", "Niestabilność lokalna. Średni poziom napięć na świecie."
    elif sent < 0.3: defcon, d_col, d_desc = 4, "#3498db", "Zwykły szum geopolityczny. Rynek ignoruje mroczne ryzyka."
    else: defcon, d_col, d_desc = 5, "#2ecc71", "Era pokoju. Geopolityczna nuda nie wywiera presji inflacyjnej."
    defcon_val = f"DEFCON {defcon}"

    # 4. Prześwietlenie (Market Breadth)
    breadth = macro.get("Breadth_Momentum")
    if breadth is not None:
        br_color = "#2ecc71" if breadth > -0.01 else "#e74c3c"
        br_desc = "Zdrowy Byk (Szeroki udział małych/średnich spółek)" if breadth > -0.01 else "Terminalnie Chory Rynek (Hossę ciągną nieliczne Big Techy)"
        br_val = f"{breadth*100:.1f} p.p."
    else:
        br_color, br_desc, br_val = "#7f8c8d", "Brak odczytu RSP vs SPY", "N/A"

    # 5. Doomsday: VIX Curve & Yield Curve
    vix_ts = macro.get("VIX_TS_Ratio", 0.0)
    bkwd = macro.get("VIX_Backwardation", False)
    vix_col = "#e74c3c" if bkwd else "#2ecc71"
    vix_desc = "Cena strachu w krótkim terminie przerosła długi (Panika)" if bkwd else "Rynek ubezpieczeń funkcjonuje normalnie (Contango)"
    
    yc_sp = macro.get("Yield_Curve_Spread", 0.0)
    yc_inv = macro.get("Yield_Curve_Inverted", False)
    yc_col = "#e74c3c" if yc_inv else "#2ecc71"
    yc_desc = "Odwrócona struktura oprocentowania (Kredyt Bankowy zagraża Reccesją)" if yc_inv else "Naturalne premie za długie zamrożenie (Stabilność)"

    # 6. Kasyno: GEX & Skew
    gex = macro.get("total_gex_billions")
    if gex is not None:
        gex_col = "#2ecc71" if gex > 0 else "#e74c3c"
        gex_desc = "Dealerzy kupują spadki by hedgować portfele (Zamrożona zmienność)" if gex > 0 else "GEX na minusie. Dealerzy napędzają wyprzedaże. Rajdy dołujące."
        gex_val = f"${gex:.1f}B"
    else:
        gex_col, gex_desc, gex_val = "#7f8c8d", "Brak danych Opcji", "N/A"
        
    skew = macro.get("skew_index")
    if skew is not None:
        skew_col = "#e74c3c" if skew > 1.2 else ("#f1c40f" if skew > 1.05 else "#2ecc71")
        skew_desc = "Extremalny popyt na OTM Put. Smart Money dyskontuje krach." if skew > 1.2 else "Bilans pomiędzy callami a putami stabilny."
        skew_val = f"{skew:.2f}"
    else:
        skew_col, skew_desc, skew_val = "#7f8c8d", "", "N/A"

    # WIDGET GRID
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(make_sensor_card("🚰 Hydraulika (M2 YoY)", m2_val, "💸", m2_color, m2_desc), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(make_sensor_card("🌡️ Grawitacja (Real Yield)", ry_val, "⚓", ry_color, ry_desc), unsafe_allow_html=True)
    with c2:
        st.markdown(make_sensor_card("🌍 Geopolityka", defcon_val, "☢️", d_col, d_desc), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(make_sensor_card("🩻 Prześwietlenie Hossy", br_val, "🩺", br_color, br_desc), unsafe_allow_html=True)
    with c3:
        st.markdown(make_sensor_card("⚠️ Spread (USA 10Y-2Y)", f"{yc_sp:.2f}%", "⏳", yc_col, yc_desc), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(make_sensor_card("🔥 VIX Term Structure", f"{vix_ts:.2f}", "📉", vix_col, vix_desc), unsafe_allow_html=True)
    with c4:
        st.markdown(make_sensor_card("🌑 Dark Pools GEX", gex_val, "🎰", gex_col, gex_desc), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(make_sensor_card("🛡️ Skew Index", skew_val, "⚖️", skew_col, skew_desc), unsafe_allow_html=True)

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


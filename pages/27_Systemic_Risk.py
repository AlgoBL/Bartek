import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.styling import apply_styling, scicard
from modules.i18n import t

st.markdown(apply_styling(), unsafe_allow_html=True)

st.title("🌐 Systemic Risk & CoVaR")
st.markdown("<p style='color:#6b7280;'>Marginal Expected Shortfall | SRISK | Conditional VaR (CoVaR)</p>", unsafe_allow_html=True)
st.divider()

st.sidebar.markdown("### ⚙️ Parametry Modelu")
confidence_covar = st.sidebar.slider("Poziom UFności CoVaR (%)", 90.0, 99.9, 95.0, 0.5) / 100.0
srisk_threshold = st.sidebar.slider("Próg Katastrofy Rynkowej (SRISK %)", -50, -5, -40, 5) / 100.0
srisk_capital_ratio = st.sidebar.slider("Wymóg Kapitałowy (k, %)", 4.0, 12.0, 8.0, 0.5) / 100.0

st.markdown("""
Systemic Risk ma fundamentalne znaczenie dla wielkich portfeli instytucjonalnych, gdzie kaskadowe wyprzedaże mają charakter zaraźliwy. 
Badamy tu, czy spółki/aktywa, które posiadasz, **zarażą** cały Twój skarbiec, czy też są wystarczająco wyizolowane (np. złoto, gotówka).
""")

# Symulacja danych przykładowych
@st.cache_data
def get_fake_data():
    dates = pd.date_range("2015-01-01", periods=2000, freq="B")
    rng = np.random.default_rng(42)
    # Market return
    market = rng.standard_t(4, len(dates)) * 0.01
    
    # Financials (wysoka korelacja ogonowa)
    bank_a = market * 1.5 + rng.normal(0, 0.015, len(dates))
    bank_b = market * 1.2 + rng.normal(0, 0.01, len(dates))
    
    # Tech
    tech_a = market * 1.1 + rng.normal(0, 0.02, len(dates))
    
    # Safe haven
    gold = market * -0.1 + rng.normal(0, 0.005, len(dates))
    
    df = pd.DataFrame({"Market": market, "Bank_A (JPM)": bank_a, "Bank_B (GS)": bank_b, "Tech_A (AAPL)": tech_a, "Gold": gold})
    df.index = dates
    return df

df = get_fake_data()
assets = [col for col in df.columns if col != "Market"]

st.markdown("### 📊 Marginal Expected Shortfall (MES)")
st.markdown(f"MES mierzy średnią stratę danej instytucji/aktywa, pod warunkiem, że cały rynek jest w stanie kryzysu (np. spadek poniżej 5% wyznaczony przez VaR).")

var_market = np.percentile(df["Market"], (1.0 - confidence_covar) * 100)
market_crash_mask = df["Market"] <= var_market

mes_results = {}
covar_results = {}

for a in assets:
    # MES
    avg_loss_in_crash = df[a][market_crash_mask].mean()
    mes_results[a] = avg_loss_in_crash
    
    # CoVaR: Rynek pod warunkiem ze A ma krach
    var_a = np.percentile(df[a], (1.0 - confidence_covar) * 100)
    a_crash_mask = df[a] <= var_a
    covar_market_given_a = df["Market"][a_crash_mask].mean() if a_crash_mask.sum() > 0 else 0
    # Delta CoVaR = CoVaR_A - unconditional VaR_Market
    delta_covar = covar_market_given_a - var_market
    covar_results[a] = delta_covar

col1, col2 = st.columns(2)

with col1:
    fig_mes = go.Figure(go.Bar(
        x=list(mes_results.keys()), 
        y=[mes * 100 for mes in mes_results.values()],
        marker_color=["#ff1744" if m < -2 else "#f39c12" if m < 0 else "#00e676" for m in mes_results.values()]
    ))
    fig_mes.update_layout(title=f"Marginal Expected Shortfall (MES)<br>przy Market VaR {confidence_covar*100:.1f}%", yaxis_title="Średnia strata (%)", template="plotly_dark", height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_mes, use_container_width=True)

with col2:
    fig_cov = go.Figure(go.Bar(
        x=list(covar_results.keys()), 
        y=[c * 100 for c in covar_results.values()],
        marker_color=["#ff1744" if c < -0.5 else "#00e676" for c in covar_results.values()]
    ))
    fig_cov.update_layout(title="ΔCoVaR (Ile to aktywo zaraża na rynek)", yaxis_title="Dodatkowa strata rynku (%)", template="plotly_dark", height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_cov, use_container_width=True)

st.markdown("### 🏛️ SRISK (Acharya, Engle, Richardson 2012)")
st.markdown("SRISK wskazuje szacowany brak kapitału (capital shortfall) instytucji, jeżeli rynek doświadczy katastrofalnego załamania pod progiem DTD (np. 40% kryzys po 6 miesiącach).")

def render_srisk_chart():
    # Przykladowe obliczenia na hipotetycznym Book Value
    book_values = {"Bank_A (JPM)": 250, "Bank_B (GS)": 100, "Tech_A (AAPL)": 150, "Gold": 0} # w miliardach $
    debt_values = {"Bank_A (JPM)": 2000, "Bank_B (GS)": 900, "Tech_A (AAPL)": 100, "Gold": 0}
    
    srisk_vals = []
    names = []
    
    for a in assets:
        if a == "Gold": continue
        # SRISK = k * Debt - (1-k) * Equity * (1 - LRMES)
        # LRMES (Long-Run MES) aproksymowane przez np. 1 - exp(18 * MES) 
        mes = abs(mes_results[a])
        lrmes = 1 - np.exp(-18 * mes)
        
        eq = book_values[a]
        d = debt_values[a]
        k = srisk_capital_ratio
        
        srisk = k * d - (1 - k) * eq * (1 - lrmes)
        srisk_vals.append(max(0, srisk))
        names.append(a)
    
    fig = go.Figure(go.Bar(x=names, y=srisk_vals, marker_color="#a855f7"))
    fig.update_layout(template="plotly_dark", height=250, yaxis_title="SRISK ($ Mld Capital Shortfall)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

scicard(
    title="Systemic Risk Index (SRISK)",
    icon="💣",
    level0_html="Brak kapitału podczas przedłużającego się krachu",
    chart_fn=render_srisk_chart,
    explanation_md="Im większa jest asymetria zadłużenia (D) względem rynkowej wyceny kapitału (W), i im wyższy jest długoterminowy MES, tym bank/instytucja ma węższy kapitał zapasowy by przetrwać krach i upadnie bez kroplówki rządu.",
    formula_code="SRISK = k·Debts - (1-k)·Equity·(1 - LRMES)",
    reference="Acharya, Engle, Richardson (2012) — 'Capital Shortfall'"
)

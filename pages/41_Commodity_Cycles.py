"""41_Commodity_Cycles.py — Supercykle Surowców: Kondratieff, Juglar, Kitchin"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Commodity Supercycles — Cykle Gospodarcze",
    subtitle="Kondratieff 54Y · Juglar 9Y · Kitchin 3.5Y · Cu/Au Ratio · Copper Leading Indicator",
    icon="⛏️", badge="Cykl Koniunkturalny"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"

tabs = st.tabs([
    "🌊 Supercycle Compositor", "⚙️ Cykle Biznesowe",
    "🥇 Cu/Au Leading Indicator", "📊 Regimes vs Commodities", "🔬 Teoria"
])

# Helper: composite wave
def make_cycle_wave(periods, amplitudes, phases, n_years=100):
    t = np.linspace(0, n_years, n_years * 12)
    wave = np.zeros_like(t)
    for P, A, phi in zip(periods, amplitudes, phases):
        wave += A * np.sin(2 * np.pi * t / P + phi)
    return t, wave

# ── TAB 1: SUPERCYCLE COMPOSITOR ──────────────────────────────────────────
with tabs[0]:
    st.markdown("### 🌊 Compositor Cykli Długich (Kondratieff + Juglar + Kitchin)")
    st.caption("Hierarchia cykli: każdy krótszy jest osadzony wewnątrz dłuższego. Interference pattern tworzy boom/bust.")

    col1, col2 = st.columns([1, 2])
    with col1:
        k_amp = st.slider("Amplitude Kondratieff (54Y)", 0, 100, 60, 5)
        j_amp = st.slider("Amplitude Juglar (9Y)", 0, 60, 30, 5)
        kit_amp = st.slider("Amplitude Kitchin (3.5Y)", 0, 40, 15, 5)
        k_phase = st.slider("Faza K (rok)", 0, 54, 12, 1)
        show_individual = st.checkbox("Pokaż składowe cykle", True)

    t, composite = make_cycle_wave(
        periods=[54, 9, 3.5],
        amplitudes=[k_amp, j_amp, kit_amp],
        phases=[2 * np.pi * k_phase / 54, 0, 0],
        n_years=100
    )
    _, k_wave = make_cycle_wave([54], [k_amp], [2 * np.pi * k_phase / 54], 100)
    _, j_wave = make_cycle_wave([9], [j_amp], [0], 100)
    _, kit_wave = make_cycle_wave([3.5], [kit_amp], [0], 100)

    # Year labels from ~1925
    base_year = 1925
    years = base_year + t

    with col2:
        fig_comp = go.Figure()
        if show_individual:
            fig_comp.add_trace(go.Scatter(x=years, y=k_wave, mode="lines",
                                           name="Kondratieff (54Y)",
                                           line=dict(color="#a855f7", width=1.5, dash="dot")))
            fig_comp.add_trace(go.Scatter(x=years, y=j_wave, mode="lines",
                                           name="Juglar (9Y)",
                                           line=dict(color="#3498db", width=1.5, dash="dash")))
            fig_comp.add_trace(go.Scatter(x=years, y=kit_wave, mode="lines",
                                           name="Kitchin (3.5Y)",
                                           line=dict(color="#ffea00", width=1, dash="dot")))
        fig_comp.add_trace(go.Scatter(x=years, y=composite, mode="lines",
                                       name="COMPOSITE (Supercycle)",
                                       line=dict(color="#00e676", width=3),
                                       fill="tozeroy", fillcolor="rgba(0,230,118,0.06)"))
        fig_comp.add_hline(y=0, line_color="#444")
        fig_comp.add_vline(x=2025, line_dash="solid", line_color="#ff1744",
                           annotation_text="TERAZ", annotation_font_color="#ff1744")
        # Historical peaks
        for yr, evt in [(1929, "Wall St Crash"), (1974, "Oil Crisis"),
                        (2008, "GFC"), (2020, "Covid")]:
            fig_comp.add_vline(x=yr, line_dash="dash", line_color="#555",
                               annotation_text=evt, annotation_font_size=9,
                               annotation_font_color="#aaa")
        fig_comp.update_layout(
            title="Hierarchia Cykli Koniunkturalnych (1925–2025)",
            xaxis=dict(title="Rok", gridcolor="#1c1c2e"),
            yaxis=dict(title="Relatywna Intensywność", gridcolor="#1c1c2e"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=450,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"),
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    # Current phase estimate
    now_idx = np.argmin(np.abs(years - 2025))
    current_val = composite[now_idx]
    trend_val = k_wave[now_idx]
    st.markdown(f"""<div style='{CARD}'>
    <b>Kompozyt 2025:</b> <span style='color:{"#00e676" if current_val > 0 else "#ff1744"};font-size:18px'>{current_val:+.0f}</span>
    | Trend Kondratieff: <span style='color:{"#00e676" if trend_val > 0 else "#ff1744"}'>{trend_val:+.0f}</span><br>
    Faza: <b>{'📈 Ekspansja' if current_val > 20 else '⚠️ Przejście' if current_val > -10 else '📉 Kontrakcja'}</b>
    </div>""", unsafe_allow_html=True)


# ── TAB 2: BUSINESS CYCLES ───────────────────────────────────────────────
with tabs[1]:
    st.markdown("### ⚙️ Cykle Gospodarcze — Typologia")

    cycles_data = {
        "Nazwa": ["Kitchin", "Juglar", "Kuznets", "Kondratieff"],
        "Okres": ["3–4 lata", "7–11 lat", "15–25 lat", "45–60 lat"],
        "Mechanizm": [
            "Cykl zapasów (inventory cycle). Firmy over/under produkują względem popytu.",
            "Cykl inwestycji (capex). CAPEX → overcapacity → recesja → depletion → nowy CAPEX.",
            "Cykl demograficzny + nieruchomości. Generacja kupuje domy w tym samym czasie.",
            "Cykl technologiczny (General Purpose Technology). Para → Elektryczność → IT → AI."
        ],
        "Commodity Impakt": [
            "Metale przemysłowe, energia: szybki cykl.",
            "Stal, cement, przemysłowe: inwestycje infrastrukturalne.",
            "Nieruchomości, drewno, miedź: budowa miast i suburbanizacja.",
            "Wszystkie surowce: 20-letnie supercycle dołki i szczyty."
        ],
        "Narzędzie Detekcji": [
            "PMI, Zapasy, ISM Manufacturing",
            "Capex growth, Business Investment",
            "Housing starts, Real estate cycle",
            "Cu/Gold ratio, Commodity Index / CPI"
        ]
    }
    df_cycles = pd.DataFrame(cycles_data)
    st.dataframe(df_cycles, use_container_width=True, hide_index=True)

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>📅 Fazy Kondratieffa (Supercycle Season)</div>
    <b style='color:#ffea00'>Wiosna (Spring):</b> Ożywienie, nowe technologie, low inflation.<br>
    <b style='color:#00e676'>Lato (Summer):</b> Wysoki wzrost, inflacja, commodity boom.<br>
    <b style='color:#ff8c00'>Jesień (Autumn):</b> Spekulacja finansowa, dług, asset inflation bez CPI.<br>
    <b style='color:#3498db'>Zima (Winter):</b> Deleveraging, deflacja, commodity bust. Złoto trzyma wartość.<br><br>
    <b>~2025:</b> Większość analityków wskazuje na późne Lato lub wczesną Jesień supercyklu.
    </div>""", unsafe_allow_html=True)


# ── TAB 3: CU/AU RATIO ────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 🥇 Copper/Gold Ratio — Leading Economic Indicator")
    st.caption("Cu/Au ratio wyprzedza 10Y Treasury yield i przemysłowe PMI o 6-12 mies. Miedź = ryzyko on, Złoto = bezpieczna przystań.")

    col1, col2 = st.columns([1, 2])
    with col1:
        cu_price = st.number_input("Cena Miedzi (USD/lb)", 1.0, 15.0, 4.50, 0.05)
        au_price = st.number_input("Cena Złota (USD/oz)", 500, 5000, 3200, 10)
        cu_au_ratio = cu_price / (au_price / 400)  # normalize

        current_yield_10y = st.slider("Obecny 10Y Yield (%)", 0.5, 8.0, 4.3, 0.05)

    # Historical synthetic Cu/Au ratio correlation with 10Y
    np.random.seed(77)
    t_hist = np.linspace(0, 25, 300)  # 25 years monthly
    cu_au_hist = 0.15 + 0.08 * np.sin(2 * np.pi * t_hist / 9) + 0.03 * np.random.randn(300)
    yield_hist = 4.0 + 3.0 * np.sin(2 * np.pi * (t_hist - 0.5) / 9) + 0.5 * np.random.randn(300)
    yield_hist = np.clip(yield_hist, 0, 10)

    with col2:
        fig_cg = make_subplots(specs=[[{"secondary_y": True}]])
        fig_cg.add_trace(go.Scatter(x=2000 + t_hist, y=cu_au_hist, mode="lines",
                                     name="Cu/Au Ratio", line=dict(color="#ff8c00", width=2.5)),
                         secondary_y=False)
        fig_cg.add_trace(go.Scatter(x=2000 + t_hist, y=yield_hist, mode="lines",
                                     name="10Y Yield % (lagged)", line=dict(color="#3498db", width=2, dash="dash")),
                         secondary_y=True)
        fig_cg.add_vline(x=2025, line_dash="dash", line_color="#ff1744",
                         annotation_text="TERAZ")
        fig_cg.update_layout(
            title="Copper/Gold Ratio vs 10Y Treasury Yield (Synthetic)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=380,
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=50, r=60, t=50, b=50)
        )
        fig_cg.update_yaxes(title_text="Cu/Au Ratio", gridcolor="#1c1c2e", secondary_y=False)
        fig_cg.update_yaxes(title_text="10Y Yield (%)", gridcolor="#1c1c2e", secondary_y=True)
        st.plotly_chart(fig_cg, use_container_width=True)

    implied_yield = cu_au_ratio * 25  # simplified linear model
    st.markdown(f"""<div style='{CARD}'>
    <b>Cu/Au Ratio:</b> {cu_au_ratio:.4f}<br>
    <b>Implied 10Y Yield (model):</b> {implied_yield:.2f}%
    vs Actual: {current_yield_10y:.2f}%<br>
    Status: <b style='color:{"#ff1744" if implied_yield > current_yield_10y + 0.5 else "#00e676" if implied_yield < current_yield_10y - 0.5 else "#ffea00"}'>
    {"Yield powinien ROSNĄĆ (Cu sygnalizuje reflację)" if implied_yield > current_yield_10y + 0.5
    else "Yield powinien SPADAĆ (Au sygnalizuje safe-haven)" if implied_yield < current_yield_10y - 0.5
    else "W równowadze"}</b>
    </div>""", unsafe_allow_html=True)


# ── TAB 4: REGIMES VS COMMODITIES ─────────────────────────────────────────
with tabs[3]:
    st.markdown("### 📊 Surowce w Różnych Reżimach Makroekonomicznych")

    regime_data = {
        "Reżim": ["Goldilocks (wzrost↑ infl.↓)", "Reflacja (wzrost↑ infl.↑)",
                  "Stagflacja (wzrost↓ infl.↑)", "Deflacja/Recesja (wzrost↓ infl.↓)"],
        "Najlepsze Surowce": ["Metals (Copper, Zinc)", "Oil, Agriculture, Base Metals",
                              "Gold, Oil, Silver", "Gold, US Treasuries (nie surowce)"],
        "Najgorsze Surowce": ["Gold", "Gold, Bonds", "Copper, Industrial", "Oil, Copper, Agriculture"],
        "Equity": ["Growth/Tech", "Value/Energy", "Commodities > Equities", "Cash/Bonds > All"],
        "FX": ["Risk-ON: AUD, NZD, EM", "Commodity FX: NOK, CAD", "Safe Haven: CHF, JPY", "USD, CHF, JPY"],
    }
    df_reg = pd.DataFrame(regime_data)
    st.dataframe(df_reg, use_container_width=True, hide_index=True)

    # Interactive regime selector
    st.markdown("#### 🎯 Twój Aktualny Reżim → Optymalna Alokacja Surowcowa")
    growth_signal = st.select_slider("Sygnał Wzrostu PKB", ["Bardzo Słaby", "Słaby", "Neutralny", "Silny", "Bardzo Silny"], "Silny")
    infl_signal = st.select_slider("Sygnał Inflacji", ["Bardzo Niska", "Niska", "Neutralna", "Wysoka", "Bardzo Wysoka"], "Wysoka")

    growth_positive = growth_signal in ["Silny", "Bardzo Silny"]
    infl_high = infl_signal in ["Wysoka", "Bardzo Wysoka"]

    if growth_positive and not infl_high:
        regime_now = "Goldilocks"
        rec_comm = "🟢 Miedź, Cynk, Aluminium — Industrial metals na hossie."
        color = "#00e676"
    elif growth_positive and infl_high:
        regime_now = "Reflacja"
        rec_comm = "🟠 Ropa, Metale Przemysłowe, Srebro — supercycle commodity."
        color = "#ff8c00"
    elif not growth_positive and infl_high:
        regime_now = "Stagflacja"
        rec_comm = "🟡 Złoto, Ropa — real assets jako hedge. Unikaj Cu."
        color = "#ffea00"
    else:
        regime_now = "Deflacja/Recesja"
        rec_comm = "🔵 Złoto — jedyny chroniony surowiec. Gotówka > surowce."
        color = "#3498db"

    st.markdown(f"""<div style='{CARD}'>
    <b>Detekcja Reżimu:</b> <span style='color:{color};font-size:18px;font-weight:700'>{regime_now}</span><br>
    {rec_comm}
    </div>""", unsafe_allow_html=True)


# ── TAB 5: TEORIA ─────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### 🔬 Teoria Supercykli Surowcowych")
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>📚 Kluczowe Mechanizmy</div>
    <b style='color:#a855f7'>1. Supply Response Lag (Hotelling Rule):</b><br>
    Decyzja o otwarciu kopalni → 5-10 lat zanim osiągnie peak production.<br>
    Surowce mają fundamentalnie długi supply cycle → price surges persistent.<br><br>
    <b style='color:#ffea00'>2. Urbanizacja i Industrializacja:</b><br>
    Każde nowe 1B ludzi wchodzące w konsumpcję miejską = +40% demand na stal/cement/Cu.<br>
    Chiny 2000-2015: +900% copper consumption → supercycle.<br>
    Indie + Afryka = następny cykl.<br><br>
    <b style='color:#00e676'>3. Energy Transition Supercycle (BofA, 2023):</b><br>
    EVs → 4× więcej miedzi niż ICE. Solar → więcej srebra, krzemu, aluminium.<br>
    Dekarbonizacja = commodity supercycle dekady (2023-2035?).<br><br>
    <b style='color:#3498db'>4. Real Asset Premium w Środowisku Inflacyjnym:</b><br>
    r_real &lt; 0 → inwestorzy uciekają w hard assets (złoto, nieruchomości, surowce).<br>
    Ref: Erb & Harvey (2006), Gorton & Rouwenhorst (2004)
    </div>""", unsafe_allow_html=True)

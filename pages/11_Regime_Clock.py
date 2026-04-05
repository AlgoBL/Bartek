"""11_Regime_Clock.py — Zegar Biznesowy (Merrill Lynch Investment Clock)"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from modules.styling import apply_styling
from modules.macro_regime_clock import (
    CLOCK_PHASES, classify_clock_phase, compute_regime_from_macro,
    historical_performance_table, clock_position_coords,
    get_hmm_regime_probabilities, get_transition_matrix
)
from modules.i18n import t
from modules.spectral_analysis import (
    compute_fourier_spectrum, plot_fourier_spectrum,
    compute_wavelet_transform, plot_wavelet_transform
)
from modules.ai.data_loader import load_data
from modules.nss_model import get_simulated_yield_curve, fit_nss, nss_yield_curve
import datetime

st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown("# 🕐 Macro Regime Clock")
st.markdown("*Merrill Lynch Investment Clock — 4 fazy cyklu biznesowego i optymalna alokacja aktywów*")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Sygnały Makro")
    st.markdown("*Dostosuj na podstawie aktualnych danych z Control Center*")
    yield_curve = st.slider("Yield Curve (10Y-2Y, %)", -2.0, 3.0, 0.3, 0.1)
    copper_trend = st.slider("Trend Miedzi (mom %)", -10.0, 10.0, 2.0, 0.5)
    real_yield = st.slider("Real Yield (TIPS 10Y, %)", -3.0, 4.0, 1.5, 0.1)
    hy_spread = st.slider("HY Spread (bps)", 200, 1000, 380, 10)
    manual_override = st.checkbox("Ręczny override fazy", value=False)
    if manual_override:
        manual_phase = st.selectbox("Faza", list(CLOCK_PHASES.keys()))

# Compute phase
macro_snap = {
    "yield_curve_10_2": yield_curve,
    "copper_trend": copper_trend / 100,
    "real_yield": real_yield,
    "hy_oas": hy_spread,
}
result = compute_regime_from_macro(macro_snap)
phase = manual_phase if manual_override else result.get("phase", "Recovery")
phase_info = CLOCK_PHASES[phase]
gdp_sig = result.get("gdp_signal", 0)
infl_sig = result.get("inflation_signal", 0)
confidence = result.get("confidence", 0.5)

# Header
c1, c2, c3, c4 = st.columns(4)
pc = phase_info["color"]
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">FASE CYKLU</div>
        <div style="font-size:28px;">{phase_info['emoji']}</div>
        <div class="metric-value" style="color:{pc};font-size:18px;">{phase}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    cc = "#00e676" if gdp_sig > 0.1 else "#ff1744"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">SYGNAŁ WZROSTU</div>
        <div class="metric-value" style="color:{cc}">{'Przyspieszanie ↑' if gdp_sig > 0 else 'Spowolnienie ↓'}</div>
        <div style="font-size:12px;color:#6b7280">score: {gdp_sig:.2f}</div>
    </div>""", unsafe_allow_html=True)
with c3:
    ic = "#ff1744" if infl_sig > 0.1 else "#00e676"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">SYGNAŁ INFLACJI</div>
        <div class="metric-value" style="color:{ic}">{'Rosnąca ↑' if infl_sig > 0 else 'Malejąca ↓'}</div>
        <div style="font-size:12px;color:#6b7280">score: {infl_sig:.2f}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    sig = phase_info["signal"]
    sc = "#00e676" if sig == "RISK_ON" else "#ff1744" if sig == "RISK_OFF" else "#ffea00"
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">SIGNAL</div>
        <div class="metric-value" style="color:{sc}">{sig}</div>
        <div style="font-size:12px;color:#6b7280">pewność: {confidence:.0%}</div>
    </div>""", unsafe_allow_html=True)

st.divider()
col_chart, col_info = st.columns([1, 1])

with col_chart:
    # Draw the investment clock
    theta_map = {"Recovery": 45, "Overheat": 135, "Stagflation": 225, "Reflation": 315}
    fig = go.Figure()

    # 4 sectors
    sector_colors = {"Recovery": "rgba(0,230,118,0.12)", "Overheat": "rgba(255,234,0,0.12)",
                     "Stagflation": "rgba(255,23,68,0.12)", "Reflation": "rgba(0,204,255,0.12)"}
    for i, (pname, pinfo) in enumerate(CLOCK_PHASES.items()):
        angle_start = i * 90
        angles = np.linspace(np.radians(angle_start), np.radians(angle_start + 90), 30)
        x_arc = np.concatenate([[0], np.cos(angles), [0]])
        y_arc = np.concatenate([[0], np.sin(angles), [0]])
        fig.add_trace(go.Scatter(
            x=x_arc, y=y_arc, fill="toself",
            fillcolor=sector_colors.get(pname, "rgba(100,100,100,0.1)"),
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
            name=pname, showlegend=True,
            mode="lines",
        ))
        # Phase label
        langle = angle_start + 45
        lx = 0.65 * np.cos(np.radians(langle))
        ly = 0.65 * np.sin(np.radians(langle))
        fig.add_annotation(x=lx, y=ly, text=f"{pinfo['emoji']}<br><b>{pname}</b>",
                           showarrow=False, font=dict(size=12, color=pinfo["color"]))

    # Clock axes
    for angle, label in [(0, "GDP+"), (90, "INF+"), (180, "GDP−"), (270, "INF−")]:
        x_end = np.cos(np.radians(angle)) * 1.05
        y_end = np.sin(np.radians(angle)) * 1.05
        fig.add_shape(type="line", x0=0, y0=0, x1=x_end * 0.95, y1=y_end * 0.95,
                      line=dict(color="rgba(255,255,255,0.2)", width=1))
        fig.add_annotation(x=x_end, y=y_end, text=label,
                           showarrow=False, font=dict(size=11, color="#9ca3af"))

    # Zamiast jednego punktu - możemy narysować "Chmurę prawdopodobieństwa" GMM
    probs = get_hmm_regime_probabilities(gdp_sig, infl_sig)
    
    # Dodajemy chmury/bańki prawdopodobieństwa dla każdego reżimu
    regime_centers = {
        "Recovery": (0.6, -0.6),
        "Overheat": (0.6, 0.6),
        "Stagflation": (-0.6, 0.6),
        "Reflation": (-0.6, -0.6)
    }
    
    for r_name, p_val in probs.items():
        if p_val > 0.05:  # Pokaż tylko istotne
            cx, cy = regime_centers[r_name]
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers+text",
                marker=dict(
                    size=p_val * 100,  # Rozmiar bańki proporcjonalny do prawdopodobieństwa
                    color=CLOCK_PHASES[r_name]["color"],
                    opacity=0.4 + (p_val * 0.4),
                    line=dict(width=2, color="white")
                ),
                text=[f"{p_val:.1%}"],
                textposition="middle center",
                textfont=dict(color="white", size=10, weight="bold"),
                name=f"HMM {r_name}",
                showlegend=False,
                hoverinfo="skip"
            ))

    # Bieżąca pozycja jako ostry punkt
    coords = clock_position_coords(gdp_sig, infl_sig)
    fx, fy = coords["x"] * 0.75, coords["y"] * 0.75
    fig.add_trace(go.Scatter(
        x=[0, fx], y=[0, fy],
        mode="lines+markers",
        line=dict(color="white", width=2, dash="dot"),
        marker=dict(size=[4, 10], color=["white", pc], symbol=["circle", "diamond"]),
        name="Obserwacja makro", showlegend=True,
    ))

    fig.update_layout(
        template="plotly_dark", height=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[-1.2, 1.2], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-1.2, 1.2], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_info:
    st.markdown(f"### {phase_info['emoji']} Faza: {phase}")
    st.markdown(f"*{phase_info['description']}*")
    
    # Dodanie informacji o HMM Probabilities
    st.markdown("#### 🧠 Prawdopodobieństwa GMM/HMM")
    for p_name, p_val in sorted(probs.items(), key=lambda item: item[1], reverse=True):
        p_color = CLOCK_PHASES[p_name]["color"]
        st.markdown(f"- **{p_name}**: <span style='color:{p_color};font-weight:bold;'>{p_val:.1%}</span>", unsafe_allow_html=True)
    
    st.markdown("**✅ Rekomendowane aktywa:**")
    for a in phase_info["recommended"]:
        st.markdown(f"  • {a}")
    st.markdown("**❌ Unikaj:**")
    for a in phase_info["avoid"]:
        st.markdown(f"  • {a}")
    st.markdown(f"**Sygnał:** `{phase_info['signal']}`")

st.divider()

# Nowa sekcja: Macierz Przejść HMM
st.markdown("### 🎲 Macierz Przejść Reżimów (Markov Chain Transition Matrix)")
st.markdown("*Przewidywane prawdopodobieństwo przejścia do kolejnej fazy cyklu w następnym kwartale.*")

tm = get_transition_matrix(probs)

# Tworzymy Plotly Heatmap dla macierzy przejść
fig_tm = go.Figure(data=go.Heatmap(
    z=tm.values,
    x=tm.columns,
    y=tm.index,
    colorscale=[[0, "#1c1c2e"], [0.5, "#3498db"], [1, "#00e676"]],
    text=[[f"{val:.1%}" for val in row] for row in tm.values],
    texttemplate="%{text}",
    hoverinfo="skip"
))

fig_tm.update_layout(
    height=300,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", family="Inter"),
    xaxis=dict(title="Następna Faza (t+1)", gridcolor="#1c1c2e"),
    yaxis=dict(title="Obecna Faza (t)", gridcolor="#1c1c2e", autorange="reversed"),
    margin=dict(l=40,r=20,t=30,b=40)
)
st.plotly_chart(fig_tm, use_container_width=True)

st.divider()

st.markdown("### 📊 Historyczne Wyniki Aktywów per Faza (1973-2023)")
perf_df = historical_performance_table()
st.dataframe(perf_df, use_container_width=True, hide_index=True)

st.markdown("### 💡 Jak używać Zegara?")
with st.expander("📚 Metodologia Investment Clock"):
    st.markdown("""
    **Merrill Lynch Investment Clock** (2004) klasyfikuje fazę cyklu na podstawie 2 osi:
    - **Oś X (GDP)**: Wzrost PKB przyspiesza (+) lub zwalnia (−)
    - **Oś Y (Inflacja)**: Inflacja rośnie (+) lub spada (−)

    **4 fazy:**
    | Faza | Wzrost | Inflacja | Premia |
    |------|--------|----------|--------|
    | 🌅 Recovery | ↑ | ↓ | Akcje |
    | ☀️ Overheat | ↑ | ↑ | Surowce |
    | 🌪️ Stagflation | ↓ | ↑ | Złoto/Cash |
    *Źródło: Merrill Lynch "The Investment Clock" (2004); 50 lat danych 1973-2023*
    """)

st.divider()

st.markdown("### 🌊 Analiza Spektralna i Falowa (Głębokie Cykle Rynkowe)")
st.markdown("*Matematyczna dekompozycja historycznych cykli giełdowych przy użyciu Fouriera i falki Morleta (Wavelet Transform).*")

with st.expander("▶️ Uruchom Analizę Spektralną (Głębokie Analizy S&P 500)", expanded=False):
    st.info("Obliczanie transformaty Fouriera i Wavelet dla danych od 2000 roku może zająć kilka sekund.")
    if st.button("Uruchom Analizę (S&P 500)", key="run_spectral"):
        with st.spinner("Pobieranie danych S&P 500 i dekompozycja sygnału..."):
            end_date = datetime.date.today()
            start_date = end_date - datetime.timedelta(days=365*25) # 25 years
            df = load_data(["^GSPC"], start_date=start_date.strftime("%Y-%m-%d"))

            if not df.empty and "^GSPC" in df.columns:
                prices = df["^GSPC"].values
                dates = df.index

                f_periods, f_pxx = compute_fourier_spectrum(prices)
                fig_fourier = plot_fourier_spectrum(f_periods, f_pxx)
                w_widths, w_power = compute_wavelet_transform(prices)
                fig_wavelet = plot_wavelet_transform(w_widths, w_power, dates)

                st.plotly_chart(fig_fourier, use_container_width=True)
                st.markdown("""
                **Interpretacja Fouriera:** Szczyty (pik-i) na powyższym wykresie oznaczają najsilniejsze cykliczności w historii rynku.
                Historycznie, rynki finansowe operują w oparciu o naturalne cykle koniunkturalne: *Kitchina* (zapasy) ok. 4 lat oraz *Juglara* (inwestycje stałe) ok. 9 lat.
                """)
                st.plotly_chart(fig_wavelet, use_container_width=True)
                st.markdown("""
                **Interpretacja Wavelet (Falki Morleta):**
                W przeciwieństwie do Fouriera, transformata falowa pokazuje **kiedy** dany cykl był najsilniejszy.
                Jaśniejsze kolory (żółty/biały) = reżim silnej cykliczności w danym paśmie częstotliwości.
                """)
            else:
                st.error("Błąd pobierania danych dla analizy.")

# ═══════════════════════════════════════════════════════════════════════
#  NOWY PANEL — ENTROPY-BASED REGIME DETECTION (A.3)
#  Ref: Bandt & Pompe (2002) PRL; Dong & Gao (2024) IJFE
# ═══════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("### 🌀 Entropia Permutacyjna — Chaos vs Porządek Rynkowy")
st.markdown(
    "*Information-theoretic wykrywanie reżimów rynkowych bez założeń rozkładu. "
    "Ref: Bandt & Pompe (2002) PRL 88:174102 · Dong & Gao (2024) IJFE*"
)

try:
    from modules.entropy_regime import (
        compute_entropy_regime, plot_entropy_regime,
        plot_entropy_distribution, demo_entropy, permutation_entropy
    )
    from modules.styling import scicard

    # ── Kontrolki sidebara dla entropii ────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🌀 Entropia — Parametry")
        ent_window  = st.slider("Okno PermEn (dni)", 20, 120, 60, 5, key="ent_win")
        ent_m       = st.selectbox("Wymiar osadzenia m", [3, 4, 5], index=0, key="ent_m")
        ent_bull_th = st.slider("Próg Trending (<)", 0.20, 0.60, 0.40, 0.05, key="ent_bull")
        ent_bear_th = st.slider("Próg Chaotic (>)", 0.60, 0.95, 0.75, 0.05, key="ent_bear")
        ent_ticker  = st.text_input("Ticker (puste = demo)", "", key="ent_tick")

    # ── Dane: pobierz lub demo ──────────────────────────────────────────
    price_series = None
    returns      = None

    if ent_ticker.strip():
        with st.spinner(f"Pobieranie danych {ent_ticker.strip()}..."):
            try:
                df_ent = load_data(
                    [ent_ticker.strip()],
                    start_date=(datetime.date.today() - datetime.timedelta(days=365*3)).strftime("%Y-%m-%d")
                )
                col_name = ent_ticker.strip()
                if not df_ent.empty and col_name in df_ent.columns:
                    price_series = df_ent[col_name].dropna()
                    returns      = price_series.pct_change().dropna()
                else:
                    st.warning(f"Brak danych dla {col_name}. Używam demo.")
            except Exception as e:
                st.warning(f"Błąd pobierania: {e}. Używam demo.")

    if returns is None:
        demo = demo_entropy(n=600)
        returns      = demo["returns"]
        price_series = (1 + returns).cumprod() * 100.0
        price_series.name = "Syntetic Index (demo)"

    # ── Oblicz entropię ─────────────────────────────────────────────────
    with st.spinner("Obliczanie Permutation Entropy..."):
        ent_df = compute_entropy_regime(
            returns,
            window=ent_window,
            m=ent_m,
            bear_threshold=ent_bear_th,
            bull_threshold=ent_bull_th,
        )

    # ── Metryki: bieżąca entropia ────────────────────────────────────────
    current_pe = float(ent_df["PermEn"].dropna().iloc[-1]) if len(ent_df) > 0 else 0.0
    current_regime = ent_df["Regime"].dropna().iloc[-1] if len(ent_df) > 0 else "Unknown"
    regime_color = {"Trending": "#00e676", "Neutral": "#ffea00",
                    "Chaotic": "#ff1744"}.get(current_regime, "#888")
    regime_icon  = {"Trending": "📈", "Neutral": "⚡", "Chaotic": "🌪️"}.get(current_regime, "❓")

    # ── SciCard Level 0 header ───────────────────────────────────────────
    level0 = f"""
    <div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap;">
      <div>
        <div style="font-size:10px;color:#6b7280;letter-spacing:1px;">OBECNA ENTROPIA (PermEn)</div>
        <div style="font-size:32px;font-weight:800;color:{regime_color};
                    font-variant-numeric:tabular-nums;">{current_pe:.3f}</div>
      </div>
      <div style="border-left:1px solid #2a2a3a;padding-left:20px;">
        <div style="font-size:10px;color:#6b7280;letter-spacing:1px;">REŻIM</div>
        <div style="font-size:20px;font-weight:700;color:{regime_color};">
          {regime_icon} {current_regime}
        </div>
        <div style="font-size:10px;color:#6b7280;margin-top:2px;">
          Próg Trending &lt;{ent_bull_th:.2f} | Chaotic &gt;{ent_bear_th:.2f}
        </div>
      </div>
      <div style="border-left:1px solid #2a2a3a;padding-left:20px;">
        <div style="font-size:10px;color:#6b7280;letter-spacing:1px;">ROZKŁAD REŻIMÓW</div>
        {"".join(
            f"<div style='font-size:11px;color:{c};margin-top:2px;'>{r}: {n} dni "
            f"({n/len(ent_df)*100:.0f}%)</div>"
            for r, n, c in zip(
                ent_df["Regime"].value_counts().index,
                ent_df["Regime"].value_counts().values,
                [{"Trending":"#00e676","Neutral":"#ffea00","Chaotic":"#ff1744"}.get(r,"#888")
                 for r in ent_df["Regime"].value_counts().index],
            )
        )}
      </div>
    </div>
    """

    # ── Main charts ──────────────────────────────────────────────────────
    def _entropy_charts():
        fig_ent = plot_entropy_regime(ent_df, price_series=price_series)
        st.plotly_chart(fig_ent, use_container_width=True)

        col_dist1, col_dist2 = st.columns([1, 1])
        with col_dist1:
            fig_vio = plot_entropy_distribution(ent_df)
            st.plotly_chart(fig_vio, use_container_width=True)
        with col_dist2:
            # Macierz przejść reżimów
            regimes_seq = ent_df["Regime"].values
            trans = {"Trending": {"Trending": 0, "Neutral": 0, "Chaotic": 0},
                     "Neutral":  {"Trending": 0, "Neutral": 0, "Chaotic": 0},
                     "Chaotic":  {"Trending": 0, "Neutral": 0, "Chaotic": 0}}
            for i in range(len(regimes_seq) - 1):
                frm = regimes_seq[i]
                to  = regimes_seq[i + 1]
                if frm in trans and to in trans[frm]:
                    trans[frm][to] += 1

            cats = ["Trending", "Neutral", "Chaotic"]
            z = []
            for frm in cats:
                row_total = sum(trans[frm].values()) or 1
                z.append([trans[frm].get(to, 0) / row_total for to in cats])

            import plotly.graph_objects as _go
            fig_tm2 = _go.Figure(data=_go.Heatmap(
                z=z, x=cats, y=cats,
                colorscale=[[0,"#1c1c2e"],[0.5,"#00ccff"],[1,"#00e676"]],
                zmin=0, zmax=1,
                text=[[f"{v:.1%}" for v in row] for row in z],
                texttemplate="%{text}",
            ))
            fig_tm2.update_layout(
                title="Macierz Przejść Reżimów Entropii",
                height=280,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Inter"),
                xaxis=dict(title="Następny Reżim", gridcolor="#1c1c2e"),
                yaxis=dict(title="Obecny Reżim", gridcolor="#1c1c2e", autorange="reversed"),
                margin=dict(l=70, r=20, t=40, b=50),
            )
            st.plotly_chart(fig_tm2, use_container_width=True)

    scicard(
        title="Permutation Entropy — Detekcja Reżimów",
        icon="🌀",
        level0_html=level0,
        chart_fn=_entropy_charts,
        explanation_md="""
**Permutation Entropy (PermEn)** mierzy stopień nieprzewidywalności wzorców szeregu czasowego.

| Wartość PermEn | Reżim | Interpretacja |
|---------------|-------|--------------|
| < {bull:.2f} | 📈 Trending | Rynek uporządkowany, silny trend — strategie momentum działają |
| {bull:.2f} – {bear:.2f} | ⚡ Neutral | Mieszany szum i struktura — brak dominującej strategii |
| > {bear:.2f} | 🌪️ Chaotic | Wysoka entropia = chaos rynkowy — sygnał przed krachem lub odbiciem |

**Praktyczne zastosowanie:** Gdy PermEn rośnie gwałtownie przy jednoczesnym
spadku cen → historycznie sygnał krótkoterminowego dna lub wzrostu zmienności.
        """.format(bull=ent_bull_th, bear=ent_bear_th),
        formula_code=(
            "PermEn(m, τ) = −Σ p(π) · log₂ p(π)<br>"
            "gdzie: π = wzorzec rzędowy okna długości m<br>"
            "       p(π) = częstość wzorca π w serii<br>"
            "Znormalizowane: PermEn ← PermEn / log₂(m!)"
        ),
        reference=(
            "Bandt &amp; Pompe (2002) Phys. Rev. Lett. 88:174102 · "
            "Dong &amp; Gao (2024) IJFE — 'Entropy-Based Market Regime Identification'"
        ),
        accent_color="#00ccff",
        key_prefix="regime_clock",
    )

except ImportError as _e:
    st.warning(f"Moduł entropy_regime niedostępny: {_e}. Sprawdź instalację.")
except Exception as _e:
    st.error(f"Błąd panelu entropii: {_e}")

# ─────────────────────────────────────────────────────────────────
# 🆕 KRZYWA DOCHODOWOŚCI (Nelson-Siegel-Svensson)
# ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Modelowanie Krzywej Dochodowości NSS")
st.markdown("Rentowności obligacji skarbowych to kluczowy element makroekonomiczny determinujący fazę *Zegara Inwestycyjnego* (Interest Rates Cycle). Krzywa Nelson-Siegel-Svensson (NSS) dekomponuje całą rentowność (od 3M do 30Y) na komponenty: **Level (Poziom), Slope (Nachylenie), Curvature (Zakrzywienie)**.")

nss_col1, nss_col2 = st.columns([1, 2])

with nss_col1:
    st.info("Dane na żywo (FRED/Treasury) podlegają symulacji na wczesnym etapie. Model znajduje najlepiej dopasowaną krzywą.", icon="ℹ️")
    yc_data = get_simulated_yield_curve()
    st.dataframe(pd.DataFrame({"Zapadalność (Lata)": yc_data['maturities'], "Yield (%)": yc_data['yields'] * 100}).set_index("Zapadalność (Lata)"), use_container_width=True)

with nss_col2:
    nss_params = fit_nss(yc_data['maturities'], yc_data['yields'])
    
    if pd.isna(nss_params.get('error', np.nan)):
        st.warning("Model nie przetworzył dopasowania krzywej.")
    else:
        # Plynna os czasu zeby nakreslic gladka krzywa
        smooth_maturities = np.linspace(0.1, 30.0, 100)
        smooth_yields = nss_yield_curve(smooth_maturities, nss_params['beta0'], nss_params['beta1'], nss_params['beta2'], nss_params['beta3'], nss_params['tau1'], nss_params['tau2'])
        
        fig_nss = go.Figure()
        
        # Punkty empiryczne
        fig_nss.add_trace(go.Scatter(x=yc_data['maturities'], y=yc_data['yields'] * 100, mode='markers', name='Empiryczne Yieldy (Bieżące)', marker=dict(size=10, color="#ffea00")))
        
        # Krzywa NSS
        fig_nss.add_trace(go.Scatter(x=smooth_maturities, y=smooth_yields * 100, mode='lines', name='Dopasowana Krzywa NSS', line=dict(color="#00ccff", width=3)))
        
        fig_nss.update_layout(
             title=f"Krzywa Dochodowości Nelson-Siegel-Svensson (MSE: {nss_params['error']:.4f}%)",
             xaxis_title="Zapadalność (Lata)", yaxis_title="Rentowność (%)",
             template="plotly_dark", height=400,
             paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_nss, use_container_width=True)

st.markdown("""
<details><summary><b>Rozkodowanie skrótów NSS</b></summary>  
<ul>
<li><b>β0 (Level)</b>: Wieloletni trend inflacyjny (ok. 5%).</li>
<li><b>β1 (Slope)</b>: Nachylenie. Ujemne β1 oznacza wyższe rentowności krótkoterminowe (inwersja).</li>
<li><b>β2 / β3 (Curvature)</b>: Brzusiec krzywej (przechodzenie z luźnej w restrykcyjną politykę banku centralnego).</li>
</ul>
</details>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# 🆕 P11: INFLATION REGIME ANALYZER
# ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔥 Indeks Reżimu Inflacyjnego (Inflation Regime Analyzer) 🆕")
st.markdown("Wykrywanie czy rynek 'price-uje' wyższą inflację w przyszłości poprzez analizę spreadów Breakeven (TIPS vs Nominalne). Pozwala to na optymalizację portfela w okresach reflacji lub stagflacji podpowiadając, kiedy Real Assets zyskają.")

ir_col1, ir_col2 = st.columns([1, 2])

with ir_col1:
    tips_yield = st.number_input("Real Yield (US 10Y TIPS %)", -2.0, 5.0, 1.8, step=0.1)
    nom_yield = st.number_input("Nominal Yield (US 10Y %)", 0.0, 10.0, 4.3, step=0.1)
    
    breakeven_inf = nom_yield - tips_yield
    
    inf_regime = "Niski / Dezinflacja"
    inf_color = "green"
    if breakeven_inf > 3.0:
        inf_regime = "Bardzo Wysoki (⚠️ Overheating)"
        inf_color = "red"
    elif breakeven_inf > 2.2:
        inf_regime = "Podwyższony (Stagflation/Reflation)"
        inf_color = "orange"
        

with ir_col2:
    st.markdown(f"### Oczekiwana Inflacja (Breakeven): <span style='color:{inf_color}'>{breakeven_inf:.2f}%</span>", unsafe_allow_html=True)
    st.markdown(f"**Diagnoza Reżimu Przez Rynek:** {inf_regime}")
    
    if breakeven_inf > 2.5:
        st.warning("Gdy rynek oczekuje wysokiej inflacji, obligacje nominalne stają się 'martwym balastem' w portfelu. Wymień je na: **TIPS (Obligacje indeksowane), Złoto, Surowce, Spółki Value z dużą mocą cenotwórczą.**")
    elif breakeven_inf < 1.5:
        st.success("Oczekiwania inflacyjne ugrzęzły. Idealne środowisko dla klasycznych aktywów wzrostowych. Zwiększ wagi w: **S&P 500, Nasdaq, Długoterminowe Obligacje Nominalne.**")
    else:
        st.info("Inflacja w granicach celu Fed. Utrzymuj optymalną, bazową alokację HRP / Black-Litterman.")
    
    # Simple bullet points info
    st.markdown("""
    **Sygnały Skorelowane:**
    * ✅ Miedź do Złota (Wskaźnik ekspansji popytu z Chin)
    * ✅ Krzywa Dochodowości (Stroma = Reflacja)
    * ⚠️ Korelacja Akcje-Obligacje: powyżej zera uderza w fundusze Risk Parity (60/40) i w tradycyjny portfel Raya Dalio.
    """)


"""36_Portfolio_Insurance.py — CPPI/TIPP Portfolio Insurance"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Portfolio Insurance — CPPI / TIPP",
    subtitle="Constant Proportion Portfolio Insurance · Time Invariant Protection · Cushion Dynamics",
    icon="🛡️", badge="Capital Protection"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"

tabs = st.tabs(["🛡️ CPPI Simulator", "⏱️ TIPP (Time-Invariant)", "📊 Porównanie Strategii", "🧮 Teoria"])

# ── TAB 1: CPPI ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### 🛡️ CPPI — Constant Proportion Portfolio Insurance")
    st.caption("Black & Jones (1987). Dynamicznie alokuje między aktywem ryzykownym a bezpiecznym by chronić floor kapitału.")

    col1, col2 = st.columns([1, 2])
    with col1:
        cppi_V0 = st.number_input("Kapitał początkowy (V₀)", 10000, 10_000_000, 100_000, 10_000)
        cppi_floor_pct = st.slider("Floor (% kapitału, chroniony minimum)", 50, 95, 80) / 100
        cppi_m = st.slider("Mnożnik m (agresywność CPPI)", 1.0, 10.0, 4.0, 0.5)
        cppi_mu = st.slider("Oczekiwany zwrot aktywu ryzykownego (% rocznie)", -5.0, 25.0, 8.0, 0.5) / 100
        cppi_sigma = st.slider("Zmienność aktywu ryzykownego (% rocznie)", 5.0, 60.0, 20.0, 0.5) / 100
        cppi_rf = st.slider("Stopa wolna od ryzyka (% rocznie)", 0.0, 8.0, 4.0, 0.1) / 100
        cppi_T = st.slider("Horyzont (lata)", 1, 20, 5)
        cppi_n = cppi_T * 252
        np.random.seed(st.number_input("Seed (dla powtarzalności)", 1, 9999, 42, 1))

    floor_value = cppi_V0 * cppi_floor_pct

    # Symulacja GBM
    dt = 1 / 252
    shocks = np.random.normal(0, 1, cppi_n)
    risky_returns = (cppi_mu - 0.5 * cppi_sigma**2) * dt + cppi_sigma * np.sqrt(dt) * shocks
    risky_price = np.exp(np.cumsum(risky_returns))
    risky_price = np.insert(risky_price, 0, 1.0)

    # CPPI algorithm
    V = cppi_V0
    risky_series = [V]
    safe_series = [0.0]
    floor_series = [floor_value]
    cushion_series = [V - floor_value]

    for i in range(cppi_n):
        cushion = V - floor_value
        if cushion <= 0:
            cushion = 0.0
        risky_alloc = min(cppi_m * cushion, V)
        safe_alloc = V - risky_alloc

        V_new = risky_alloc * (1 + risky_returns[i]) + safe_alloc * (1 + cppi_rf * dt)
        V = max(V_new, floor_value * 0.99)  # slight floor breach allowed

        risky_series.append(risky_alloc)
        safe_series.append(safe_alloc)
        floor_series.append(floor_value)
        cushion_series.append(max(0, V - floor_value))

    V_cppi = [cppi_V0]
    V_temp = cppi_V0
    for i in range(cppi_n):
        cushion = V_temp - floor_value
        cushion = max(0, cushion)
        risky_alloc = min(cppi_m * cushion, V_temp)
        safe_alloc = V_temp - risky_alloc
        V_temp = risky_alloc * (1 + risky_returns[i]) + safe_alloc * (1 + cppi_rf * dt)
        V_temp = max(V_temp, floor_value * 0.99)
        V_cppi.append(V_temp)

    t_axis = np.linspace(0, cppi_T, cppi_n + 1)
    buy_hold = cppi_V0 * risky_price

    with col2:
        fig = make_subplots(rows=2, cols=1,
                            subplot_titles=["Wartość Portfela CPPI vs Buy&Hold", "Poduszka (Cushion) i Alokacja"],
                            row_heights=[0.6, 0.4], shared_xaxes=True, vertical_spacing=0.12)

        fig.add_trace(go.Scatter(x=t_axis, y=V_cppi, mode="lines",
                                  name="CPPI Portfolio", line=dict(color="#00e676", width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_axis, y=buy_hold, mode="lines",
                                  name="Buy & Hold (Risky)", line=dict(color="#3498db", width=2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_axis, y=[floor_value] * (cppi_n + 1), mode="lines",
                                  name="Floor (Ochrona)", line=dict(color="#ff1744", width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_axis, y=cushion_series, mode="lines",
                                  name="Cushion", line=dict(color="#ffea00", width=2),
                                  fill="tozeroy", fillcolor="rgba(255,234,0,0.08)"), row=2, col=1)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=520,
            legend=dict(orientation="h", y=-0.12),
            margin=dict(l=50, r=20, t=60, b=50)
        )
        fig.update_xaxes(gridcolor="#1c1c2e", title_text="Lata")
        fig.update_yaxes(gridcolor="#1c1c2e")
        st.plotly_chart(fig, use_container_width=True)

    final_cppi = V_cppi[-1]
    final_bh = buy_hold[-1]
    floor_breached = any(v < floor_value * 0.95 for v in V_cppi)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CPPI Końcowa Wartość", f"{final_cppi:,.0f} PLN",
              f"{(final_cppi/cppi_V0-1)*100:+.1f}%")
    m2.metric("Buy & Hold Końcowa", f"{final_bh:,.0f} PLN",
              f"{(final_bh/cppi_V0-1)*100:+.1f}%")
    m3.metric("Floor (Chroniony)", f"{floor_value:,.0f} PLN", f"{cppi_floor_pct*100:.0f}%")
    m4.metric("Naruszenie Flooru", "❌ TAK" if floor_breached else "✅ NIE",
              delta_color="inverse")


# ── TAB 2: TIPP ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### ⏱️ TIPP — Time Invariant Portfolio Protection")
    st.caption("Estep & Kritzman (1988). Floor rośnie razem z wartością portfela — chroni zyski.")

    col1, col2 = st.columns([1, 2])
    with col1:
        tipp_V0 = st.number_input("Kapitał (V₀)", 10000, 10_000_000, 100_000, 10_000, key="tipp_v0")
        tipp_floor_pct = st.slider("Początkowy Floor (%)", 50, 90, 80, key="tipp_f") / 100
        tipp_m = st.slider("Mnożnik m", 1.0, 10.0, 4.0, 0.5, key="tipp_m")
        tipp_mu = st.slider("Zwrot ryzykowny (% r.)", -5.0, 25.0, 8.0, 0.5, key="tipp_mu") / 100
        tipp_sigma = st.slider("Zmienność (% r.)", 5.0, 60.0, 20.0, 0.5, key="tipp_s") / 100
        tipp_rf = st.slider("Stopa RF (% r.)", 0.0, 8.0, 4.0, 0.1, key="tipp_rf") / 100
        tipp_T = st.slider("Horyzont (lata)", 1, 20, 5, key="tipp_T")
        tipp_n = tipp_T * 252

    np.random.seed(42)
    dt = 1 / 252
    shocks2 = np.random.normal(0, 1, tipp_n)
    rr2 = (tipp_mu - 0.5 * tipp_sigma**2) * dt + tipp_sigma * np.sqrt(dt) * shocks2

    # TIPP: floor = max(initial_floor, tipp_floor_pct * max_V_to_date)
    V_tipp = [tipp_V0]
    floor_tipp = [tipp_V0 * tipp_floor_pct]
    V_t = tipp_V0
    max_V = tipp_V0

    for i in range(tipp_n):
        floor_t = tipp_floor_pct * max_V
        cushion = max(0, V_t - floor_t)
        risky = min(tipp_m * cushion, V_t)
        safe = V_t - risky
        V_t = risky * (1 + rr2[i]) + safe * (1 + tipp_rf * dt)
        V_t = max(V_t, floor_t * 0.99)
        max_V = max(max_V, V_t)
        V_tipp.append(V_t)
        floor_tipp.append(floor_t)

    t2 = np.linspace(0, tipp_T, tipp_n + 1)
    rp2 = np.exp(np.cumsum(rr2))
    bh2 = tipp_V0 * np.insert(rp2, 0, 1.0)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=t2, y=V_tipp, name="TIPP Portfolio",
                                   line=dict(color="#a855f7", width=2.5)))
        fig2.add_trace(go.Scatter(x=t2, y=bh2, name="Buy & Hold",
                                   line=dict(color="#3498db", width=2, dash="dash")))
        fig2.add_trace(go.Scatter(x=t2, y=floor_tipp, name="TIPP Floor (rosnący)",
                                   line=dict(color="#ff1744", width=1.5, dash="dot"),
                                   fill="tozeroy", fillcolor="rgba(255,23,68,0.05)"))
        fig2.update_layout(
            title="TIPP — Dynamiczny Floor Chroniący Zyski",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=420,
            xaxis=dict(title="Lata", gridcolor="#1c1c2e"),
            yaxis=dict(title="Wartość (PLN)", gridcolor="#1c1c2e"),
            legend=dict(orientation="h", y=-0.15),
            margin=dict(l=50, r=20, t=50, b=60)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>🔄 TIPP vs CPPI — Kluczowa Różnica</div>
    <b style='color:#a855f7'>TIPP:</b> Floor = max(initial_floor, τ × max_V_to_date)<br>
    Floor rośnie gdy portfel zyska. Chroni skumulowane zyski.<br><br>
    <b style='color:#00e676'>CPPI:</b> Floor = stały % V₀<br>
    Prostszy, ale nie chroni zysków. Może oddać całą hossę.
    </div>""", unsafe_allow_html=True)


# ── TAB 3: PORÓWNANIE ────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 📊 Porównanie Strategii Ochrony Kapitału")

    n_sim = 500
    T_comp = 5
    n_steps = T_comp * 252
    mu_c, sigma_c, rf_c = 0.08, 0.20, 0.04
    V0_c = 100_000
    floor_c = 0.80
    m_c = 4.0
    dt_c = 1 / 252

    np.random.seed(123)
    all_shocks = np.random.normal(0, 1, (n_sim, n_steps))

    results = {"CPPI": [], "TIPP": [], "Buy_Hold": [], "Bonds": []}

    for sim in range(n_sim):
        rr = (mu_c - 0.5 * sigma_c**2) * dt_c + sigma_c * np.sqrt(dt_c) * all_shocks[sim]
        # Buy & Hold
        bh_final = V0_c * np.exp(np.sum(rr))
        results["Buy_Hold"].append(bh_final)
        # Bonds
        results["Bonds"].append(V0_c * (1 + rf_c) ** T_comp)
        # CPPI
        V_cp = V0_c
        fl_cp = V0_c * floor_c
        for i in range(n_steps):
            cush = max(0, V_cp - fl_cp)
            r_alloc = min(m_c * cush, V_cp)
            V_cp = r_alloc * (1 + rr[i]) + (V_cp - r_alloc) * (1 + rf_c * dt_c)
            V_cp = max(V_cp, fl_cp * 0.99)
        results["CPPI"].append(V_cp)
        # TIPP
        V_tp = V0_c
        mx_tp = V0_c
        for i in range(n_steps):
            fl_tp = floor_c * mx_tp
            cush = max(0, V_tp - fl_tp)
            r_alloc = min(m_c * cush, V_tp)
            V_tp = r_alloc * (1 + rr[i]) + (V_tp - r_alloc) * (1 + rf_c * dt_c)
            V_tp = max(V_tp, fl_tp * 0.99)
            mx_tp = max(mx_tp, V_tp)
        results["TIPP"].append(V_tp)

    fig3 = go.Figure()
    colors_c = {"CPPI": "#00e676", "TIPP": "#a855f7", "Buy_Hold": "#3498db", "Bonds": "#ffea00"}
    for name, vals in results.items():
        fig3.add_trace(go.Histogram(x=vals, name=name, opacity=0.6,
                                     marker_color=colors_c[name], nbinsx=40))
    fig3.add_vline(x=V0_c * floor_c, line_dash="dash", line_color="#ff1744",
                   annotation_text="Floor (80k)", annotation_font_color="#ff1744")
    fig3.update_layout(
        barmode="overlay",
        title=f"Rozkład Wartości Końcowej po {T_comp} Latach ({n_sim} Symulacji Monte Carlo)",
        xaxis=dict(title="Wartość Końcowa (PLN)", gridcolor="#1c1c2e"),
        yaxis=dict(title="Liczba Scenariuszy", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), height=420,
        margin=dict(l=50, r=20, t=60, b=50)
    )
    st.plotly_chart(fig3, use_container_width=True)

    stats_rows = []
    for name, vals in results.items():
        arr = np.array(vals)
        below_floor = (arr < V0_c * floor_c).mean() * 100
        stats_rows.append({
            "Strategia": name,
            "Mediana": f"{np.median(arr):,.0f}",
            "P10 (pessymistyczny)": f"{np.percentile(arr, 10):,.0f}",
            "P90 (optymistyczny)": f"{np.percentile(arr, 90):,.0f}",
            "Poniżej Flooru (%)": f"{below_floor:.1f}%",
            "Sharpe (approx)": f"{(np.mean(arr)/V0_c - 1) / (np.std(arr)/V0_c + 1e-8):.2f}"
        })
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)


# ── TAB 4: TEORIA ─────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 🧮 Teoria — CPPI / TIPP")
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>📐 Matematyka CPPI</div>
    <b>Definicje:</b><br>
    • V(t) = wartość portfela w czasie t<br>
    • F = Floor (chronione minimum)<br>
    • C(t) = Cushion = V(t) - F(t) (poduszka powyżej flooru)<br>
    • m = mnożnik CPPI (leverage factor)<br>
    • E(t) = min(m·C(t), V(t)) = alokacja w aktywo ryzykowne<br><br>
    <b style='color:#ffea00'>Równanie dynamiki:</b><br>
    dV(t) = E(t)·(μdt + σdW) + (V(t) - E(t))·r·dt<br><br>
    <b style='color:#00e676'>Gwarancja Flooru:</b> Przy ciągłym rebalancingu i μ, σ finite,
    V(t) ≥ F dla wszystkich t z prawdopodobieństwem 1.<br>
    W praktyce (dyskretny rebalancing) istnieje gap risk przy skoku cen.
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>📐 Gap Risk (Ryzyko Skoku Cen)</div>
    Przy skoku ceny aktywu ryzykownego o -J:<br>
    V_new = E·(1-J) + (V-E)·(1+r·dt)<br>
    Naruszenie gdy: V_new &lt; F<br>
    ⟺ E·J &gt; Cushion<br>
    ⟺ m·C·J &gt; C<br>
    ⟺ <b style='color:#ff1744'>J &gt; 1/m</b><br><br>
    Dla m=4: jump &gt; 25% niszczy ochronę.<br>
    Dla m=8: jump &gt; 12.5%.<br><br>
    <b style='color:#00e676'>Wniosek:</b> Im wyższy mnożnik m, tym więcej upside,
    ale tym mniejszy jump wystarczy by naruszyć floor.
    </div>""", unsafe_allow_html=True)

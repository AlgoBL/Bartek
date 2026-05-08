"""40_FX_Dynamics.py — Dynamika Walutowa: UIP, Carry Trade, PPP"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="FX Dynamics — Dynamika Walutowa",
    subtitle="UIP · CIP · PPP · Carry Trade + Kelly · REER · Puzzle Walutowe",
    icon="💱", badge="FX Analytics"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"

tabs = st.tabs([
    "💱 UIP / CIP", "🎯 Carry Trade + Kelly",
    "📊 PPP & REER", "🌀 FX Puzzle", "🧮 Kalkulator"
])

# ── TAB 1: UIP / CIP ─────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### 💱 Uncovered vs Covered Interest Rate Parity")

    col1, col2 = st.columns([1, 2])
    with col1:
        r_dom = st.slider("Stopa krajowa r_d (%)", 0.0, 20.0, 5.5, 0.1)
        r_for = st.slider("Stopa zagraniczna r_f (%)", 0.0, 20.0, 4.0, 0.1)
        spot = st.number_input("Kurs Spot S (dom/for)", 0.1, 10.0, 4.25, 0.01)
        tenor_months = st.slider("Tenor (miesiące)", 1, 60, 12)
        T_fwd = tenor_months / 12

    # CIP Forward rate
    fwd_cip = spot * ((1 + r_dom / 100) / (1 + r_for / 100)) ** T_fwd
    cip_basis = (fwd_cip / spot - 1) * 100 - (r_dom - r_for) * T_fwd

    # UIP expected spot
    spot_uip = spot * ((1 + r_for / 100) / (1 + r_dom / 100)) ** T_fwd

    with col2:
        tenors = np.linspace(1/12, 5, 60)
        fwds = spot * ((1 + r_dom / 100) / (1 + r_for / 100)) ** tenors
        uip_spots = spot * ((1 + r_for / 100) / (1 + r_dom / 100)) ** tenors

        fig_ir = go.Figure()
        fig_ir.add_trace(go.Scatter(x=tenors, y=fwds, mode="lines", name="CIP Forward",
                                     line=dict(color="#00e676", width=2.5)))
        fig_ir.add_trace(go.Scatter(x=tenors, y=uip_spots, mode="lines",
                                     name="UIP Expected Spot", line=dict(color="#00ccff", width=2.5, dash="dash")))
        fig_ir.add_hline(y=spot, line_dash="dot", line_color="#aaa",
                         annotation_text=f"Spot = {spot:.4f}")
        fig_ir.add_vline(x=T_fwd, line_dash="dash", line_color="#ffea00",
                         annotation_text=f"T={tenor_months}M")
        fig_ir.update_layout(
            title="CIP Forward vs UIP Expected Spot",
            xaxis=dict(title="Tenor (lata)", gridcolor="#1c1c2e"),
            yaxis=dict(title="Kurs", gridcolor="#1c1c2e"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=380,
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_ir, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Spread r_d - r_f", f"{r_dom - r_for:+.2f}%")
    m2.metric("CIP Forward", f"{fwd_cip:.4f}")
    m3.metric("UIP Expected Spot", f"{spot_uip:.4f}")
    m4.metric("CIP Basis (abberration)", f"{cip_basis:+.4f}%",
              help="Odchylenie od CIP = możliwość arb lub ryzyko funding")

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>📐 Parity Relationships</div>
    <b style='color:#00e676'>CIP (Covered):</b> F = S × (1+r_d)/(1+r_f) — arbitraż deterministyczny, działa ZAWSZE<br>
    <b style='color:#00ccff'>UIP (Uncovered):</b> E[S_T] = S × (1+r_f)/(1+r_d) — oczekiwanie, empirycznie zawodzi<br>
    <b style='color:#ff1744'>Forward Bias Puzzle:</b> Empirycznie r_d > r_f → waluta aprecjuje (nie deprecjonuje!)<br>
    → To jest źródło carry trade premium.
    </div>""", unsafe_allow_html=True)


# ── TAB 2: CARRY TRADE + KELLY ────────────────────────────────────────────
with tabs[1]:
    st.markdown("### 🎯 Carry Trade + Kelly Criterion")
    st.caption("Carry trade = pożycz w walucie nisko-oprocentowanej, zainwestuj w wysoko. Edge = spread. Risk = sudden reversal.")

    col1, col2 = st.columns([1, 2])
    with col1:
        carry_spread = st.slider("Carry Spread (r_high - r_low, %)", 0.5, 15.0, 5.0, 0.25)
        fx_vol = st.slider("FX Zmienność (% rocznie)", 5.0, 40.0, 12.0, 0.5)
        crash_prob = st.slider("Prob. Nagłego Reversal (%/rok)", 0.0, 30.0, 10.0, 1.0) / 100
        crash_size = st.slider("Wielk. Reversal (%, jeśli crash)", 5.0, 50.0, 20.0, 1.0)
        n_years = st.slider("Symulacja (lata)", 1, 20, 5)

    carry_annual = carry_spread / 100
    vol_annual = fx_vol / 100

    # Kelly fraction (simplified Bernoulli carry)
    win_prob = 1 - crash_prob
    win_amount = carry_annual  # annual carry gain
    loss_amount = crash_size / 100  # crash loss
    # Kelly: f* = (p*b - q) / b where b = win/loss ratio
    b = win_amount / loss_amount if loss_amount > 0 else 0
    kelly_frac = (win_prob * b - (1 - win_prob)) / b if b > 0 else 0
    kelly_frac = max(0.0, min(kelly_frac, 1.0))
    half_kelly = kelly_frac / 2

    # Sharpe ratio of carry (ignoring crash)
    sharpe_carry = (carry_annual - 0.5 * vol_annual**2) / vol_annual

    with col2:
        # Simulation paths
        np.random.seed(42)
        n_paths = 200
        n_steps = n_years * 52  # weekly
        dt_w = 1 / 52

        portfolio_ends = []
        for _ in range(n_paths):
            val = 1.0
            for _ in range(n_steps):
                # Normal carry week
                fx_ret = np.random.normal(carry_annual * dt_w,
                                          vol_annual * np.sqrt(dt_w))
                # Random crash
                if np.random.rand() < crash_prob * dt_w:
                    fx_ret -= crash_size / 100
                val *= (1 + fx_ret * kelly_frac)
                val = max(val, 0.001)
            portfolio_ends.append(val)

        pct10 = np.percentile(portfolio_ends, 10)
        pct50 = np.percentile(portfolio_ends, 50)
        pct90 = np.percentile(portfolio_ends, 90)

        fig_ct = go.Figure()
        fig_ct.add_trace(go.Histogram(x=portfolio_ends, nbinsx=40,
                                       marker_color="#00e676", opacity=0.7,
                                       name="Wynik po {n_years}L"))
        fig_ct.add_vline(x=1.0, line_dash="dash", line_color="#aaa",
                         annotation_text="Breakeven")
        fig_ct.add_vline(x=pct50, line_dash="dash", line_color="#00e676",
                         annotation_text=f"Mediana {pct50:.2f}×")
        fig_ct.add_vline(x=pct10, line_dash="dash", line_color="#ff1744",
                         annotation_text=f"P10 {pct10:.2f}×")
        fig_ct.update_layout(
            title=f"Carry Trade z Kelly ({n_paths} symulacji, {n_years}L)",
            xaxis=dict(title="Mnożnik Kapitału", gridcolor="#1c1c2e"),
            yaxis=dict(title="Ścieżki", gridcolor="#1c1c2e"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=360,
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_ct, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Kelly Fraction", f"{kelly_frac:.1%}")
    c2.metric("Half-Kelly (zalecane)", f"{half_kelly:.1%}")
    c3.metric("Carry Sharpe (ex-crash)", f"{sharpe_carry:.2f}")
    c4.metric("Mediana Wynik", f"{pct50:.2f}×")

    if kelly_frac <= 0:
        st.error("❌ Negatywne Kelly — carry NIE ma pozytywnego EV przy tych parametrach.")
    elif kelly_frac < 0.2:
        st.warning("⚠️ Mały Kelly — niewielka przewaga. Używaj ostrożnie.")
    else:
        st.success(f"✅ Kelly = {kelly_frac:.1%} ({half_kelly:.1%} half-Kelly dla ochrony ruin).")


# ── TAB 3: PPP & REER ─────────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 📊 Purchasing Power Parity (PPP) & REER")
    st.caption("PPP: kurs równowagi długoterminowej. REER: Real Effective Exchange Rate vs koszyk walut.")

    col1, col2 = st.columns([1, 2])
    with col1:
        infl_dom = st.slider("Inflacja krajowa (% r.)", 0.0, 20.0, 4.5, 0.1)
        infl_for = st.slider("Inflacja zagraniczna (% r.)", 0.0, 20.0, 2.5, 0.1)
        spot_ppp = st.number_input("Obecny kurs spot", 0.1, 20.0, 4.25, 0.01, key="ppp_s")
        ppp_val = st.number_input("PPP (wartość równowagi)", 0.1, 20.0, 4.00, 0.01)
        years_ppp = st.slider("Horyzont (lata)", 1, 20, 10)

    # PPP implied appreciation
    ppp_rate = (1 + infl_dom / 100) / (1 + infl_for / 100) - 1
    t_arr = np.arange(0, years_ppp + 1)
    ppp_path = ppp_val * ((1 + ppp_rate) ** t_arr)
    deviation = (spot_ppp / ppp_val - 1) * 100

    with col2:
        fig_ppp = go.Figure()
        fig_ppp.add_trace(go.Scatter(x=t_arr, y=ppp_path, mode="lines",
                                      name="PPP Equilibrium Path",
                                      line=dict(color="#00e676", width=2.5)))
        fig_ppp.add_hline(y=spot_ppp, line_dash="dash", line_color="#ff1744",
                          annotation_text=f"Spot = {spot_ppp}")
        fig_ppp.update_layout(
            title="PPP Implied Equilibrium Path",
            xaxis=dict(title="Lata", gridcolor="#1c1c2e"),
            yaxis=dict(title="Kurs", gridcolor="#1c1c2e"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=360,
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_ppp, use_container_width=True)

    dev_color = "#ff1744" if abs(deviation) > 20 else "#f39c12" if abs(deviation) > 10 else "#00e676"
    over_under = "PRZEWARTOŚCIOWANA" if deviation > 0 else "NIEDOWARTOŚCIOWANA"
    st.markdown(f"""<div style='{CARD}'>
    <b>Odchylenie od PPP:</b> <span style='color:{dev_color};font-size:18px;font-weight:700'>{deviation:+.1f}%</span>
    — Waluta {over_under}<br>
    <b>PPP drift:</b> {ppp_rate*100:+.2f}%/rok (różnica inflacji)<br>
    <i>PPP działa długoterminowo (3-7 lat). Krótkoterminowo rządzi carry i flow.</i>
    </div>""", unsafe_allow_html=True)


# ── TAB 4: FX PUZZLE ──────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 🌀 FX Puzzle — Dlaczego UIP Zawodzi?")

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>📐 UIP i 3 Główne Puzzle</div>
    <b style='color:#ff1744'>1. Forward Bias Puzzle (Fama 1984):</b><br>
    UIP predykuje: E[ΔS] = r_d - r_f. Empirycznie β &lt; 0 — waluta z <i>wyższą</i> stopą <i>aprecjonuje</i>.<br>
    → Dlatego carry trade działa! Ale kumuluje tail risk (sudden stops).<br><br>
    <b style='color:#ffea00'>2. PPP Puzzle (Rogoff 1996):</b><br>
    PPP działa na 15-20 letnim horyzoncie, half-life deviacji ≈ 3-5 lat.<br>
    Za wolno by być efektywne jako trading signal.<br><br>
    <b style='color:#00ccff'>3. Disconnect Puzzle (Meese-Rogoff 1983):</b><br>
    Żaden model makroekonomiczny nie bije Random Walk w predykcji FX na &lt;1Y horyzoncie.<br>
    → Kurs jest "disconnected" od fundamentals krótkoterminowo.<br><br>
    <b style='color:#a855f7'>Risk Premium Explanation (Lustig & Verdelhan 2007):</b><br>
    Carry return to kompensata za <b>global risk factor</b> (crash risk, liquidity spiral).<br>
    High-carry currencies correlate with global volatility shocks (risk-off).
    </div>""", unsafe_allow_html=True)

    # Carry Trade Crash Risk Visualization
    st.markdown("#### Carry Trade vs Risk-Off Events (symulowany)")
    np.random.seed(10)
    n_d = 500
    # Normal regime
    vix_sim = np.abs(np.random.randn(n_d)) * 5 + 15
    carry_ret_sim = 0.02/252 - 0.00015 * (vix_sim - 15) + np.random.randn(n_d) * 0.003
    # Spike VIX crashes
    spike_idx = np.random.choice(n_d, 10, replace=False)
    vix_sim[spike_idx] += np.random.exponential(20, 10)
    carry_ret_sim[spike_idx] -= np.random.exponential(0.03, 10)

    fig_carry_risk = go.Figure()
    fig_carry_risk.add_trace(go.Scatter(
        x=vix_sim, y=carry_ret_sim * 100,
        mode="markers",
        marker=dict(color=carry_ret_sim * 100,
                    colorscale=[[0, "#ff1744"], [0.5, "#888"], [1, "#00e676"]],
                    size=6, opacity=0.7),
        text=[f"VIX={v:.0f}, Ret={r*100:.2f}%" for v, r in zip(vix_sim, carry_ret_sim)],
        hoverinfo="text", name="Carry vs VIX"
    ))
    fig_carry_risk.update_layout(
        title="Carry Trade Return vs VIX (Crash Risk Correlation)",
        xaxis=dict(title="VIX Level", gridcolor="#1c1c2e"),
        yaxis=dict(title="Daily Carry Return (%)", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), height=380,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    st.plotly_chart(fig_carry_risk, use_container_width=True)


# ── TAB 5: KALKULATOR FX ──────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### 🧮 FX Multi-Kalkulator")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Forward Rate Calculator (CIP)**")
        fx_spot = st.number_input("Spot S", 0.01, 100.0, 4.25, 0.01, key="fx_sp")
        fx_rd = st.slider("r_d (%)", 0.0, 20.0, 5.5, 0.1, key="fx_rd")
        fx_rf = st.slider("r_f (%)", 0.0, 20.0, 4.0, 0.1, key="fx_rf_c")
        fx_tenor = st.slider("Tenor (dni)", 1, 1825, 365, 1)
        T_fx = fx_tenor / 365
        fwd = fx_spot * ((1 + fx_rd/100) / (1 + fx_rf/100)) ** T_fx
        swap_points = (fwd - fx_spot) * 10000
        st.metric("Forward Rate", f"{fwd:.5f}")
        st.metric("Swap Points (pips)", f"{swap_points:+.1f}")
        st.metric("Implied Annual FX Return", f"{(fwd/fx_spot-1)/T_fx*100:+.2f}%")

    with c2:
        st.markdown("**Position Sizing (FX Trading)**")
        acct_size = st.number_input("Rozmiar konta (PLN)", 1000, 10_000_000, 100_000, 1000)
        risk_pct = st.slider("Ryzyko na trade (%)", 0.1, 5.0, 1.0, 0.1)
        stop_pips = st.slider("Stop-Loss (pipy)", 5, 500, 50)
        pip_value = st.number_input("Wartość 1 pipa (PLN per 1 lot)", 1.0, 100.0, 10.0, 0.1)
        lot_size = 100_000

        risk_amount = acct_size * risk_pct / 100
        lots = risk_amount / (stop_pips * pip_value)
        units = lots * lot_size

        st.metric("Kwota Ryzyka", f"{risk_amount:,.0f} PLN")
        st.metric("Rozmiar Pozycji", f"{lots:.2f} loty")
        st.metric("Nominalna Ekspozycja", f"{units:,.0f} units")
        st.metric("Leverage Efektywny", f"{units/acct_size:.1f}×")

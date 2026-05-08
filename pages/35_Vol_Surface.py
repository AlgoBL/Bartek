"""35_Vol_Surface.py — Powierzchnia Zmienności Implikowanej"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq
from modules.styling import apply_styling, module_header
from modules.i18n import t

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Vol Surface — Powierzchnia Zmienności",
    subtitle="Implied Vol Surface · SVI Parametryzacja · Local Vol (Dupire) · Term Structure",
    icon="🌋", badge="Options Analytics"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"

# ── Black-Scholes helpers ──────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type == "call" else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol(market_price, S, K, T, r, option_type="call"):
    intrinsic = max(0, S - K) if option_type == "call" else max(0, K - S)
    if market_price <= intrinsic + 1e-8:
        return np.nan
    try:
        return brentq(lambda sig: bs_price(S, K, T, r, sig, option_type) - market_price,
                      1e-4, 5.0, maxiter=100)
    except Exception:
        return np.nan

# ── SVI Parametrization ────────────────────────────────────────────────────
def svi_vol(k, a, b, rho, m, sigma_svi):
    """Raw SVI: w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))"""
    w = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma_svi**2))
    return np.sqrt(np.maximum(w, 0))

# ─────────────────────────────────────────────────────────────────────────
st.markdown("---")

tabs = st.tabs([
    "🌋 Vol Surface 3D", "📈 SVI Parametryzacja",
    "📉 Term Structure", "⚡ Local Vol (Dupire)", "🧮 BS Kalkulator"
])

# ── TAB 1: VOL SURFACE 3D ──────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### 🌋 Powierzchnia Zmienności Implikowanej (3D)")
    st.caption("Volatility Smile + Term Structure = Vol Surface. Odchylenie od płaskiej powierzchni to premia za ryzyko ogona.")

    col1, col2 = st.columns([1, 3])
    with col1:
        S0 = st.number_input("Cena Spot (S)", 50, 10000, 500, 10)
        r_rate = st.slider("Stopa wolna od ryzyka (%)", 0.0, 10.0, 4.5, 0.1) / 100
        surface_mode = st.selectbox("Typ powierzchni", ["Equity (Skew)", "FX (Smile)", "Flat (BS)"])

    strikes_pct = np.linspace(0.70, 1.30, 13)  # moneyness
    tenors = np.array([0.083, 0.25, 0.5, 1.0, 1.5, 2.0])  # in years

    # Generate synthetic vol surface
    K_grid, T_grid = np.meshgrid(strikes_pct, tenors)
    IV_grid = np.zeros_like(K_grid)

    for i, T in enumerate(tenors):
        for j, k_pct in enumerate(strikes_pct):
            log_m = np.log(k_pct)  # log-moneyness
            atm_vol = 0.18 + 0.04 * np.sqrt(T)
            if surface_mode == "Equity (Skew)":
                skew = -0.10 * log_m / np.sqrt(T + 0.1)
                smile = 0.05 * log_m**2 / (T + 0.1)
                IV_grid[i, j] = atm_vol + skew + smile
            elif surface_mode == "FX (Smile)":
                smile = 0.08 * log_m**2 / (T + 0.1)
                IV_grid[i, j] = atm_vol + smile
            else:
                IV_grid[i, j] = atm_vol
            IV_grid[i, j] = max(0.05, IV_grid[i, j])

    fig_surf = go.Figure(data=[go.Surface(
        x=strikes_pct * S0,
        y=tenors * 12,
        z=IV_grid * 100,
        colorscale=[[0, "#0a0b0e"], [0.3, "#1a237e"], [0.6, "#00e676"], [1.0, "#ff1744"]],
        showscale=True,
        colorbar=dict(title="IV (%)", tickfont=dict(color="white")),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
        hovertemplate="Strike: %{x:.0f}<br>Tenor: %{y:.1f}M<br>IV: %{z:.1f}%<extra></extra>"
    )])
    fig_surf.update_layout(
        title=f"Vol Surface — {surface_mode}",
        scene=dict(
            xaxis=dict(title="Strike", gridcolor="#2a2a3a", backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(title="Tenor (mies.)", gridcolor="#2a2a3a", backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(title="Implied Vol (%)", gridcolor="#2a2a3a", backgroundcolor="rgba(0,0,0,0)"),
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2))
        ),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white", family="Inter"),
        height=520, margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig_surf, use_container_width=True)

    with st.expander("📘 Sticky Strike vs Sticky Delta"):
        st.markdown(f"""<div style='{CARD}'>
        <b style='color:#00e676'>Sticky Strike:</b> IV dla danego strike K pozostaje stała gdy S się rusza.
        Implikuje: przy wzroście S → ATM vol spada (jak rynek akcji).<br><br>
        <b style='color:#00ccff'>Sticky Delta:</b> IV dla danego delta = const gdy S się rusza.
        IV "podróżuje" razem z rynkiem. Typowe dla FX.<br><br>
        <b style='color:#ffea00'>Sticky Moneyness (Sticky Log-Strike):</b> Kompromis — IV stała dla log(K/F).
        </div>""", unsafe_allow_html=True)


# ── TAB 2: SVI ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### 📈 SVI — Stochastic Volatility Inspired Parametrization")
    st.caption("Gatheral (2004) — jedyny model z gwarancją braku arbitrażu butterflowego dla danego tenoru.")

    col_svi1, col_svi2 = st.columns([1, 2])
    with col_svi1:
        svi_a = st.slider("a (ATM poziom)", -0.1, 0.2, 0.04, 0.005)
        svi_b = st.slider("b (Wing slope)", 0.01, 0.5, 0.15, 0.01)
        svi_rho = st.slider("ρ (skew/asymetria)", -0.99, 0.99, -0.30, 0.01)
        svi_m = st.slider("m (ATM shift)", -0.5, 0.5, 0.0, 0.01)
        svi_sigma = st.slider("σ_svi (smoothing)", 0.01, 0.5, 0.10, 0.01)

    log_strikes = np.linspace(-0.5, 0.5, 200)
    iv_svi = svi_vol(log_strikes, svi_a, svi_b, svi_rho, svi_m, svi_sigma)

    fig_svi = go.Figure()
    fig_svi.add_trace(go.Scatter(x=log_strikes, y=iv_svi * 100, mode="lines",
                                  line=dict(color="#00e676", width=3), name="SVI Vol Smile"))
    fig_svi.add_vline(x=0, line_dash="dash", line_color="#ffea00", annotation_text="ATM")
    fig_svi.update_layout(
        title=f"SVI Vol Smile (a={svi_a:.3f}, b={svi_b:.2f}, ρ={svi_rho:.2f})",
        xaxis=dict(title="Log-Moneyness ln(K/F)", gridcolor="#1c1c2e"),
        yaxis=dict(title="Implied Vol (%)", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), height=380,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    st.plotly_chart(fig_svi, use_container_width=True)

    # Arbitrage check
    wings_ok = svi_b * (1 + abs(svi_rho)) < 4
    st.markdown(f"""<div style='{CARD}'>
    <b>Butterfly Arbitrage Check (Lee 2004):</b>
    b·(1+|ρ|) = {svi_b*(1+abs(svi_rho)):.3f} {'≤ 4 ✅ Brak arbitrażu' if wings_ok else '> 4 ❌ ARBITRAŻ!'}
    </div>""", unsafe_allow_html=True)


# ── TAB 3: TERM STRUCTURE ──────────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 📉 Term Structure Zmienności ATM")
    st.caption("Jak zmienia się vol ATM w funkcji terminu wygaśnięcia? Contango vs Backwardation.")

    col_ts1, col_ts2 = st.columns([1, 2])
    with col_ts1:
        ts_mode = st.selectbox("Scenariusz rynkowy", ["Normal (Contango)", "Stress (Backwardation)", "Hump"])
        spot_vol = st.slider("Vol spot (1M, %)", 5.0, 60.0, 18.0, 0.5)

    t_arr = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])
    t_labels = ["1M", "2M", "3M", "6M", "9M", "1Y", "18M", "2Y"]

    if ts_mode == "Normal (Contango)":
        vols = spot_vol + np.sqrt(t_arr) * 3
    elif ts_mode == "Stress (Backwardation)":
        vols = spot_vol - np.sqrt(t_arr) * 4
        vols = np.maximum(vols, spot_vol * 0.5)
    else:  # Hump
        vols = spot_vol + 8 * np.exp(-3 * t_arr) * t_arr

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=t_labels, y=vols, mode="lines+markers",
                                line=dict(color="#00e676" if ts_mode != "Stress (Backwardation)" else "#ff1744",
                                          width=3), marker=dict(size=10), name="ATM Vol"))
    fig_ts.add_hline(y=spot_vol, line_dash="dash", line_color="#aaa",
                     annotation_text=f"Spot Vol {spot_vol:.0f}%")
    fig_ts.update_layout(
        title=f"ATM Term Structure — {ts_mode}",
        xaxis=dict(title="Tenor", gridcolor="#1c1c2e"),
        yaxis=dict(title="Implied Vol (%)", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), height=380,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    backw = ts_mode == "Stress (Backwardation)"
    st.markdown(f"""<div style='{CARD}'>
    <b>Status:</b> {'🔴 BACKWARDATION — Rynek wycenia ostry stres krótkoterminowy. Opcje short-dated drogie.' 
    if backw else '✅ CONTANGO — Normalna struktura. Vol rośnie z tenorem (dłuższy czas = więcej niepewności).'}
    </div>""", unsafe_allow_html=True)


# ── TAB 4: LOCAL VOL (DUPIRE) ─────────────────────────────────────────────
with tabs[3]:
    st.markdown("### ⚡ Local Volatility (Dupire 1994)")
    st.caption("σ_local(K,T) = f(IV surface). Model deterministyczny który replikuje dowolny market-consistent smile.")

    st.markdown(f"""<div style='{CARD}'>
    <b style='color:#00e676'>Wzór Dupire (Continuum limit):</b><br>
    <code>σ²_loc(K,T) = [∂C/∂T + rK·∂C/∂K] / [½·K²·∂²C/∂K²]</code><br><br>
    Gdzie C(K,T) to cena call z rynku.<br><br>
    <b>Właściwości:</b><br>
    • Jedinyny model który dokładnie fituje <i>dowolny</i> dany smile<br>
    • Problem: dynamika lokalnej vol nie odpowiada obserwowanej rynkowej<br>
    • Rozwiązanie: Stochastic Local Vol (SLV) = Heston × Dupire
    </div>""", unsafe_allow_html=True)

    # Numeric local vol from SVI surface
    K_range = np.linspace(0.7, 1.3, 50)
    T_range = np.array([0.25, 0.5, 1.0, 2.0])
    local_vol_grid = np.zeros((len(T_range), len(K_range)))

    for i, T in enumerate(T_range):
        for j, k_pct in enumerate(K_range):
            lm = np.log(k_pct)
            base_iv = 0.18 + 0.04 * np.sqrt(T)
            skew = -0.10 * lm / np.sqrt(T + 0.1)
            smile = 0.05 * lm**2 / (T + 0.1)
            iv = max(0.05, base_iv + skew + smile)
            # Local vol approx: σ_local ≈ σ_imp * √(T/τ) simplified
            local_vol_grid[i, j] = iv * (1 + 0.3 * lm**2)

    fig_lv = go.Figure()
    colors_lv = ["#00e676", "#00ccff", "#ffea00", "#ff1744"]
    for i, T in enumerate(T_range):
        fig_lv.add_trace(go.Scatter(
            x=K_range * 500, y=local_vol_grid[i] * 100,
            mode="lines", name=f"T={T}Y",
            line=dict(color=colors_lv[i], width=2.5)
        ))
    fig_lv.update_layout(
        title="Local Vol σ_loc(K,T) — Approximate Dupire",
        xaxis=dict(title="Strike K", gridcolor="#1c1c2e"),
        yaxis=dict(title="Local Vol (%)", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), height=380,
        margin=dict(l=50, r=20, t=50, b=50)
    )
    st.plotly_chart(fig_lv, use_container_width=True)


# ── TAB 5: BS KALKULATOR ──────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### 🧮 Black-Scholes Kalkulator + Greeks")

    c1, c2, c3 = st.columns(3)
    with c1:
        bs_S = st.number_input("Spot S", 1.0, 10000.0, 500.0, 1.0)
        bs_K = st.number_input("Strike K", 1.0, 10000.0, 500.0, 1.0)
    with c2:
        bs_T = st.slider("Czas do wygaśnięcia (lat)", 0.01, 3.0, 0.25, 0.01)
        bs_r = st.slider("Stopa r (%)", 0.0, 15.0, 4.5, 0.1) / 100
    with c3:
        bs_sigma = st.slider("Implied Vol σ (%)", 1.0, 150.0, 20.0, 0.5) / 100
        bs_type = st.selectbox("Typ", ["call", "put"])

    d1 = (np.log(bs_S / bs_K) + (bs_r + 0.5 * bs_sigma**2) * bs_T) / (bs_sigma * np.sqrt(bs_T))
    d2 = d1 - bs_sigma * np.sqrt(bs_T)
    price = bs_price(bs_S, bs_K, bs_T, bs_r, bs_sigma, bs_type)

    delta = norm.cdf(d1) if bs_type == "call" else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (bs_S * bs_sigma * np.sqrt(bs_T))
    theta_call = (-(bs_S * norm.pdf(d1) * bs_sigma) / (2 * np.sqrt(bs_T))
                  - bs_r * bs_K * np.exp(-bs_r * bs_T) * norm.cdf(d2))
    theta = theta_call if bs_type == "call" else theta_call + bs_r * bs_K * np.exp(-bs_r * bs_T)
    vega = bs_S * norm.pdf(d1) * np.sqrt(bs_T) / 100
    rho = bs_K * bs_T * np.exp(-bs_r * bs_T) * (norm.cdf(d2) if bs_type == "call" else -norm.cdf(-d2)) / 100

    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("💰 Cena", f"{price:.4f}")
    mc2.metric("Δ Delta", f"{delta:.4f}")
    mc3.metric("Γ Gamma", f"{gamma:.6f}")
    mc4.metric("Θ Theta/d", f"{theta/365:.4f}")
    mc5.metric("ν Vega/1%", f"{vega:.4f}")
    mc6.metric("ρ Rho/1%", f"{rho:.4f}")

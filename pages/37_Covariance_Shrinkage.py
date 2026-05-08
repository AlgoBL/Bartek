"""37_Covariance_Shrinkage.py — Ledoit-Wolf & Oracle Shrinkage Estimators"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Covariance Shrinkage — Ledoit-Wolf",
    subtitle="Shrinkage Estimators · James-Stein · Oracle Approximating · Condition Number",
    icon="🔬", badge="Estimation Risk"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"

st.markdown(f"""<div style='{CARD}'>
<b style='color:#00e676;font-size:15px'>Problem Estimation Risk:</b><br>
Klasyczna próbkowa macierz kowariancji Σ̂ jest <b>skrajnie wrażliwa na szum</b> gdy T (obserwacje) ≈ p (aktywa).
Ledoit & Wolf (2003) udowodnili, że optymalne Σ̂_shrink = (1-α)·Σ̂_sample + α·F minimalizuje MSE.<br>
F = target matrix (np. diagonalna), α = optymalny współczynnik shrinkage.<br><br>
<b style='color:#ffea00'>Efekt:</b> Skrajne eigenvalues są "wciągane" do środka → lepsza kondycja → stabilniejsze wagi portfela.
</div>""", unsafe_allow_html=True)

tabs = st.tabs([
    "🔬 Shrinkage Demo", "📊 Eigenvalue Spectrum",
    "🏆 Portfolio Stabilność", "📐 Teoria"
])

# ── TAB 1: SHRINKAGE DEMO ─────────────────────────────────────────────────
with tabs[0]:
    col1, col2 = st.columns([1, 2])
    with col1:
        n_assets = st.slider("Liczba aktywów (p)", 5, 100, 30)
        n_obs = st.slider("Liczba obserwacji (T)", 30, 500, 60)
        corr_strength = st.slider("Siła korelacji (ρ avg)", 0.0, 0.9, 0.3, 0.05)
        noise_level = st.slider("Szum / Estymacji Błąd (σ_noise)", 0.0, 0.5, 0.1, 0.05)
        target_type = st.selectbox("Target Matrix F", ["Scaled Identity", "Constant Correlation", "Diagonal"])

    np.random.seed(42)
    # Generate true corr matrix (block + noise)
    true_cov = np.full((n_assets, n_assets), corr_strength)
    np.fill_diagonal(true_cov, 1.0)
    true_cov = true_cov + noise_level * np.random.randn(n_assets, n_assets)
    true_cov = (true_cov + true_cov.T) / 2
    # Make PD
    eigvals = np.linalg.eigvalsh(true_cov)
    if eigvals.min() < 0.01:
        true_cov += (-eigvals.min() + 0.01) * np.eye(n_assets)

    # Generate returns
    L = np.linalg.cholesky(true_cov + 1e-8 * np.eye(n_assets))
    returns = (np.random.randn(n_obs, n_assets) @ L.T)
    returns_df = pd.DataFrame(returns, columns=[f"A{i+1}" for i in range(n_assets)])

    # Estimators
    sample_cov = np.cov(returns.T)
    lw = LedoitWolf().fit(returns)
    oas = OAS().fit(returns)

    lw_cov = lw.covariance_
    oas_cov = oas.covariance_

    alpha_lw = float(lw.shrinkage_)
    alpha_oas = float(oas.shrinkage_)

    # Condition numbers
    cond_sample = np.linalg.cond(sample_cov)
    cond_lw = np.linalg.cond(lw_cov)
    cond_oas = np.linalg.cond(oas_cov)

    with col2:
        # Heatmap porównanie
        fig_heat = go.Figure()
        fig_heat.add_trace(go.Heatmap(
            z=sample_cov, colorscale="RdBu_r", zmid=0,
            showscale=True, colorbar=dict(title="Cov", tickfont=dict(color="white")),
            name="Sample Cov"
        ))
        fig_heat.update_layout(
            title=f"Próbkowa Macierz Kowariancji (p={n_assets}, T={n_obs})",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=380,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Condition# Sample", f"{cond_sample:.0f}", help="Im wyższy, tym niestabilniejsza inwersja")
    m2.metric(f"Condition# Ledoit-Wolf (α={alpha_lw:.2f})", f"{cond_lw:.0f}")
    m3.metric(f"Condition# OAS (α={alpha_oas:.2f})", f"{cond_oas:.0f}")

    improvement_lw = (cond_sample - cond_lw) / cond_sample * 100
    st.progress(min(1.0, improvement_lw / 100), text=f"Ledoit-Wolf poprawia kondycję o {improvement_lw:.0f}%")


# ── TAB 2: EIGENVALUE SPECTRUM ─────────────────────────────────────────────
with tabs[1]:
    st.markdown("### 📊 Marchenko-Pastur — Eigenvalue Cleaning")
    st.caption("Teoria Random Matrix (Marchenko & Pastur 1967): eigenvalues próbkowej cov zawierają szum. Signal = powyżej λ_max.")

    col1, col2 = st.columns([1, 2])
    with col1:
        mp_p = st.slider("Liczba aktywów p", 10, 200, 50)
        mp_T = st.slider("Liczba obserwacji T", 50, 1000, 100)
        mp_var = st.slider("Wariancja σ²", 0.5, 3.0, 1.0, 0.1)

    q = mp_T / mp_p  # T/p ratio
    # Marchenko-Pastur bounds
    lambda_minus = mp_var * (1 - 1 / np.sqrt(q))**2
    lambda_plus = mp_var * (1 + 1 / np.sqrt(q))**2

    # MP density
    lambda_range = np.linspace(max(0.001, lambda_minus * 0.5), lambda_plus * 1.2, 500)
    def mp_density(lam, q, sigma2):
        lam_p = sigma2 * (1 + 1/np.sqrt(q))**2
        lam_m = sigma2 * (1 - 1/np.sqrt(q))**2
        if lam < lam_m or lam > lam_p:
            return 0.0
        return (q / (2 * np.pi * sigma2 * lam)) * np.sqrt((lam_p - lam) * (lam - lam_m))

    mp_pdf = np.array([mp_density(l, q, mp_var) for l in lambda_range])

    # Empirical eigenvalues from random matrix
    np.random.seed(7)
    X_rand = np.random.randn(mp_T, mp_p) * np.sqrt(mp_var)
    S_emp = X_rand.T @ X_rand / mp_T
    eigs_emp = np.linalg.eigvalsh(S_emp)

    with col2:
        fig_mp = go.Figure()
        fig_mp.add_trace(go.Histogram(x=eigs_emp, nbinsx=50,
                                       histnorm="probability density",
                                       name="Empiryczne Eigenvalues",
                                       marker_color="#3498db", opacity=0.6))
        fig_mp.add_trace(go.Scatter(x=lambda_range, y=mp_pdf, mode="lines",
                                     name="Marchenko-Pastur (teoria)",
                                     line=dict(color="#ff1744", width=3)))
        fig_mp.add_vline(x=lambda_plus, line_dash="dash", line_color="#ffea00",
                         annotation_text=f"λ_max={lambda_plus:.2f} (Signal Cutoff)")
        fig_mp.update_layout(
            title=f"Marchenko-Pastur Spectrum (q=T/p={q:.1f})",
            xaxis=dict(title="Eigenvalue", gridcolor="#1c1c2e"),
            yaxis=dict(title="Density", gridcolor="#1c1c2e"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=380,
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_mp, use_container_width=True)

    n_signal = (eigs_emp > lambda_plus).sum()
    st.markdown(f"""<div style='{CARD}'>
    <b style='color:#00e676'>Analiza Spektralna:</b><br>
    • λ_max (szum) = <b>{lambda_plus:.3f}</b><br>
    • Eigenvalues powyżej λ_max (sygnał): <b style='color:#ffea00'>{n_signal}</b> z {mp_p}<br>
    • q = T/p = {q:.2f} {'✅ T >> p, dobra estymacja' if q > 5 else '⚠️ T ≈ p, dominuje szum Random Matrix!'}<br><br>
    Eigenvalues <b>poniżej</b> λ_max to czysty szum statystyczny — należy je <b>zerować lub zastępować</b> medianą.
    </div>""", unsafe_allow_html=True)


# ── TAB 3: PORTFOLIO STABILITY ─────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 🏆 Wpływ Shrinkage na Stabilność Wag Portfela")
    st.caption("Problem praktyczny: minimalizacja wariancji ze złą macierzą kowariancji daje ekstremalne, niestabilne wagi.")

    np.random.seed(42)
    p = 20
    T_roll = 60

    # Symulacja: rolująca estymacja wag min-var
    n_periods = 100
    weights_sample_all = []
    weights_lw_all = []

    for period in range(n_periods):
        returns_sim = np.random.randn(T_roll, p) * 0.01
        S = np.cov(returns_sim.T)
        lw_est = LedoitWolf().fit(returns_sim)
        S_lw = lw_est.covariance_

        # Min-variance weights
        def min_var_weights(cov):
            try:
                inv = np.linalg.inv(cov + 1e-6 * np.eye(p))
                ones = np.ones(p)
                raw = inv @ ones
                return raw / raw.sum()
            except Exception:
                return np.ones(p) / p

        weights_sample_all.append(min_var_weights(S))
        weights_lw_all.append(min_var_weights(S_lw))

    w_s = np.array(weights_sample_all)
    w_lw = np.array(weights_lw_all)

    # Turnover
    turnover_s = np.abs(np.diff(w_s, axis=0)).sum(axis=1).mean()
    turnover_lw = np.abs(np.diff(w_lw, axis=0)).sum(axis=1).mean()

    # Weight std
    std_s = w_s.std(axis=0).mean()
    std_lw = w_lw.std(axis=0).mean()

    fig_w = go.Figure()
    for j in range(min(5, p)):
        fig_w.add_trace(go.Scatter(y=w_s[:, j], mode="lines", name=f"A{j+1} Sample",
                                    line=dict(width=1.5, dash="dash"), opacity=0.6))
    for j in range(min(5, p)):
        fig_w.add_trace(go.Scatter(y=w_lw[:, j], mode="lines", name=f"A{j+1} L-W",
                                    line=dict(width=1.5)))
    fig_w.update_layout(
        title="Rolujące Wagi Min-Variance: Sample (--) vs Ledoit-Wolf (—)",
        xaxis=dict(title="Okres", gridcolor="#1c1c2e"),
        yaxis=dict(title="Waga", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), height=380,
        showlegend=True, margin=dict(l=50, r=20, t=50, b=50)
    )
    st.plotly_chart(fig_w, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Turnover (Sample)", f"{turnover_s:.4f}", help="Niższy = stabilniejszy")
    c2.metric("Avg Turnover (LW)", f"{turnover_lw:.4f}",
              f"{(turnover_lw/turnover_s-1)*100:+.0f}%")
    c3.metric("Weight Std (Sample)", f"{std_s:.4f}")
    c4.metric("Weight Std (LW)", f"{std_lw:.4f}",
              f"{(std_lw/std_s-1)*100:+.0f}%")


# ── TAB 4: TEORIA ─────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 📐 Teoria Shrinkage Estymatorów")
    st.markdown(f"""<div style='{CARD}'>
    <b style='color:#00e676;font-size:14px'>James-Stein (1961) — Fundamenty</b><br>
    Klasyczny MLE (sample mean) jest <b>inadmissible</b> w p≥3 wymiarach.<br>
    Skurczony estymator: <b>μ̂_JS = (1 - (p-2)/(T·||x̄||²)) · x̄</b> zawsze przewyższa MLE w MSE.<br><br>
    <b style='color:#00ccff;font-size:14px'>Ledoit & Wolf (2003, 2004)</b><br>
    Σ̂_LW = (1-α)·Σ̂_sample + α·μ̂·I<br>
    α* = analytycznie wyznaczony optimal shrinkage intensity.<br>
    Minimalizuje Expected Loss E[||Σ̂ - Σ_true||²_F].<br><br>
    <b style='color:#a855f7;font-size:14px'>Oracle Approximating Shrinkage (OAS, Chen 2010)</b><br>
    Ulepszona wersja L-W — mniejsze bias przy małym T.<br>
    Dobrze sprawdza się gdy T/p < 2.<br><br>
    <b style='color:#ffea00;font-size:14px'>Nonlinear Shrinkage (Ledoit & Wolf 2020)</b><br>
    Różny α dla każdego eigenvalue osobno.<br>
    Asymptotycznie optymalne (Oracle). Wymaga biblioteki <code>nlshrink</code>.
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style='{CARD}'>
    <b style='color:#00e676;font-size:14px'>Kiedy używać którego:</b><br><br>
    | Sytuacja | Metoda |<br>
    |---|---|<br>
    | T >> p (T/p > 10) | Sample Cov jest OK |<br>
    | T ≈ p (2 < T/p < 10) | Ledoit-Wolf |<br>
    | T ≈ p i mały portfel | OAS |<br>
    | T < p | Ledoit-Wolf + Marchenko cleaning |<br>
    | Najlepiej (offline) | Nonlinear Shrinkage |
    </div>""", unsafe_allow_html=True)

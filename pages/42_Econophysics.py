"""42_Econophysics.py — Ekonofizyka: Ising Model, Sornette, Power Laws"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Ekonofizyka — Fizyka Rynków",
    subtitle="Ising Model Rynku · Sornette Crash Model · Power Law Wealth · Percolacja",
    icon="⚛️", badge="Econophysics"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"

tabs = st.tabs([
    "🧲 Ising Model Rynku", "💥 Sornette Crash Predictor",
    "📊 Dystrybucja Bogactwa", "🕸️ Perkolacja & Kryzysy", "📐 Teoria"
])

# ── TAB 1: ISING MODEL ────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### 🧲 Ising Model Rynku (Lux & Marchesi 1999)")
    st.caption("Każdy trader = spin (+1 bull / -1 bear). Interakcja sąsiednia → faza kolektywna → crash.")

    col1, col2 = st.columns([1, 2])
    with col1:
        grid_size = st.slider("Rozmiar siatki (N×N)", 10, 60, 30)
        temperature = st.slider("Temperatura T (szum decyzji)", 0.1, 5.0, 2.0, 0.1)
        J_coupling = st.slider("Sprzężenie J (siła naśladownictwa)", 0.1, 3.0, 1.0, 0.1)
        n_steps = st.slider("Kroki symulacji (MCMC)", 100, 2000, 500, 100)
        ext_field = st.slider("Zewnętrzne pole H (news/momentum)", -2.0, 2.0, 0.0, 0.1)
        run_ising = st.button("▶ Symuluj Ising", use_container_width=True)

    # Ising MCMC (Metropolis)
    np.random.seed(42)
    spins = np.random.choice([-1, 1], size=(grid_size, grid_size))

    if run_ising:
        beta = 1.0 / max(temperature, 0.01)
        for _ in range(n_steps):
            for __ in range(grid_size * grid_size):
                i, j = np.random.randint(0, grid_size, 2)
                neighbors = (
                    spins[(i - 1) % grid_size, j] + spins[(i + 1) % grid_size, j] +
                    spins[i, (j - 1) % grid_size] + spins[i, (j + 1) % grid_size]
                )
                dE = 2 * spins[i, j] * (J_coupling * neighbors + ext_field)
                if dE < 0 or np.random.rand() < np.exp(-beta * dE):
                    spins[i, j] *= -1

    magnetization = spins.mean()
    bull_pct = (spins == 1).mean() * 100

    with col2:
        fig_ising = go.Figure(go.Heatmap(
            z=spins, colorscale=[[0, "#ff1744"], [0.5, "#1a1c28"], [1, "#00e676"]],
            showscale=False, zmin=-1, zmax=1,
            hovertemplate="Spin: %{z}<extra></extra>"
        ))
        fig_ising.update_layout(
            title=f"Ising Lattice — Magnetyzacja M={magnetization:.3f} (Byki: {bull_pct:.0f}%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=420,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_ising, use_container_width=True)

    phase_color = "#00e676" if magnetization > 0.3 else "#ff1744" if magnetization < -0.3 else "#ffea00"
    phase_name = "BULL MARKET (Ferromagnetyk)" if magnetization > 0.3 else "BEAR MARKET" if magnetization < -0.3 else "CHAOS / Paramagnetyk"
    st.markdown(f"""<div style='{CARD}'>
    <b>Faza Rynkowa:</b> <span style='color:{phase_color};font-size:16px'>{phase_name}</span><br>
    Temperatura krytyczna T_c ≈ {2.27 * J_coupling:.2f} (dla kwadratowej siatki 2D)<br>
    Twoja T={temperature:.1f}: {'POWYŻEJ T_c → losowy rynek (szum)' if temperature > 2.27 * J_coupling else 'PONIŻEJ T_c → naśladownictwo dominuje → bańki i krachy'}<br><br>
    <i>Przy T → T_c: rynek jest na krawędzi chaosu (critical point) — power-law distributions, fat tails.</i>
    </div>""", unsafe_allow_html=True)

    # Price series from magnetization walk
    st.markdown("#### Syntetyczna Seria Cen (z magnetyzacji Isinga)")
    np.random.seed(42)
    beta_sim = 1.0 / max(temperature, 0.01)
    mag_series = []
    sp_sim = np.random.choice([-1, 1], size=(20, 20))
    for _ in range(200):
        for __ in range(400):
            ii, jj = np.random.randint(0, 20, 2)
            nb = (sp_sim[(ii-1)%20, jj] + sp_sim[(ii+1)%20, jj] +
                  sp_sim[ii, (jj-1)%20] + sp_sim[ii, (jj+1)%20])
            dE2 = 2 * sp_sim[ii, jj] * (J_coupling * nb + ext_field)
            if dE2 < 0 or np.random.rand() < np.exp(-beta_sim * dE2):
                sp_sim[ii, jj] *= -1
        mag_series.append(sp_sim.mean())

    price_ising = 100 * np.exp(np.cumsum(np.array(mag_series) * 0.02))
    fig_price_ising = go.Figure(go.Scatter(
        y=price_ising, mode="lines",
        line=dict(color="#00e676", width=2),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.05)"
    ))
    fig_price_ising.update_layout(
        title=f"Syntetyczne Ceny z Ising (T={temperature:.1f}, J={J_coupling:.1f})",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), height=280,
        xaxis=dict(title="Czas", gridcolor="#1c1c2e"),
        yaxis=dict(title="Cena", gridcolor="#1c1c2e"),
        margin=dict(l=50, r=20, t=50, b=50)
    )
    st.plotly_chart(fig_price_ising, use_container_width=True)


# ── TAB 2: SORNETTE CRASH PREDICTOR ───────────────────────────────────────
with tabs[1]:
    st.markdown("### 💥 Sornette Log-Periodic Power Law (LPPL)")
    st.caption("Sornette & Johansen (2001): bubbles exhibit log-periodic oscillations approaching critical time t_c.")

    col1, col2 = st.columns([1, 2])
    with col1:
        tc = st.slider("Czas krytyczny t_c", 50, 150, 100)
        alpha = st.slider("α (power law exponent)", 0.1, 0.9, 0.33, 0.01)
        beta_lp = st.slider("β (oscillation amplitude)", 0.0, 0.5, 0.15, 0.01)
        omega = st.slider("ω (log-frequency)", 3.0, 12.0, 6.36, 0.1)
        phi = st.slider("φ (phase)", 0.0, 6.28, 1.0, 0.1)
        A_param = st.slider("A (baseline)", 5.0, 15.0, 10.0, 0.5)
        B_param = st.slider("B (growth rate)", -2.0, 0.0, -0.5, 0.1)
        C_param = st.slider("C (oscillation scale)", -0.5, 0.5, 0.1, 0.05)

    # LPPL: log(p(t)) = A + B*(tc-t)^alpha * [1 + C*cos(omega*log(tc-t) + phi)]
    t_range = np.linspace(1, tc - 0.5, 200)
    dt = tc - t_range
    lppl = A_param + B_param * dt**alpha * (1 + C_param * np.cos(omega * np.log(dt + 1e-10) + phi))

    with col2:
        fig_lppl = go.Figure()
        fig_lppl.add_trace(go.Scatter(x=t_range, y=lppl, mode="lines",
                                       name="LPPL Signal",
                                       line=dict(color="#ffea00", width=3)))
        fig_lppl.add_vline(x=tc, line_dash="dash", line_color="#ff1744",
                           annotation_text=f"Crash t_c={tc}",
                           annotation_font_color="#ff1744")
        # Add noisy "observed" prices
        noise = np.random.randn(len(t_range)) * 0.3
        fig_lppl.add_trace(go.Scatter(x=t_range, y=lppl + noise, mode="lines",
                                       name="Observed Prices (z szumem)",
                                       line=dict(color="#3498db", width=1.5, dash="dot"),
                                       opacity=0.7))
        fig_lppl.update_layout(
            title="Log-Periodic Power Law (Sornette Bubble Signature)",
            xaxis=dict(title="Czas", gridcolor="#1c1c2e"),
            yaxis=dict(title="log(Cena)", gridcolor="#1c1c2e"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=400,
            legend=dict(x=0.01, y=0.99),
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_lppl, use_container_width=True)

    # Current time indicator
    t_now = st.slider("Twoja pozycja w czasie (t_now)", 1, tc - 1, int(tc * 0.85))
    remaining = tc - t_now
    crash_prob_signal = max(0, 1 - remaining / tc)
    st.markdown(f"""<div style='{CARD}'>
    <b>Czas do t_c:</b> {remaining} jednostek<br>
    <b>Sygnał LPPL:</b> <span style='color:{"#ff1744" if crash_prob_signal > 0.7 else "#ffea00" if crash_prob_signal > 0.4 else "#00e676"}'>
    {crash_prob_signal:.0%}</span> — {'🔴 WYSOKI' if crash_prob_signal > 0.7 else '⚠️ UMIARKOWANY' if crash_prob_signal > 0.4 else '✅ NISKI'} sygnał bańki<br>
    <i>LPPL nie jest predyktorem — to narzędzie detekcji super-eksponencjalnego wzrostu z log-oscylacjami.</i>
    </div>""", unsafe_allow_html=True)


# ── TAB 3: WEALTH DISTRIBUTION ────────────────────────────────────────────
with tabs[2]:
    st.markdown("### 📊 Dystrybucja Bogactwa — Pareto vs Boltzmann-Gibbs")
    st.caption("Bogactwo: ogon Pareto (power law). Dochód z pracy: Boltzmann-Gibbs (exponential). Gini jako miara.")

    col1, col2 = st.columns([1, 2])
    with col1:
        n_agents = st.slider("Liczba agentów", 100, 5000, 1000, 100)
        pareto_alpha_w = st.slider("Pareto α (nierówność)", 1.0, 4.0, 1.5, 0.1,
                                    help="Im niższy, tym bardziej nierówny")
        boltz_temp = st.slider("Boltzmann T (śr. dochód z pracy)", 100, 10000, 2000, 100)
        mix_ratio = st.slider("% Bogactwa w ogonie Pareto", 10, 90, 40)

    np.random.seed(99)
    n_pareto = int(n_agents * mix_ratio / 100)
    n_boltz = n_agents - n_pareto

    w_pareto = np.random.pareto(pareto_alpha_w, n_pareto) * 1000 + 1000
    w_boltz = np.random.exponential(boltz_temp, n_boltz)
    all_wealth = np.concatenate([w_pareto, w_boltz])
    all_wealth = np.sort(all_wealth)

    # Gini
    n_g = len(all_wealth)
    gini = (2 * np.sum(np.arange(1, n_g + 1) * all_wealth) /
            (n_g * np.sum(all_wealth)) - (n_g + 1) / n_g)

    with col2:
        fig_wd = make_subplots(rows=1, cols=2,
                               subplot_titles=["Histogram Bogactwa (log scale)", "Krzywa Lorenza (Gini)"])
        fig_wd.add_trace(go.Histogram(x=all_wealth, nbinsx=80, name="Bogactwo",
                                       marker_color="#a855f7", opacity=0.7), row=1, col=1)
        # Lorenz curve
        cum_wealth = np.cumsum(all_wealth) / all_wealth.sum()
        cum_pop = np.linspace(0, 1, len(all_wealth))
        fig_wd.add_trace(go.Scatter(x=cum_pop, y=cum_wealth, mode="lines",
                                     name="Lorenz", line=dict(color="#00e676", width=2.5)), row=1, col=2)
        fig_wd.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="Doskonała Równość",
                                     line=dict(color="#aaa", dash="dash")), row=1, col=2)
        fig_wd.update_xaxes(type="log", row=1, col=1, gridcolor="#1c1c2e")
        fig_wd.update_yaxes(gridcolor="#1c1c2e")
        fig_wd.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=380,
            showlegend=False, margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_wd, use_container_width=True)

    top1_pct = all_wealth[int(0.99 * n_agents):].sum() / all_wealth.sum() * 100
    st.markdown(f"""<div style='{CARD}'>
    <b>Gini Coefficient:</b> {gini:.3f} ({'wysoka' if gini > 0.5 else 'umiarkowana'} nierówność)<br>
    <b>Top 1% bogactwa:</b> {top1_pct:.1f}% całości<br>
    <b>Prawo Pareto (80/20):</b> Top 20% posiada {'%.0f'%( all_wealth[int(0.8*n_agents):].sum()/all_wealth.sum()*100)}% bogactwa
    </div>""", unsafe_allow_html=True)


# ── TAB 4: PERCOLATION ─────────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 🕸️ Perkolacja — Propagacja Kryzysu w Sieci Finansowej")
    st.caption("Model perkolacji Broadbent & Hammersley (1957). Kryzys finansowy = perkolacja na sieci banków.")

    col1, col2 = st.columns([1, 2])
    with col1:
        perc_p = st.slider("p — prawdop. połączenia (interkonektywność)", 0.01, 1.0, 0.5, 0.01)
        perc_n = st.slider("N — liczba węzłów (banków)", 10, 100, 50)
        perc_seed = st.slider("Liczba bankrutów (seed failures)", 1, 10, 2)

    # Bond percolation on random graph
    np.random.seed(42)
    n_nodes = perc_n
    adj = (np.random.rand(n_nodes, n_nodes) < perc_p).astype(int)
    adj = np.triu(adj, 1) + np.triu(adj, 1).T

    # BFS from seed failures
    failed = set(range(perc_seed))
    frontier = list(failed)
    cascade = [len(failed)]

    for wave in range(20):
        new_failed = set()
        for node in frontier:
            neighbors = np.where(adj[node] == 1)[0]
            for nb in neighbors:
                if nb not in failed and np.random.rand() < perc_p:
                    new_failed.add(nb)
        if not new_failed:
            break
        failed.update(new_failed)
        frontier = list(new_failed)
        cascade.append(len(failed))

    failure_rate = len(failed) / n_nodes

    with col2:
        fig_perc = make_subplots(rows=1, cols=2,
                                  subplot_titles=["Kaskada Bankructw", "Sieć (węzły)"])
        fig_perc.add_trace(go.Scatter(x=list(range(len(cascade))), y=cascade, mode="lines+markers",
                                       name="Bankruci", line=dict(color="#ff1744", width=2.5),
                                       fill="tozeroy", fillcolor="rgba(255,23,68,0.1)"), row=1, col=1)
        # Node colors
        colors = ["#ff1744" if i in failed else "#00e676" for i in range(n_nodes)]
        x_pos = np.random.rand(n_nodes)
        y_pos = np.random.rand(n_nodes)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adj[i, j]:
                    fig_perc.add_trace(go.Scatter(
                        x=[x_pos[i], x_pos[j]], y=[y_pos[i], y_pos[j]],
                        mode="lines", line=dict(color="#2a2a3a", width=0.5),
                        showlegend=False), row=1, col=2)
        fig_perc.add_trace(go.Scatter(x=x_pos, y=y_pos, mode="markers",
                                       marker=dict(color=colors, size=8),
                                       showlegend=False), row=1, col=2)
        fig_perc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=380,
            margin=dict(l=40, r=20, t=60, b=40)
        )
        st.plotly_chart(fig_perc, use_container_width=True)

    # Critical threshold for Erdos-Renyi: p_c = 1/N
    p_c = 1.0 / n_nodes
    st.markdown(f"""<div style='{CARD}'>
    <b>Bankruci:</b> {len(failed)}/{n_nodes} ({failure_rate:.0%})<br>
    <b>Próg krytyczny (Erdős-Rényi) p_c:</b> {p_c:.3f} vs p={perc_p:.2f}
    {'→ ⚠️ POWYŻEJ p_c: systemic failure likely!' if perc_p > p_c else '→ ✅ PONIŻEJ p_c: kryzys ograniczony.'}
    </div>""", unsafe_allow_html=True)


# ── TAB 5: TEORIA ─────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown("### 📐 Ekonofizyka — Fundamenty")
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>⚛️ Analogie Fizyka → Ekonomia</div>
    <table style='width:100%;color:#ddd;font-size:13px'>
    <tr style='color:#00e676'><th>Fizyka</th><th>Ekonomia</th></tr>
    <tr><td>Spin (+1 / -1)</td><td>Trader (bull / bear)</td></tr>
    <tr><td>Temperatura T</td><td>Szum decyzji / Heterogeniczność</td></tr>
    <tr><td>Sprzężenie J</td><td>Naśladownictwo (herding)</td></tr>
    <tr><td>Magnetyzacja M</td><td>Sentyment rynku</td></tr>
    <tr><td>Przejście fazowe (T_c)</td><td>Crash / Bubble formation</td></tr>
    <tr><td>Boltzmann: P(E) ∝ e^{-E/kT}</td><td>Dystrybucja dochodów z pracy</td></tr>
    <tr><td>Perkolacja (p_c)</td><td>Próg systemic risk w sieci bankowej</td></tr>
    <tr><td>Power Law: P(x) ∝ x^{-α}</td><td>Ogon dystr. bogactwa, zwroty (Pareto)</td></tr>
    </table>
    </div>""", unsafe_allow_html=True)

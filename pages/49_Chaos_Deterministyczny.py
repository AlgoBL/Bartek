"""49_Chaos_Deterministyczny.py — Chaos Deterministyczny: Podwójne Wahadło & Decyzje Życiowe"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Chaos Deterministyczny",
    subtitle="Podwójne Wahadło · Wykładnik Lapunova · Efekt Motyla · Strategia Antychaotyczna",
    icon="🌀", badge="Chaos Theory"
), unsafe_allow_html=True)

# ─── STAŁE STYLOWANIA ────────────────────────────────────────────────────────
CARD = ("background:linear-gradient(135deg,#0f111a,#1a1c28);"
        "border:1px solid #2a2a3a;border-radius:14px;padding:18px 20px;margin-bottom:10px;"
        "font-family:'Inter',sans-serif")
CARD_GREEN = ("background:linear-gradient(135deg,#0f1a14,#1a281c);"
              "border:1px solid #00e67640;border-radius:14px;padding:18px 20px;margin-bottom:10px;"
              "font-family:'Inter',sans-serif")
CARD_RED = ("background:linear-gradient(135deg,#1a120f,#281c1a);"
            "border:1px solid #ff174440;border-radius:14px;padding:18px 20px;margin-bottom:10px;"
            "font-family:'Inter',sans-serif")
CARD_BLUE = ("background:linear-gradient(135deg,#0f131a,#1a1e28);"
             "border:1px solid #00ccff40;border-radius:14px;padding:18px 20px;margin-bottom:10px;"
             "font-family:'Inter',sans-serif")
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:8px"
H3_RED = "color:#ff1744;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:8px"
H3_BLUE = "color:#00ccff;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:8px"
NOTE = "color:#9da8b8;font-size:13px;line-height:1.7"
FORMULA = ("background:#0a0b10;border:1px solid #2a2a3a;border-radius:8px;"
           "padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:13px;"
           "color:#00ccff;margin:8px 0")


# ════════════════════════════════════════════════════════════════════════════
# SILNIK SYMULACJI — Podwójne Wahadło (RK4)
# ════════════════════════════════════════════════════════════════════════════
def double_pendulum_rhs(state, t, L1=1.0, L2=1.0, m1=1.0, m2=1.0, g=9.81):
    """Prawa strona równań ruchu podwójnego wahadła."""
    th1, w1, th2, w2 = state
    dth = th2 - th1
    denom1 = (m1 + m2) * L1 - m2 * L1 * np.cos(dth) ** 2
    denom2 = (L2 / L1) * denom1

    dw1 = (m2 * L1 * w1**2 * np.sin(dth) * np.cos(dth)
           + m2 * g * np.sin(th2) * np.cos(dth)
           + m2 * L2 * w2**2 * np.sin(dth)
           - (m1 + m2) * g * np.sin(th1)) / denom1

    dw2 = (-m2 * L2 * w2**2 * np.sin(dth) * np.cos(dth)
           + (m1 + m2) * g * np.sin(th1) * np.cos(dth)
           - (m1 + m2) * L1 * w1**2 * np.sin(dth)
           - (m1 + m2) * g * np.sin(th2)) / denom2

    return np.array([w1, dw1, w2, dw2])


@st.cache_data(show_spinner=False)
def simulate_double_pendulum(theta1_deg, delta_deg, T_sec, dt=0.02,
                              L1=1.0, L2=1.0, m1=1.0, m2=1.0):
    """Symuluje dwie trajektorie podwójnego wahadła — identyczne poza delta_deg."""
    th1 = np.radians(theta1_deg)
    th1b = th1 + np.radians(delta_deg)
    th2 = np.radians(theta1_deg * 0.8)

    state_a = np.array([th1, 0.0, th2, 0.0])
    state_b = np.array([th1b, 0.0, th2, 0.0])

    steps = int(T_sec / dt)
    times = np.linspace(0, T_sec, steps)
    traj_a = np.zeros((steps, 4))
    traj_b = np.zeros((steps, 4))
    traj_a[0] = state_a
    traj_b[0] = state_b

    def rk4_step(state, t, dt):
        k1 = double_pendulum_rhs(state, t, L1, L2, m1, m2)
        k2 = double_pendulum_rhs(state + 0.5*dt*k1, t + 0.5*dt, L1, L2, m1, m2)
        k3 = double_pendulum_rhs(state + 0.5*dt*k2, t + 0.5*dt, L1, L2, m1, m2)
        k4 = double_pendulum_rhs(state + dt*k3, t + dt, L1, L2, m1, m2)
        return state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    for i in range(1, steps):
        traj_a[i] = rk4_step(traj_a[i-1], times[i-1], dt)
        traj_b[i] = rk4_step(traj_b[i-1], times[i-1], dt)

    # Kartezjańskie współrzędne końca drugiego wahadła
    x2_a = L1*np.sin(traj_a[:,0]) + L2*np.sin(traj_a[:,2])
    y2_a = -L1*np.cos(traj_a[:,0]) - L2*np.cos(traj_a[:,2])
    x2_b = L1*np.sin(traj_b[:,0]) + L2*np.sin(traj_b[:,2])
    y2_b = -L1*np.cos(traj_b[:,0]) - L2*np.cos(traj_b[:,2])

    # Odległość między dwiema trajektoriami (kąt theta2)
    ang_diff = np.abs(np.degrees(traj_a[:,2] - traj_b[:,2]))

    return times, x2_a, y2_a, x2_b, y2_b, ang_diff, traj_a, traj_b


# ════════════════════════════════════════════════════════════════════════════
# ZAKŁADKI
# ════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🔁 Podwójne Wahadło",
    "📐 Matematyka Chaosu",
    "🧠 Chaos w Twoim Życiu",
    "🛡️ Strategia Antychaotyczna",
    "💊 Zastosowania Personalne"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PODWÓJNE WAHADŁO
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## 🔁 Podwójne Wahadło — Laboratorium Chaosu")
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>🔬 Dlaczego podwójne wahadło jest chaotyczne?</div>
    <p style='{NOTE}'>
    Podwójne wahadło działa według <b style='color:#00e676'>dokładnych równań fizyki Newtona</b> — 
    nie ma tu żadnej losowości. Ale jest <b style='color:#ff1744'>ekstremalnie wrażliwe na warunki 
    początkowe</b>. Dwie trajektorie różniące się o ułamek stopnia po kilku sekundach stają się 
    <b>zupełnie inne</b>. To właśnie jest <b style='color:#ffea00'>chaos deterministyczny</b>.
    </p>
    </div>""", unsafe_allow_html=True)

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        theta_start = st.slider("Kąt startowy θ₁ (°)", 10.0, 170.0, 120.0, 5.0, key="th1")
    with col_ctrl2:
        delta = st.slider("Różnica startowa Δθ (°)", 0.001, 5.0, 0.01, 0.001,
                          format="%.3f", key="delta_pend")
    with col_ctrl3:
        T_pend = st.slider("Czas symulacji (s)", 5.0, 60.0, 20.0, 1.0, key="T_pend")

    with st.spinner("Symulacja RK4 w toku..."):
        times, x2a, y2a, x2b, y2b, ang_diff, traj_a, traj_b = simulate_double_pendulum(
            theta_start, delta, T_pend
        )

    # Metryki
    threshold = 30.0
    diverge_idx = np.where(ang_diff > threshold)[0]
    t_predict = times[diverge_idx[0]] if len(diverge_idx) > 0 else T_pend
    lyap_est = np.log(ang_diff[-1] / max(delta, 0.001)) / T_pend if ang_diff[-1] > delta else 0.0

    m1c, m2c, m3c = st.columns(3)
    m1c.metric("Różnica startowa", f"{delta:.3f}°")
    m2c.metric("⏱️ Czas przewidywalności", f"{t_predict:.1f}s",
               help=f"Do kiedy różnica między trajektoriami < {threshold}°")
    m3c.metric("λ Lapunova (est.)", f"{lyap_est:.3f} /s",
               delta="Chaotyczny" if lyap_est > 0 else "Regularny",
               delta_color="inverse" if lyap_est > 0 else "normal")

    # Wykresy
    col_l, col_r = st.columns(2)

    with col_l:
        fig_traj = go.Figure()
        # Trajektoria A (zielona)
        fig_traj.add_trace(go.Scatter(
            x=x2a, y=y2a, mode="lines",
            name=f"Trajektoria A (θ={theta_start:.0f}°)",
            line=dict(color="#00e676", width=1.5),
            opacity=0.9
        ))
        # Trajektoria B (czerwona)
        fig_traj.add_trace(go.Scatter(
            x=x2b, y=y2b, mode="lines",
            name=f"Trajektoria B (θ={theta_start:.0f}+{delta:.3f}°)",
            line=dict(color="#ff1744", width=1.5),
            opacity=0.9
        ))
        # Punkt startowy
        fig_traj.add_trace(go.Scatter(
            x=[x2a[0]], y=[y2a[0]], mode="markers",
            marker=dict(color="#ffea00", size=10, symbol="star"),
            name="Start (prawie identyczny)"
        ))
        fig_traj.update_layout(
            title=f"Trajektorie końca wahadła — Δθ = {delta:.3f}°",
            xaxis=dict(title="x [m]", gridcolor="#1c1c2e", range=[-2.2, 2.2]),
            yaxis=dict(title="y [m]", gridcolor="#1c1c2e", range=[-2.2, 2.2]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=400,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
            margin=dict(l=50, r=20, t=60, b=50)
        )
        st.plotly_chart(fig_traj, use_container_width=True)

    with col_r:
        # Rozbieżność w czasie
        fig_div = go.Figure()
        fig_div.add_trace(go.Scatter(
            x=times, y=ang_diff, mode="lines",
            name="Różnica kątów |Δθ₂|",
            line=dict(color="#00ccff", width=2.5),
            fill="tozeroy", fillcolor="rgba(0,204,255,0.06)"
        ))
        fig_div.add_hline(y=threshold, line_dash="dash", line_color="#ff1744",
                          annotation_text=f"Próg {threshold}° (utrata przewidywalności)",
                          annotation_font_color="#ff1744")
        if len(diverge_idx) > 0:
            fig_div.add_vline(x=t_predict, line_dash="dot", line_color="#ffea00",
                              annotation_text=f"t* = {t_predict:.1f}s",
                              annotation_font_color="#ffea00")
        fig_div.update_layout(
            title="Rozbieżność trajektorii w czasie",
            xaxis=dict(title="Czas [s]", gridcolor="#1c1c2e"),
            yaxis=dict(title="|Δθ₂| [°]", gridcolor="#1c1c2e"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=400,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
            margin=dict(l=50, r=20, t=60, b=50)
        )
        st.plotly_chart(fig_div, use_container_width=True)

    # Log-skala rozbieżności
    st.markdown("### 📈 Wykładniczy Wzrost Błędu (skala logarytmiczna)")
    fig_log = go.Figure()
    safe_diff = np.maximum(ang_diff, 1e-6)
    fig_log.add_trace(go.Scatter(
        x=times, y=np.log10(safe_diff), mode="lines",
        name="log₁₀|Δθ₂|",
        line=dict(color="#a855f7", width=2.5),
    ))
    # Trend liniowy (= wykładniczy wzrost)
    valid = safe_diff > 0.01
    if valid.sum() > 10:
        t_valid = times[valid]
        y_valid = np.log10(safe_diff[valid])
        slope = np.polyfit(t_valid[:len(t_valid)//2], y_valid[:len(t_valid)//2], 1)
        y_fit = np.polyval(slope, times)
        fig_log.add_trace(go.Scatter(
            x=times, y=y_fit, mode="lines",
            name=f"Trend liniowy (λ≈{slope[0]:.3f}/s)",
            line=dict(color="#ffea00", width=1.5, dash="dash"),
        ))
    fig_log.update_layout(
        title="Wykładniczy wzrost różnicy — linia prosta = wzrost e^(λt)",
        xaxis=dict(title="Czas [s]", gridcolor="#1c1c2e"),
        yaxis=dict(title="log₁₀|Δθ₂|", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), height=280,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
        margin=dict(l=50, r=20, t=50, b=50)
    )
    st.plotly_chart(fig_log, use_container_width=True)

    st.markdown(f"""<div style='{CARD_GREEN}'>
    <div style='{H3}'>🔑 Kluczowy wniosek z symulacji</div>
    <p style='{NOTE}'>
    Przy różnicy startowej <b style='color:#ffea00'>{delta:.3f}°</b> trajektorie rozbiegają się 
    po <b style='color:#00e676'>~{t_predict:.1f}s</b>. Szacowany wykładnik Lapunova: 
    <b style='color:#00ccff'>λ ≈ {lyap_est:.3f}/s</b>.<br><br>
    To oznacza że <b>błąd podwaja się co ~{0.693/max(lyap_est,0.01):.1f}s</b>. 
    Superkomputer nie pomoże — nawet z 10× dokładniejszym pomiarem zyska tylko 
    <b style='color:#ff1744'>+{0.693/max(lyap_est,0.01):.1f}s</b> dodatkowej przewidywalności.
    </p>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — MATEMATYKA CHAOSU
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 📐 Matematyka Chaosu Deterministycznego")

    col_th, col_eq = st.columns([3, 2])

    with col_th:
        st.markdown(f"""<div style='{CARD_BLUE}'>
        <div style='{H3_BLUE}'>📐 Równania Ruchu Podwójnego Wahadła</div>
        <div style='{FORMULA}'>
        θ̈₁ = [m₂L₁ω₁²sin(Δ)cos(Δ) + m₂g·sin(θ₂)cos(Δ) + m₂L₂ω₂²sin(Δ) - (m₁+m₂)g·sin(θ₁)]<br>
        &nbsp;&nbsp;&nbsp;&nbsp;/ [(m₁+m₂)L₁ - m₂L₁cos²(Δ)]<br><br>
        gdzie Δ = θ₂ - θ₁
        </div>
        <p style='{NOTE}'>
        Równania są <b style='color:#ff1744'>nieliniowe</b> — zawierają produkty trygonometryczne 
        kątów i prędkości. Nie istnieje "ładne" analityczne rozwiązanie jak dla zwykłego wahadła 
        (sin θ ≈ θ). Jedynym wyjściem jest całkowanie numeryczne krok po kroku.<br><br>
        Właśnie ta <b style='color:#ffea00'>nieliniowość + sprzężenie</b> obu wahadł tworzy chaos.
        </p>
        </div>""", unsafe_allow_html=True)

    with col_eq:
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>⚡ Wykładnik Lapunova λ</div>
        <div style='{FORMULA}'>|δ(t)| ≈ |δ(0)| · e^(λt)</div>
        <p style='{NOTE}'>
        <b style='color:#00e676'>λ > 0:</b> układ chaotyczny — błąd rośnie wykładniczo<br>
        <b style='color:#3498db'>λ = 0:</b> granica stabilności<br>
        <b style='color:#aaa'>λ < 0:</b> układ stabilny, błędy zanikają<br><br>
        <b style='color:#ffea00'>Horyzont przewidywalności:</b><br>
        </p>
        <div style='{FORMULA}'>T* = (1/λ) · ln(ε_max / ε_0)</div>
        <p style='{NOTE}'>
        gdzie ε₀ = błąd pomiaru, ε_max = dopuszczalny błąd<br><br>
        Nawet 1000× lepszy pomiar → tylko +ln(1000)/λ ≈ <b style='color:#ff1744'>+7/λ</b> sekund extra!
        </p>
        </div>""", unsafe_allow_html=True)

    # Porównanie: zwykłe vs podwójne wahadło
    st.markdown("### 🔍 Zwykłe Wahadło vs Podwójne — Porównanie Zachowania")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        # Zwykłe wahadło — regularne
        t_sim = np.linspace(0, 20, 1000)
        theta0_single = np.radians(45)
        omega_single = np.sqrt(9.81 / 1.0)
        theta_single_a = theta0_single * np.cos(omega_single * t_sim)
        theta_single_b = (theta0_single + np.radians(1.0)) * np.cos(omega_single * t_sim)

        fig_single = go.Figure()
        fig_single.add_trace(go.Scatter(x=t_sim, y=np.degrees(theta_single_a),
                                         name="θ₀ = 45°", line=dict(color="#00e676", width=2)))
        fig_single.add_trace(go.Scatter(x=t_sim, y=np.degrees(theta_single_b),
                                         name="θ₀ = 46°", line=dict(color="#ff1744", width=2,
                                                                      dash="dash")))
        fig_single.update_layout(
            title="Zwykłe Wahadło — Regularny (różnica = const)",
            xaxis=dict(title="Czas [s]", gridcolor="#1c1c2e"),
            yaxis=dict(title="θ [°]", gridcolor="#1c1c2e"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=300,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
            margin=dict(l=50, r=20, t=60, b=50)
        )
        st.plotly_chart(fig_single, use_container_width=True)
        st.markdown(f"""<div style='{CARD_GREEN}'>
        <div style='{H3}'>✅ Zwykłe Wahadło</div>
        <p style='{NOTE}'>λ ≈ 0 (regularny). Różnica między trajektoriami pozostaje 
        <b style='color:#00e676'>stała</b> — proporcjonalna do różnicy startowej.<br>
        Pełne rozwiązanie analityczne: θ(t) = θ₀·cos(ωt)</p>
        </div>""", unsafe_allow_html=True)

    with col_s2:
        # Podwójne wahadło — chaotyczne (reużyj danych z tab1)
        fig_double = go.Figure()
        fig_double.add_trace(go.Scatter(
            x=times, y=np.degrees(traj_a[:, 2]),
            name=f"θ₂ traj.A", line=dict(color="#00e676", width=1.5)))
        fig_double.add_trace(go.Scatter(
            x=times, y=np.degrees(traj_b[:, 2]),
            name=f"θ₂ traj.B (+{delta:.3f}°)", line=dict(color="#ff1744", width=1.5)))
        fig_double.update_layout(
            title="Podwójne Wahadło — Chaotyczny (trajektorie rozbiegają się)",
            xaxis=dict(title="Czas [s]", gridcolor="#1c1c2e"),
            yaxis=dict(title="θ₂ [°]", gridcolor="#1c1c2e"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=300,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
            margin=dict(l=50, r=20, t=60, b=50)
        )
        st.plotly_chart(fig_double, use_container_width=True)
        st.markdown(f"""<div style='{CARD_RED}'>
        <div style='{H3_RED}'>🌀 Podwójne Wahadło</div>
        <p style='{NOTE}'>λ > 0 (chaotyczny). Różnica między trajektoriami 
        <b style='color:#ff1744'>rośnie wykładniczo</b>. Brak rozwiązania analitycznego.<br>
        Przewidywalność ograniczona przez precyzję pomiaru warunków początkowych.</p>
        </div>""", unsafe_allow_html=True)

    # Przestrzeń fazowa
    st.markdown("### 🌐 Przestrzeń Fazowa — Portret Fazowy Trajektorii")
    fig_phase = make_subplots(rows=1, cols=2,
                               subplot_titles=["Trajektoria A (zielona)", "Trajektoria B (czerwona)"])
    fig_phase.add_trace(go.Scatter(
        x=np.degrees(traj_a[:, 0]), y=np.degrees(traj_a[:, 1]),
        mode="lines", line=dict(color="#00e676", width=0.8),
        name="Traj.A: θ₁ vs ω₁", opacity=0.8
    ), row=1, col=1)
    fig_phase.add_trace(go.Scatter(
        x=np.degrees(traj_b[:, 0]), y=np.degrees(traj_b[:, 1]),
        mode="lines", line=dict(color="#ff1744", width=0.8),
        name="Traj.B: θ₁ vs ω₁", opacity=0.8
    ), row=1, col=2)
    fig_phase.update_layout(
        height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        margin=dict(l=50, r=20, t=60, b=50)
    )
    fig_phase.update_xaxes(title_text="θ₁ [°]", gridcolor="#1c1c2e")
    fig_phase.update_yaxes(title_text="ω₁ [°/s]", gridcolor="#1c1c2e")
    st.plotly_chart(fig_phase, use_container_width=True)
    st.caption("Atraktor chaotyczny — dwie prawie identyczne trajektorie zajmują zupełnie różne obszary przestrzeni fazowej.")

    # Horyzont przewidywalności
    st.markdown("### ⏱️ Kalkulator Horyzontu Przewidywalności")
    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        lambda_input = st.number_input("Wykładnik Lapunova λ (/s)", 0.01, 5.0, 0.5, 0.01)
    with col_h2:
        eps0 = st.number_input("Błąd pomiaru ε₀ (°)", 0.0001, 1.0, 0.01, 0.001, format="%.4f")
    with col_h3:
        eps_max = st.number_input("Dopuszczalny błąd ε_max (°)", 1.0, 180.0, 30.0, 1.0)

    T_star = (1 / lambda_input) * np.log(eps_max / eps0)
    T_star_10x = (1 / lambda_input) * np.log(eps_max / (eps0 / 10))

    c1, c2, c3 = st.columns(3)
    c1.metric("Horyzont T*", f"{T_star:.1f}s",
              help="Czas do przekroczenia progu błędu")
    c2.metric("T* przy 10× lepszym pomiarze", f"{T_star_10x:.1f}s",
              delta=f"+{T_star_10x - T_star:.1f}s")
    c3.metric("Zysk z 10× czulszego sensora", f"{T_star_10x - T_star:.1f}s",
              help="Logarytmiczny zwrot z precyzji pomiaru")

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>💡 Wniosek: logarytmiczny zwrot z precyzji</div>
    <div style='{FORMULA}'>T*(ε₀/10) - T*(ε₀) = ln(10)/λ ≈ {np.log(10)/lambda_input:.1f}s</div>
    <p style='{NOTE}'>
    Dziesięciokrotnie dokładniejszy pomiar to tylko <b style='color:#ffea00'>+{np.log(10)/lambda_input:.1f}s</b> 
    dodatkowej przewidywalności. Milionkrotnie dokładniejszy pomiar → +{np.log(1e6)/lambda_input:.1f}s.<br><br>
    To ograniczenie <b style='color:#ff1744'>fundamentalne</b>, nie technologiczne. 
    Dokładnie ten sam mechanizm ogranicza prognozę pogody (λ ≈ 0.05/dzień → T* ≈ 5-14 dni).
    </p>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHAOS W TWOIM ŻYCIU
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## 🧠 Chaos Deterministyczny w Twoim Życiu")

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>🔑 Kluczowe przetłumaczenie</div>
    <p style='{NOTE}'>
    Chaos deterministyczny w życiu <b>nie znaczy losowość</b>. Oznacza:<br>
    👉 <b style='color:#00e676'>Małe decyzje → nieliniowe, skumulowane skutki</b><br>
    👉 <b style='color:#ff1744'>Długoterminowe prognozy są fundamentalnie niemożliwe</b><br>
    👉 <b style='color:#ffea00'>Ale kierunek i odporność da się kształtować</b><br><br>
    Pytanie nie brzmi "co się stanie?" tylko "czy przetrwam gdy się pomylę?"
    </p>
    </div>""", unsafe_allow_html=True)

    # ── Mapa efektu motyla ────────────────────────────────────────────────
    st.markdown("### 🦋 Mapa Efektu Motyla — Eksponencjalny Efekt Małych Decyzji")

    col_mb1, col_mb2 = st.columns([2, 3])
    with col_mb1:
        domain = st.selectbox("Domena życiowa:", [
            "🏃 Zdrowie i Trening", "💰 Finanse i Oszczędności",
            "🧠 Wiedza i Umiejętności", "🤝 Relacje i Sieć"
        ], key="domain_sel")
        daily_action = st.slider("Codzienna akcja (skala 1-10)", 1, 10, 5, key="daily_action",
                                 help="1 = minimalna zmiana, 10 = maksymalna zmiana")
        years = st.slider("Horyzont czasowy (lata)", 1, 30, 10, key="years_mb")

    with col_mb2:
        t_years = np.linspace(0, years, 300)

        # Efekt liniowy
        linear_effect = daily_action * t_years

        # Efekt wykładniczy (chaos + kumulacja)
        # Małe codzienne zmiany kumulują się wykładniczo
        daily_rate = 1 + (daily_action - 5) * 0.005  # ±0.5% dziennie zmiana
        exp_effect = 100 * daily_rate**(t_years * 365) / 100

        # Efekt bez zmiany (baseline)
        baseline = np.ones_like(t_years) * 5 * years

        domain_labels = {
            "🏃 Zdrowie i Trening": ("Forma fizyczna (punkty)", "#00e676", "Ryzyko chorób (%)"),
            "💰 Finanse i Oszczędności": ("Kapitał (×)", "#ffea00", "Dług stresu (%)"),
            "🧠 Wiedza i Umiejętności": ("Kompetencja (punkty)", "#00ccff", "Stagnacja (%)"),
            "🤝 Relacje i Sieć": ("Wartość sieci (×)", "#a855f7", "Izolacja (%)")
        }
        label_y, color_pos, label_neg = domain_labels[domain]

        fig_butterfly = go.Figure()

        # Linia zerowej zmiany
        fig_butterfly.add_trace(go.Scatter(
            x=t_years, y=np.ones_like(t_years),
            mode="lines", name="Brak zmiany (baseline)",
            line=dict(color="#555", width=1.5, dash="dot")
        ))

        # Efekt liniowy (błędne myślenie)
        normalized_linear = 1 + linear_effect / (5 * years)
        fig_butterfly.add_trace(go.Scatter(
            x=t_years, y=normalized_linear,
            mode="lines", name="Myślenie liniowe (błąd!)",
            line=dict(color="#888", width=2, dash="dash")
        ))

        # Efekt rzeczywisty (wykładniczy)
        fig_butterfly.add_trace(go.Scatter(
            x=t_years, y=exp_effect,
            mode="lines", name=f"Efekt rzeczywisty (skumulowany)",
            line=dict(color=color_pos, width=3),
            fill="tonexty", fillcolor=f"rgba(0,230,118,0.05)"
        ))

        # Adnotacja końcowa
        final_val = exp_effect[-1]
        fig_butterfly.add_annotation(
            x=years, y=final_val,
            text=f"{final_val:.1f}× po {years} latach",
            showarrow=True, arrowcolor=color_pos,
            font=dict(color=color_pos, size=12, family="Inter"),
            bgcolor="#0f111a", bordercolor=color_pos, borderwidth=1
        )

        fig_butterfly.update_layout(
            title=f"Efekt motyla w: {domain}",
            xaxis=dict(title="Lata", gridcolor="#1c1c2e"),
            yaxis=dict(title="Efekt mnożnikowy (1.0 = baseline)", gridcolor="#1c1c2e"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=350,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
            margin=dict(l=50, r=20, t=60, b=50)
        )
        st.plotly_chart(fig_butterfly, use_container_width=True)

    # Kalkulator efektu kumulatywnego
    final_x = exp_effect[-1]
    direction = "wzrost" if daily_action > 5 else "spadek"
    color_dir = "#00e676" if daily_action > 5 else "#ff1744"
    st.markdown(f"""<div style='{CARD_GREEN if daily_action > 5 else CARD_RED}'>
    <div style='{H3 if daily_action > 5 else H3_RED}'>
    📊 Twój Efekt Motyla: akcja {daily_action}/10 przez {years} lat
    </div>
    <p style='{NOTE}'>
    Dzienny poziom działania <b style='color:{color_dir}'>{daily_action}/10</b> przez 
    <b>{years} lat</b> = efekt mnożnikowy <b style='color:{color_dir}'>{final_x:.1f}×</b> 
    względem braku działania.<br><br>
    {'🟢 Małe pozytywne działania <b>kumulują się wykładniczo</b> — nie liniowo. ' +
     'Różnica 1 punktu dziennie to gigantyczna różnica po dekadzie.' if daily_action > 5 else
     '🔴 Małe zaniedbania <b>kumulują się wykładniczo</b> — w dół. ' +
     'To ten sam mechanizm co wahadło — tylko z negatywnym wykładnikiem.'}
    </p>
    </div>""", unsafe_allow_html=True)

    # ── Kalkulator marginesu błędu ────────────────────────────────────────
    st.markdown("### 📏 Kalkulator Marginesu Błędu — Jak Rośnie Niepewność Życiowa")

    col_me1, col_me2 = st.columns(2)

    with col_me1:
        init_uncertainty = st.slider("Twoja obecna niepewność (%)", 5, 50, 15, key="life_uncert")
        life_lambda = st.slider("Złożoność sytuacji (jak chaotyczne?)", 0.01, 0.5, 0.1, 0.01,
                                key="life_lambda",
                                help="0.01 = stabilna sytuacja, 0.5 = bardzo zmienna")
        horizon_months = st.slider("Horyzont prognozy (miesiące)", 1, 60, 24, key="life_horizon")

    with col_me2:
        t_months = np.linspace(0, horizon_months, 300)
        # Wzrost niepewności wykładniczy
        uncertainty_growth = init_uncertainty * np.exp(life_lambda * t_months / 12)
        uncertainty_growth = np.minimum(uncertainty_growth, 100)

        # Obszar "przewidywalny" i "chaotyczny"
        chaos_threshold = 50  # 50% niepewność = chaos
        chaos_start_idx = np.where(uncertainty_growth >= chaos_threshold)[0]
        t_chaos = t_months[chaos_start_idx[0]] if len(chaos_start_idx) > 0 else horizon_months

        fig_uncert = go.Figure()
        fig_uncert.add_trace(go.Scatter(
            x=t_months, y=uncertainty_growth,
            mode="lines", name="Niepewność (%)",
            line=dict(color="#ff1744", width=2.5),
            fill="tozeroy", fillcolor="rgba(255,23,68,0.06)"
        ))
        fig_uncert.add_hline(y=chaos_threshold, line_dash="dash", line_color="#ffea00",
                              annotation_text=f"Próg chaosu ({chaos_threshold}%)",
                              annotation_font_color="#ffea00")
        if t_chaos < horizon_months:
            fig_uncert.add_vline(x=t_chaos, line_dash="dot", line_color="#ff1744",
                                  annotation_text=f"Chaos od {t_chaos:.0f}m",
                                  annotation_font_color="#ff1744")
        fig_uncert.update_layout(
            title="Wzrost Niepewności w Czasie",
            xaxis=dict(title="Miesiące", gridcolor="#1c1c2e"),
            yaxis=dict(title="Niepewność (%)", gridcolor="#1c1c2e", range=[0, 105]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=300,
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_uncert, use_container_width=True)

    # 5 zasad praktycznych
    st.markdown("### 🗝️ 5 Zasad Życia w Chaotycznym Środowisku")
    principles = [
        ("🎯", "Skup się na procesie, nie wyniku",
         "Wynik jest chaotyczny — zależy od setek zmiennych których nie kontrolujesz. "
         "Proces (codzienne działania) jest kontrolowalny i kumulatywny."),
        ("🛡️", "Buduj odporność, nie optymalizację",
         "W chaotycznych układach 'optymalne' rozwiązanie jest kruche. "
         "Odporny system działa w wielu scenariuszach, nie perfekcyjnie w jednym."),
        ("🎲", "Małe, odwracalne decyzje",
         "Im mniejsza i bardziej odwracalna decyzja, tym mniejszy koszt błędu. "
         "Chaos nagradza eksperymentowanie, karze duże nieodwracalne zakłady."),
        ("📊", "Myśl w prawdopodobieństwach, nie pewnikach",
         "Nie pytaj 'co się stanie?' — pytaj 'co jest bardziej prawdopodobne?'. "
         "Buduj plan B zanim go potrzebujesz."),
        ("⏳", "Krótkoterminowo: reaguj. Długoterminowo: rezygnuj z prognozy",
         "Prognoza na 3 miesiące: możliwa. Na 3 lata: niemożliwa. "
         "Planuj strukturę i zasady, nie konkretne wyniki."),
    ]

    for icon, title, desc in principles:
        st.markdown(f"""<div style='{CARD}'>
        <div style='display:flex;gap:12px;align-items:flex-start'>
        <span style='font-size:24px;flex-shrink:0'>{icon}</span>
        <div>
        <div style='color:#00e676;font-weight:700;font-size:14px;margin-bottom:4px'>{title}</div>
        <p style='{NOTE}'>{desc}</p>
        </div>
        </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — STRATEGIA ANTYCHAOTYCZNA
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🛡️ Strategia Antychaotyczna — Narzędzia Decyzyjne")

    # ── Score Antykruchości ───────────────────────────────────────────────
    st.markdown("### ⚖️ Kalkulator Antykruchości Decyzji (Taleb)")
    st.caption("Dla każdej rozważanej decyzji — oceń 3 wymiary na skali 1-10")

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        loss_risk = st.slider("🔴 Ryzyko straty (1=minimalne, 10=katastrofalne)", 1, 10, 4, key="loss_r")
    with col_d2:
        gain_potential = st.slider("🟢 Potencjał zysku (1=minimalny, 10=ogromny)", 1, 10, 7, key="gain_p")
    with col_d3:
        reversibility = st.slider("🔵 Odwracalność (1=nieodwracalne, 10=łatwe do cofnięcia)", 1, 10, 6, key="revers")

    # Score antykruchości
    antifragile_score = (gain_potential * reversibility) / (loss_risk + 0.1) * 10
    asymmetry = gain_potential / loss_risk
    survivability = reversibility / loss_risk

    # Normalizacja do 0-100
    score_norm = min(100, antifragile_score * 2)

    color_score = "#00e676" if score_norm >= 60 else "#ffea00" if score_norm >= 35 else "#ff1744"
    verdict_score = ("✅ ANTYKRUCHY — działaj" if score_norm >= 60 else
                     "⚠️ MIESZANY — ogranicz ekspozycję" if score_norm >= 35 else
                     "🛑 KRUCHY — unikaj lub zabezpiecz się")

    fig_anti = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_norm,
        number={"font": {"size": 52, "color": color_score}, "suffix": "/100"},
        title={"text": "Score Antykruchości", "font": {"color": "white", "size": 14}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color_score, "thickness": 0.15},
            "steps": [{"range": [0, 35], "color": "rgba(255,23,68,0.15)"},
                      {"range": [35, 60], "color": "rgba(255,234,0,0.15)"},
                      {"range": [60, 100], "color": "rgba(0,230,118,0.15)"}],
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
        }
    ))
    fig_anti.update_layout(height=220, margin=dict(l=30, r=30, t=20, b=20),
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="white", family="Inter"))

    col_gauge, col_metrics = st.columns([1, 1])
    with col_gauge:
        st.plotly_chart(fig_anti, use_container_width=True)
    with col_metrics:
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>📊 Dekompozycja Score</div>
        <p style='{NOTE}'>
        <b style='color:#00e676'>Asymetria zysk/strata:</b> {asymmetry:.2f}× 
        {"✅ Dobra asymetria" if asymmetry > 1 else "❌ Zła asymetria"}<br>
        <b style='color:#00ccff'>Przeżywalność (odwrac./ryzyko):</b> {survivability:.2f}
        {"✅ Odwracalne" if survivability > 1 else "⚠️ Trudne do cofnięcia"}<br><br>
        <b style='font-size:16px;color:{color_score}'>{verdict_score}</b>
        </p>
        </div>""", unsafe_allow_html=True)

    # ── Macierz Scenariuszy ───────────────────────────────────────────────
    st.markdown("### 🗺️ Macierz Scenariuszy — 'Co Jeśli Się Mylę?'")
    st.caption("Zdefiniuj decyzję i oceń ją dla 4 scenariuszy")

    decision_text = st.text_input("Opisz rozważaną decyzję:",
                                   placeholder="np. 'Inwestuję 30% oszczędności w VWCE'",
                                   key="decision_text")

    scenarios = ["🌟 Najlepszy", "📊 Bazowy", "⛈️ Najgorszy", "🦢 Czarny Łabędź"]
    colors_sc = ["#00e676", "#3498db", "#f39c12", "#ff1744"]

    sc_cols = st.columns(4)
    scenario_outcomes = []
    scenario_survives = []

    for i, (sc, col_sc) in enumerate(zip(scenarios, sc_cols)):
        with col_sc:
            st.markdown(f"**{sc}**")
            outcome = st.slider(f"Wynik ({sc.split()[1]})", -100, 200, [30, 10, -20, -50][i],
                                key=f"sc_out_{i}", help="% zmiana wartości/wyniku")
            survives = st.checkbox("Przeżyję to?", value=(i < 3), key=f"sc_surv_{i}")
            scenario_outcomes.append(outcome)
            scenario_survives.append(survives)

    fig_scenario = go.Figure()
    fig_scenario.add_trace(go.Bar(
        x=[s.split()[1] for s in scenarios],
        y=scenario_outcomes,
        marker_color=[colors_sc[i] if scenario_survives[i] else "#666" for i in range(4)],
        text=[f"{v:+d}%\n{'✅' if scenario_survives[i] else '☠️'}"
              for i, v in enumerate(scenario_outcomes)],
        textposition="outside",
        textfont=dict(color="white", size=13),
        width=0.6
    ))
    fig_scenario.add_hline(y=0, line_color="#555", line_width=1)
    fig_scenario.update_layout(
        title=f"Macierz Scenariuszy{': ' + decision_text[:40] if decision_text else ''}",
        xaxis=dict(title="Scenariusz", gridcolor="#1c1c2e"),
        yaxis=dict(title="Wynik (%)", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), height=320,
        margin=dict(l=50, r=20, t=60, b=80), showlegend=False
    )
    st.plotly_chart(fig_scenario, use_container_width=True)

    survive_count = sum(scenario_survives)
    color_surv = "#00e676" if survive_count == 4 else "#ffea00" if survive_count >= 3 else "#ff1744"
    st.markdown(f"""<div style='background:rgba(0,0,0,0.3);border:2px solid {color_surv};
    border-radius:12px;padding:16px;text-align:center'>
    <span style='font-size:18px;font-weight:700;color:{color_surv}'>
    Przeżyjesz {survive_count}/4 scenariuszy</span><br>
    <span style='color:#6b7280;font-size:12px'>
    {"✅ Dobra odporność — działaj" if survive_count >= 3 else 
     "⚠️ Słaba odporność — zmień parametry lub zaakceptuj ryzyko"}</span>
    </div>""", unsafe_allow_html=True)

    # ── Barbell Życiowy ───────────────────────────────────────────────────
    st.markdown("### 🏋️ Barbell Życiowy — Zasada Sztangi Taleba")
    col_bb1, col_bb2 = st.columns(2)

    with col_bb1:
        safe_life = st.slider("% zasobów w 'bezpiecznym' (stabilność)", 50, 95, 80, key="safe_life")
        risky_life = 100 - safe_life

        domains_bb = ["Finanse", "Czas", "Energia", "Relacje"]
        safe_vals = [safe_life] * 4
        risky_vals = [risky_life] * 4

        fig_bb_life = go.Figure()
        fig_bb_life.add_trace(go.Bar(
            name="🛡️ Bezpieczne", x=domains_bb, y=safe_vals,
            marker_color="#3498db", text=[f"{v}%" for v in safe_vals],
            textposition="inside", textfont=dict(color="white")
        ))
        fig_bb_life.add_trace(go.Bar(
            name="🚀 Asymetryczne", x=domains_bb, y=risky_vals,
            marker_color="#ff1744", text=[f"{v}%" for v in risky_vals],
            textposition="inside", textfont=dict(color="white")
        ))
        fig_bb_life.update_layout(
            barmode="stack",
            title=f"Barbell {safe_life}/{risky_life} w każdej domenie",
            xaxis=dict(gridcolor="#1c1c2e"),
            yaxis=dict(title="%", gridcolor="#1c1c2e", range=[0, 110]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=320,
            legend=dict(x=0.01, y=1.15, orientation="h", bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=40, r=20, t=70, b=50)
        )
        st.plotly_chart(fig_bb_life, use_container_width=True)

    with col_bb2:
        st.markdown(f"""<div style='{CARD_BLUE}'>
        <div style='{H3_BLUE}'>🏋️ Zasada Sztangi — Jak Działa</div>
        <p style='{NOTE}'>
        <b style='color:#3498db'>{safe_life}% Bezpieczne:</b><br>
        Stabilna podstawa — gotówka, stały dochód, sprawdzone rutyny, core relacje.
        Chroni przed ruiną. Daje pozycję i czas.<br><br>
        <b style='color:#ff1744'>{risky_life}% Asymetryczne:</b><br>
        Małe zakłady z asymetrycznym potencjałem — projekty, inwestycje spekulatywne, 
        nowe kontakty, eksperymenty. Ograniczone ryzyko straty, nieograniczony potencjał.<br><br>
        <b style='color:#ff1744'>Unikaj środka</b> — "umiarkowane ryzyko" (np. 50/50) to 
        najgorsza strategia w chaotycznym środowisku. Bierzesz ryzyko bez asymetrii.
        </p>
        </div>""", unsafe_allow_html=True)

        # Filtr decyzyjny chaosu
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>🎯 Chaos Filter — 3 Pytania Przed Każdym Ruchem</div>
        <p style='{NOTE}'>
        <b style='color:#ff1744'>1. Przetrwanie:</b> Czy przeżyję jeśli się mylę?<br>
        <b style='color:#ffea00'>2. Asymetria:</b> Czy zysk potencjalny > strata potencjalna?<br>
        <b style='color:#00e676'>3. Odwracalność:</b> Czy mogę cofnąć lub zmienić kurs?<br><br>
        Trzy ✅ → działaj. Dwa ✅ → działaj ostrożnie. Jedno ✅ → poczekaj lub zmień warunki.
        </p>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ZASTOSOWANIA PERSONALNE
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 💊 Zastosowania Personalne — Chaos w Twoim Kontekście")

    sub_tabs = st.tabs(["🩺 Zdrowie Sercowe", "📈 Finanse (Portfel)", "🧘 Decyzje Codzienne"])

    # ─── PODZAKŁADKA 1: ZDROWIE ──────────────────────────────────────────
    with sub_tabs[0]:
        st.markdown("### 🩺 Chaos w Układzie Sercowo-Naczyniowym")
        st.markdown(f"""<div style='{CARD_BLUE}'>
        <div style='{H3_BLUE}'>💓 Serce jako Układ Chaotyczny</div>
        <p style='{NOTE}'>
        Rytm serca jest <b>chaotyczny w zdrowym znaczeniu</b> — zmienność rytmu (HRV) to oznaka zdrowia.
        Zbyt regularny rytm (jak metronom) to sygnał choroby.<br><br>
        Migotanie przedsionków to z kolei <b style='color:#ff1744'>chaotyczny rozpad synchronizacji</b> — 
        układ traci swój "atraktor" i wpada w nieregularny, niekontrolowany chaos.<br><br>
        <b style='color:#00e676'>Kluczowa analogia z wahadłem:</b> Codzienne małe działania 
        (ruch, sen, stres) to warunki początkowe. W nieliniowym układzie sercowym → 
        małe zmiany kumulują się do dużych efektów zdrowotnych.
        </p>
        </div>""", unsafe_allow_html=True)

        col_h1, col_h2 = st.columns(2)

        with col_h1:
            st.markdown("#### 🏃 Symulator: Codzienny Ruch → Ryzyko Sercowe")
            daily_minutes = st.slider("Codzienny ruch (minuty)", 0, 120, 30, 5, key="heart_min")
            sleep_quality = st.slider("Jakość snu (1-10)", 1, 10, 7, key="sleep_q")
            stress_level = st.slider("Poziom stresu przewlekłego (1-10)", 1, 10, 5, key="stress_l")
            years_h = st.slider("Horyzont (lata)", 5, 30, 15, key="years_heart")

        with col_h2:
            t_h = np.linspace(0, years_h, 300)

            # Baseline ryzyko migotania (uproszczony model)
            base_risk = 5.0  # 5% ryzyko przy 0 ruchu

            # Nieliniowy efekt ruchu (odwrócony eksponencjalny)
            exercise_factor = np.exp(-daily_minutes * 0.015)  # im więcej ruchu, tym mniej ryzyka
            sleep_factor = 1 + (5 - sleep_quality) * 0.08     # słaby sen → wyższe ryzyko
            stress_factor = 1 + (stress_level - 5) * 0.06      # wysoki stres → wyższe ryzyko

            # Skumulowane ryzyko w czasie
            annual_risk = base_risk * exercise_factor * sleep_factor * stress_factor
            cumulative_risk = 100 * (1 - (1 - annual_risk/100) ** t_h)

            # Porównanie: bez zmiany nawyków
            annual_risk_no_change = base_risk * 1.2
            cumulative_no_change = 100 * (1 - (1 - annual_risk_no_change/100) ** t_h)

            fig_heart = go.Figure()
            fig_heart.add_trace(go.Scatter(
                x=t_h, y=cumulative_no_change,
                mode="lines", name="Bez zmiany nawyków",
                line=dict(color="#ff1744", width=2.5, dash="dash")
            ))
            fig_heart.add_trace(go.Scatter(
                x=t_h, y=cumulative_risk,
                mode="lines", name=f"Twój plan ({daily_minutes}min/dzień)",
                line=dict(color="#00e676", width=2.5),
                fill="tonexty", fillcolor="rgba(0,230,118,0.05)"
            ))
            risk_saved = cumulative_no_change[-1] - cumulative_risk[-1]
            fig_heart.add_annotation(
                x=years_h * 0.7, y=(cumulative_no_change[-1] + cumulative_risk[-1]) / 2,
                text=f"Redukcja ryzyka:<br><b>{risk_saved:.1f}pp</b>",
                showarrow=False,
                font=dict(color="#ffea00", size=12), bgcolor="#0f111a",
                bordercolor="#ffea00", borderwidth=1
            )
            fig_heart.update_layout(
                title="Skumulowane ryzyko incydentu sercowego (%)",
                xaxis=dict(title="Lata", gridcolor="#1c1c2e"),
                yaxis=dict(title="Ryzyko skumulowane (%)", gridcolor="#1c1c2e", range=[0, 50]),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Inter"), height=320,
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
                margin=dict(l=50, r=20, t=60, b=50)
            )
            st.plotly_chart(fig_heart, use_container_width=True)

        # Rekomendacje zdrowotne
        rec_color = "#00e676" if daily_minutes >= 30 else "#ffea00" if daily_minutes >= 15 else "#ff1744"
        rec_text = ("✅ Doskonały poziom — nieliniowy efekt ochronny w pełni aktywny" if daily_minutes >= 45 else
                    "✅ Dobry poziom — utrzymaj i stopniowo zwiększaj" if daily_minutes >= 30 else
                    "⚠️ Poniżej progu — minimum 30 min/dzień dla efektu kardioprotekcji" if daily_minutes >= 15 else
                    "🔴 Krytyczny niedobór — każde 10 min więcej = istotna redukcja ryzyka")

        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>🎯 Twój Plan Redukcji Chaosu Sercowego</div>
        <p style='{NOTE}'>
        <b style='color:{rec_color}'>{rec_text}</b><br><br>
        Przy {daily_minutes}min/dzień przez {years_h} lat:<br>
        Roczne ryzyko bazowe: <b>{annual_risk:.2f}%</b> vs 
        bez zmiany: <b>{annual_risk_no_change:.2f}%</b><br>
        Skumulowana redukcja ryzyka: <b style='color:#00e676'>{risk_saved:.1f}pp po {years_h} latach</b><br><br>
        <b style='color:#ffea00'>Pamiętaj:</b> To jest układ chaotyczny — efekt nie jest liniowy.
        Każda dodatkowa minuta ruchu ma malejący marginalny zysk, ale stały efekt anty-chaotyczny 
        na rytm serca (HRV, autonomia układu nerwowego).
        </p>
        </div>""", unsafe_allow_html=True)

    # ─── PODZAKŁADKA 2: FINANSE ──────────────────────────────────────────
    with sub_tabs[1]:
        st.markdown("### 📈 Chaos Rynkowy — Portfel Odporny")
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>📊 Rynek jako Układ Chaotyczny</div>
        <p style='{NOTE}'>
        Rynki finansowe mają dodatni wykładnik Lapunova — są chaotyczne. 
        Różnica: rynek jest <b style='color:#ffea00'>chaotyczny stochastycznie</b> 
        (chaos + losowość), nie deterministycznie jak wahadło.<br><br>
        Konsekwencja: <b style='color:#ff1744'>precyzyjna prognoza cen jest niemożliwa</b>. 
        Ale <b style='color:#00e676'>budowanie odporności portfela — jest możliwe</b>.
        </p>
        </div>""", unsafe_allow_html=True)

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown("#### ⚙️ Parametry Portfela")
            capital = st.number_input("Kapitał (PLN)", 10000, 10000000, 100000, 10000,
                                      key="capital_chaos")
            vwce_pct = st.slider("% VWCE (globalny ETF)", 0, 100, 60, 5, key="vwce_pct_c")
            bond_pct = st.slider("% Obligacje/Gotówka", 0, 100 - vwce_pct, 40, 5, key="bond_pct_c")
            years_f = st.slider("Horyzont (lata)", 5, 40, 20, key="years_finance_c")
            n_sim = 200

        with col_f2:
            np.random.seed(42)

            # Parametry zwrotów
            vwce_mu, vwce_sigma = 0.09, 0.18
            bond_mu, bond_sigma = 0.04, 0.04

            # Monte Carlo — symulacja chaotycznych ścieżek portfela
            all_paths = []
            for _ in range(n_sim):
                val = capital
                path = [val]
                for y in range(years_f):
                    r_vwce = np.random.normal(vwce_mu, vwce_sigma)
                    r_bond = np.random.normal(bond_mu, bond_sigma)
                    r_total = (vwce_pct / 100) * r_vwce + (bond_pct / 100) * r_bond
                    val = val * (1 + r_total)
                    path.append(val)
                all_paths.append(path)

            all_paths = np.array(all_paths)
            t_years_f = np.arange(years_f + 1)

            p10 = np.percentile(all_paths, 10, axis=0)
            p25 = np.percentile(all_paths, 25, axis=0)
            p50 = np.percentile(all_paths, 50, axis=0)
            p75 = np.percentile(all_paths, 75, axis=0)
            p90 = np.percentile(all_paths, 90, axis=0)

            fig_mc = go.Figure()

            # Fan chart
            fig_mc.add_trace(go.Scatter(
                x=np.concatenate([t_years_f, t_years_f[::-1]]),
                y=np.concatenate([p90, p10[::-1]]) / 1e6,
                fill="toself", fillcolor="rgba(0,230,118,0.05)",
                line=dict(width=0), name="p10-p90", showlegend=False
            ))
            fig_mc.add_trace(go.Scatter(
                x=np.concatenate([t_years_f, t_years_f[::-1]]),
                y=np.concatenate([p75, p25[::-1]]) / 1e6,
                fill="toself", fillcolor="rgba(0,230,118,0.12)",
                line=dict(width=0), name="p25-p75", showlegend=False
            ))

            # Przykładowe ścieżki
            for i in range(min(20, n_sim)):
                color = "#00e676" if all_paths[i, -1] > p50[-1] else "rgba(100,100,120,0.3)"
                fig_mc.add_trace(go.Scatter(
                    x=t_years_f, y=all_paths[i] / 1e6,
                    mode="lines", line=dict(color=color, width=0.5),
                    opacity=0.3, showlegend=False
                ))

            # Mediany i percentyle
            for vals, color, name in [(p10, "#ff1744", "p10"), (p50, "#00e676", "Mediana (p50)"),
                                       (p90, "#00ccff", "p90")]:
                fig_mc.add_trace(go.Scatter(
                    x=t_years_f, y=vals / 1e6,
                    mode="lines", name=name,
                    line=dict(color=color, width=2)
                ))

            fig_mc.update_layout(
                title=f"Monte Carlo: {vwce_pct}% VWCE + {bond_pct}% Obligacje ({n_sim} symulacji)",
                xaxis=dict(title="Lata", gridcolor="#1c1c2e"),
                yaxis=dict(title="Wartość Portfela (mln PLN)", gridcolor="#1c1c2e"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Inter"), height=360,
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.4)"),
                margin=dict(l=50, r=20, t=60, b=50)
            )
            st.plotly_chart(fig_mc, use_container_width=True)

        # Metryki portfela
        ruin_threshold = capital * 0.5
        ruin_count = sum(1 for p in all_paths if p[-1] < ruin_threshold)
        ruin_pct = ruin_count / n_sim * 100

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Mediana po {:.0f}l".format(years_f), f"{p50[-1]/1e6:.2f}M PLN",
                   f"{(p50[-1]/capital - 1)*100:.0f}% zwrot")
        mc2.metric("Pesymistyczny (p10)", f"{p10[-1]/1e6:.2f}M PLN")
        mc3.metric("Optymistyczny (p90)", f"{p90[-1]/1e6:.2f}M PLN")
        mc4.metric("Ryzyko ruiny (<50% start)", f"{ruin_pct:.0f}%",
                   delta="OK" if ruin_pct < 5 else "Za wysokie!",
                   delta_color="normal" if ruin_pct < 5 else "inverse")

        st.markdown(f"""<div style='{CARD_GREEN}'>
        <div style='{H3}'>💡 Wniosek Antychaotyczny dla Portfela</div>
        <p style='{NOTE}'>
        Nie optymalizuj pod "najlepszy scenariusz" — optymalizuj pod "przeżycie w najgorszym".<br><br>
        Portfel {vwce_pct}% VWCE + {bond_pct}% Obligacje:<br>
        • <b style='color:#00e676'>Mediana po {years_f} latach: {p50[-1]/1e6:.2f}M PLN</b><br>
        • <b style='color:#ff1744'>Ryzyko utraty >50% kapitału: {ruin_pct:.0f}%</b><br>
        • <b style='color:#ffea00'>Rozrzut (p10-p90): {p10[-1]/1e6:.2f}M – {p90[-1]/1e6:.2f}M PLN</b><br><br>
        Szeroki rozrzut to właśnie chaos rynkowy w akcji. 
        Twoja odpowiedź: <b>diversyfikacja + horyzont czasowy</b>, 
        nie precyzyjna prognoza (niemożliwa przy λ > 0).
        </p>
        </div>""", unsafe_allow_html=True)

    # ─── PODZAKŁADKA 3: DECYZJE CODZIENNE ───────────────────────────────
    with sub_tabs[2]:
        st.markdown("### 🧘 Chaos Filter — Algorytm Decyzji Codziennych")

        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>🔬 Fundamentalna Zmiana Perspektywy</div>
        <p style='{NOTE}'>
        W chaotycznym środowisku mózg naturalnie chce <b style='color:#ff1744'>przewidywać 
        i optymalizować</b>. To instynkt, ale w nieliniowych układach — błąd.<br><br>
        Właściwe pytanie: <b style='color:#00e676'>Nie "co się stanie?" → "co zrobię gdy nie wiem co się stanie?"</b>
        </p>
        </div>""", unsafe_allow_html=True)

        action_cd = st.text_input("Opisz decyzję/działanie do oceny:",
                                   placeholder="np. 'Wysłać email z propozycją do X' lub 'Kupić kurs online'",
                                   key="chaos_filter_action")

        if action_cd:
            st.markdown("#### Oceń przez pryzmat chaosu:")

            q1_ok = st.radio("1. 🛡️ Przetrwanie: Czy to działanie naraża mnie na nieodwracalną stratę?",
                              ["❌ TAK — naraża → STOP", "✅ NIE — bezpieczne → kontynuuj"],
                              key="cf_q1", horizontal=True)

            q2_ok = st.radio("2. 📈 Asymetria: Czy potencjalny zysk jest > od potencjalnej straty?",
                              ["❌ NIE — liniowe/symetryczne → minimalizuj", "✅ TAK — asymetryczne → inwestuj w proces"],
                              key="cf_q2", horizontal=True)

            q3_ok = st.radio("3. 🔄 Antykruchość: Czy to działanie buduje moją odporność na przyszłe szoki?",
                              ["❌ NIE — tylko reagowanie → ostrożnie", "✅ TAK — buduje zdolność adaptacji → działaj"],
                              key="cf_q3", horizontal=True)

            q4_ok = st.radio("4. ⏰ Czas: Czy robię to z cierpliwości czy desperacji?",
                              ["❌ DESPERACJA — poczekaj, zwiększ zasoby/BATNA", "✅ CIERPLIWOŚĆ — dobry sygnał"],
                              key="cf_q4", horizontal=True)

            passed = sum([
                "NIE" in q1_ok,
                "TAK" in q2_ok,
                "TAK" in q3_ok,
                "CIERPLIWOŚĆ" in q4_ok
            ])

            if passed == 4:
                verdict_cd = "✅ ZIELONE ŚWIATŁO — działaj z dyscypliną procesu"
                col_cd = "#00e676"
            elif passed >= 3:
                verdict_cd = "🟡 POMARAŃCZOWE — działaj ostrożnie, adresuj słabe punkty"
                col_cd = "#ffea00"
            else:
                verdict_cd = "🔴 CZERWONE — nie działaj teraz. Zmień warunki."
                col_cd = "#ff1744"

            st.markdown(f"""<div style='background:rgba(0,0,0,0.4);border:2px solid {col_cd};
            border-radius:14px;padding:20px;text-align:center;margin-top:16px'>
            <div style='font-size:22px;font-weight:700;color:{col_cd};margin-bottom:8px'>{verdict_cd}</div>
            <div style='color:#6b7280;font-size:12px'>
            Akcja: "{action_cd}" · Wynik: {passed}/4 ✅
            </div>
            </div>""", unsafe_allow_html=True)

        # Tabela wzorców decyzji chaotycznych
        st.markdown("#### 📋 Wzorce Decyzji: Chaotyczne vs Antychaotyczne")
        patterns = [
            ("All-in inwestycja", "Całe oszczędności na jeden zakład",
             "Barbell: 80% bezpieczne + 20% asymetryczne"),
            ("Perfekcyjna prognoza", "Czekam aż 'będę wiedział co się stanie'",
             "Buduję odporność na różne scenariusze"),
            ("Optymalizacja pod jeden scenariusz", "Planuję na 'najlepszy przypadek'",
             "Planuję na 'co jeśli się mylę?'"),
            ("Wielka jednorazowa zmiana", "'Od jutra zaczynam nowe życie'",
             "Małe, codzienne nieodwracalne działania"),
            ("Reagowanie na szum", "Sprawdzam wyniki co godzinę",
             "Ustalam zasady i ignoruję krótkoterminowy szum"),
        ]

        df_patterns = pd.DataFrame(patterns, columns=["Decyzja", "❌ Myślenie liniowe", "✅ Antychaotyczne"])
        st.dataframe(
            df_patterns,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Decyzja": st.column_config.TextColumn("Sytuacja", width=150),
                "❌ Myślenie liniowe": st.column_config.TextColumn(width=250),
                "✅ Antychaotyczne": st.column_config.TextColumn(width=250),
            }
        )

        # Codzienny rytuał antychaotyczny
        st.markdown("### 📅 Codzienny Rytuał Antychaotyczny")
        st.markdown(f"""<div style='{CARD_GREEN}'>
        <div style='{H3}'>🌅 Protokół Antychaotyczny — Codzienne 3 Pytania</div>
        <p style='{NOTE}'>
        <b style='color:#00e676'>RANO:</b> "Jaką JEDNĄ małą, odwracalną akcję wykonam dziś 
        która buduje moją pozycję na chaos?" (Nie: "co osiągnę dziś?")<br><br>
        <b style='color:#ffea00'>W CIĄGU DNIA:</b> "Czy reaguję na szum (chaos krótkoterminowy) 
        czy na sygnał (trend długoterminowy)?" Jeśli nie wiesz — to szum.<br><br>
        <b style='color:#00ccff'>WIECZOREM:</b> "Czy mój margines bezpieczeństwa jest dziś 
        większy czy mniejszy niż rano? (gotówka, energia, relacje, reputacja)"<br><br>
        <b>Nie oceniaj wyniku. Oceniaj proces i margines.</b>
        </p>
        </div>""", unsafe_allow_html=True)


# ── Stopka ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""<div style='text-align:center;color:#4a4e6a;font-size:11px;padding:8px 0'>
🌀 Chaos Deterministyczny v1.0 · Symulacja RK4 · Wykładnik Lapunova · Antykruchość Taleba<br>
Lorenz (1963) · Li &amp; Yorke (1975) · Taleb (2012) · Gleick (1987)
</div>""", unsafe_allow_html=True)

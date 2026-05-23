"""48_Stochastic_Errors.py — Błędy Addytywne i Multiplikatywne w Procesach Stochastycznych"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, gaussian_kde
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Błędy w Procesach Stochastycznych",
    subtitle="Addytywne vs Multiplikatywne · Fokker-Planck · Noise-Induced Transitions · Stochastic Resonance",
    icon="🎲", badge="Stochastic Analysis"
), unsafe_allow_html=True)

# ─── STAŁE STYLOWANIA ────────────────────────────────────────────────────────
CARD  = ("background:linear-gradient(135deg,#0f111a,#1a1c28);"
         "border:1px solid #2a2a3a;border-radius:14px;padding:18px 20px;margin-bottom:10px")
CARD_AE  = ("background:linear-gradient(135deg,#0f1a14,#1a281c);"
             "border:1px solid #00e67640;border-radius:14px;padding:18px 20px;margin-bottom:10px")
CARD_ME  = ("background:linear-gradient(135deg,#1a120f,#281c1a);"
             "border:1px solid #ff174440;border-radius:14px;padding:18px 20px;margin-bottom:10px")
H3    = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:8px"
H3_AE = "color:#00e676;font-size:14px;font-weight:700;margin-bottom:6px"
H3_ME = "color:#ff1744;font-size:14px;font-weight:700;margin-bottom:6px"
FORMULA = ("background:#0a0b10;border:1px solid #2a2a3a;border-radius:8px;"
           "padding:12px 16px;font-family:JetBrains Mono,monospace;font-size:13px;"
           "color:#00ccff;margin:8px 0")

CLR_CLEAN = "#aaaaaa"
CLR_AE    = "#00e676"
CLR_ME    = "#ff1744"
def hex_to_rgba(hex_code, alpha):
    h = hex_code.lstrip('#')
    if len(h) != 6: return hex_code
    return f"rgba({int(h[:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Parametry Symulacji")
    n_paths = st.slider("Liczba ścieżek", 20, 300, 80, 10,
                        help="Więcej ścieżek = dokładniejsze histogramy, wolniejsza symulacja")
    T_sim   = st.slider("Horyzont T", 1.0, 20.0, 5.0, 0.5)
    dt_opt  = st.selectbox("Krok dt", [0.005, 0.01, 0.02], index=1,
                           format_func=lambda x: f"{x:.3f}")
    seed    = st.number_input("Seed (reprodukowalność)", 0, 9999, 42)
    st.divider()
    st.markdown("### 📊 Poziomy Szumu")
    sigma_base = st.slider("σ bazowy procesu",   0.05, 1.0,  0.20, 0.05)
    sigma_ae   = st.slider("σ błędu addytywnego (AE)",   0.0,  1.0, 0.15, 0.05,
                           help="Stały szum niezależny od stanu X")
    sigma_me   = st.slider("σ błędu multiplikat. (ME)", 0.0,  1.0, 0.20, 0.05,
                           help="Szum skaluje się z wartością X")
    st.divider()
    st.markdown("### 📐 Model Liniowy")
    linear_model = st.selectbox("Wybierz model", ["ou", "gbm"],
                                format_func=lambda x: {"ou": "Ornstein-Uhlenbeck", "gbm": "GBM (rynek)"}[x])
    mu_param    = st.slider("μ (dryf/średnia)", -0.5, 1.0, 0.05, 0.05)
    theta_param = st.slider("θ (szybkość powrotu, OU)", 0.1, 10.0, 2.0, 0.1)
    st.divider()
    st.markdown("### 🌀 Model Bistabilny")
    a_param = st.slider("a (liniowy, studnia)", 0.1, 3.0, 1.0, 0.1)
    b_param = st.slider("b (kubiczny, bariery)", 0.1, 3.0, 1.0, 0.1)
    st.caption(f"Minima potencjału: x* = ±{np.sqrt(a_param/b_param):.2f}")

# ─── ZAKŁADKI ────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🎓 Teoria",
    "📈 Procesy Liniowe",
    "🌀 Procesy Nieliniowe",
    "🔔 Stochastic Resonance",
    "📐 Fokker-Planck",
    "⚡ Konsekwencje"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — TEORIA
# ════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## 🎓 Fundamenty Teoretyczne")
    st.markdown(
        "Każdy rzeczywisty pomiar, model i dane zawierają błędy. "
        "Sposób, w jaki błąd jest **powiązany ze stanem układu**, determinuje wszystkie jego konsekwencje."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div style='{CARD_AE}'>
        <div style='{H3_AE}'>🟢 Błąd Addytywny (Additive Error, AE)</div>
        <div style='{FORMULA}'>dX = f(X,t)dt + σ_base·dW + <b>σ_AE·dW_noise</b></div>
        <b>Właściwości:</b><br>
        • Amplituda szumu <b>stała</b> — niezależna od X<br>
        • Poszerzenie rozkładu symetryczne<br>
        • Brak dodatkowego biasu (Itô: koryguje tylko dla ME)<br>
        • Wariancja rośnie addytywnie: Var_tot = Var_base + σ²_AE<br><br>
        <b>Przykłady fizyczne:</b><br>
        • Szum termiczny (Johnson-Nyquist)<br>
        • Błąd kwantyzacji ADC<br>
        • Bid-ask spread na rynku (mała wartość, stała)<br>
        • Szum pomiarowy termometru elektronicznego
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""<div style='{CARD_ME}'>
        <div style='{H3_ME}'>🔴 Błąd Multiplikatywny (Multiplicative Error, ME)</div>
        <div style='{FORMULA}'>dX = f(X,t)dt + σ_base·dW + <b>σ_ME·X·dW_extra</b></div>
        <b>Właściwości:</b><br>
        • Amplituda szumu <b>skaluje się z X</b><br>
        • Może wywoływać asymetrię i bimodalność rozkładu<br>
        • Korekcja Itô: dodaje człon dryfu <b>+½σ²_ME·X</b><br>
        • Wariancja narasta wraz z X: Var_ME(X) = σ²_ME·X²<br><br>
        <b>Przykłady fizyczne:</b><br>
        • Relatywny błąd pomiaru (% od wartości)<br>
        • Zmienność zmienności (vol-of-vol) w finansach<br>
        • Szum biologiczny w genetyce (płodność ∝ populacja)<br>
        • Błąd estymacji ryzyka (proporcjonalny do ekspozycji)
        </div>""", unsafe_allow_html=True)

    # Wykres koncepcyjny
    st.markdown("### 📊 Jak szum skaluje się z wartością X?")
    from modules.stochastic_errors import compute_noise_scaling_demo
    demo = compute_noise_scaling_demo(sigma_ae=sigma_ae if sigma_ae > 0 else 0.3,
                                     sigma_me=sigma_me if sigma_me > 0 else 0.3)
    fig_scale = go.Figure()
    fig_scale.add_trace(go.Scatter(
        x=demo["x"], y=demo["noise_ae"],
        name="Błąd Addytywny (AE) — stały",
        line=dict(color=CLR_AE, width=3),
        fill="tozeroy", fillcolor="rgba(0,230,118,0.06)"
    ))
    fig_scale.add_trace(go.Scatter(
        x=demo["x"], y=demo["noise_me"],
        name="Błąd Multiplikatywny (ME) — rośnie z X",
        line=dict(color=CLR_ME, width=3),
        fill="tozeroy", fillcolor="rgba(255,23,68,0.06)"
    ))
    cross = sigma_ae / sigma_me if sigma_me > 0 else None
    if cross and 0 < cross < 5:
        fig_scale.add_vline(x=cross, line_dash="dash", line_color="#ffea00",
                            annotation_text=f"ME>AE gdy X>{cross:.2f}",
                            annotation_font_color="#ffea00", annotation_font_size=11)
    fig_scale.update_layout(
        title="Amplituda szumu w funkcji stanu X",
        xaxis=dict(title="Wartość X", gridcolor="#1c1c2e"),
        yaxis=dict(title="Amplituda szumu σ(X)", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), height=330,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=50, r=20, t=50, b=50)
    )
    st.plotly_chart(fig_scale, use_container_width=True)

    # Tabela porównawcza
    st.markdown("### 🔬 Porównanie matematyczne i konsekwencje")
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>📐 Równania SDE i ich własności</div>
    <table style='width:100%;color:#ddd;font-size:13px;border-collapse:collapse'>
    <tr style='background:#0a0b10;'>
        <th style='padding:8px;color:#aaa;text-align:left;border-bottom:1px solid #2a2a3a'>Cecha</th>
        <th style='padding:8px;color:{CLR_AE};text-align:center;border-bottom:1px solid #2a2a3a'>Addytywny (AE)</th>
        <th style='padding:8px;color:{CLR_ME};text-align:center;border-bottom:1px solid #2a2a3a'>Multiplikatywny (ME)</th>
    </tr>
    <tr>
        <td style='padding:8px;border-bottom:1px solid #1c1c2e'>Definicja SDE</td>
        <td style='padding:8px;text-align:center;border-bottom:1px solid #1c1c2e;font-family:monospace;color:#00ccff'>+σ_AE·dW</td>
        <td style='padding:8px;text-align:center;border-bottom:1px solid #1c1c2e;font-family:monospace;color:#00ccff'>+σ_ME·g(X)·dW</td>
    </tr>
    <tr>
        <td style='padding:8px;border-bottom:1px solid #1c1c2e'>Amplituda szumu</td>
        <td style='padding:8px;text-align:center;border-bottom:1px solid #1c1c2e'>Stała σ_AE</td>
        <td style='padding:8px;text-align:center;border-bottom:1px solid #1c1c2e'>Rośnie z X: σ_ME·|X|</td>
    </tr>
    <tr>
        <td style='padding:8px;border-bottom:1px solid #1c1c2e'>Korekcja Itô</td>
        <td style='padding:8px;text-align:center;border-bottom:1px solid #1c1c2e'>Brak (D nie zależy od X)</td>
        <td style='padding:8px;text-align:center;border-bottom:1px solid #1c1c2e'>+½σ²_ME·g·g' (zmienia dryf!)</td>
    </tr>
    <tr>
        <td style='padding:8px;border-bottom:1px solid #1c1c2e'>Rozkład stacjonarny</td>
        <td style='padding:8px;text-align:center;border-bottom:1px solid #1c1c2e'>Szerszy (zachowany kształt)</td>
        <td style='padding:8px;text-align:center;border-bottom:1px solid #1c1c2e'>Może być bimodalny!</td>
    </tr>
    <tr>
        <td style='padding:8px;border-bottom:1px solid #1c1c2e'>Przejścia fazowe</td>
        <td style='padding:8px;text-align:center;border-bottom:1px solid #1c1c2e'>❌ Nie wywołuje</td>
        <td style='padding:8px;text-align:center;border-bottom:1px solid #1c1c2e'>✅ Noise-Induced Transitions</td>
    </tr>
    <tr>
        <td style='padding:8px;border-bottom:1px solid #1c1c2e'>Bias (odchylenie E[X])</td>
        <td style='padding:8px;text-align:center;border-bottom:1px solid #1c1c2e'>Brak biasu</td>
        <td style='padding:8px;text-align:center;border-bottom:1px solid #1c1c2e'>Systematyczny bias przez korektę Itô</td>
    </tr>
    <tr>
        <td style='padding:8px'>Zastosowanie</td>
        <td style='padding:8px;text-align:center'>Szum mierzący, termiczny, cyfrowy</td>
        <td style='padding:8px;text-align:center'>Finanse, biologia, turbulencja</td>
    </tr>
    </table>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>💡 Korekcja Itô — Dlaczego ME zmienia wartość oczekiwaną?</div>
    Dla procesu <code>dX = f(X)dt + g(X)·dW</code>, lemat Itô daje:<br><br>
    <div style='{FORMULA}'>E[dX] = f(X)dt + ½·g(X)·g'(X)·dt</div>
    Ten dodatkowy człon <b>½·g·g'·dt</b> pojawia się TYLKO gdy g zależy od X (błąd multiplikatywny).<br>
    Dla AE: g(X)=const → g'=0 → brak korekty.<br>
    Dla ME: g(X)=σ·X → g'=σ → korekta = <b>+½σ²_ME·X·dt</b><br><br>
    <i>W finansach: to jest różnica między GBM w notacji Itô (finanse) a Stratonovicha (fizyka). 
    Model Black-Scholes używa Itô — dlatego cena oczekiwana to e^(μt), nie e^((μ+σ²/2)t).</i>
    </div>""", unsafe_allow_html=True)

    # Literatura
    with st.expander("📚 Podstawowa literatura naukowa"):
        st.markdown("""
| Autor | Rok | Dzieło | Temat |
|-------|-----|--------|-------|
| Gardiner, C.W. | 2009 | *Handbook of Stochastic Methods* (4th ed.) | Pełna teoria SDE, Fokker-Planck |
| Horsthemke & Lefever | 1984 | *Noise-Induced Transitions* | Przejścia fazowe wywołane szumem ME |
| Gammaitoni et al. | 1998 | *Rev. Mod. Phys. 70, 223* | Stochastic Resonance — przegląd |
| Risken, H. | 1989 | *The Fokker-Planck Equation* | Rozkłady stacjonarne |
| Arnold, L. | 1998 | *Random Dynamical Systems* | Wykładniki Lapunova, chaos stochastyczny |
| Van Kampen, N.G. | 1992 | *Stochastic Processes in Physics and Chemistry* | Ekspansja systemowa |
| Benzi et al. | 1981 | *J. Phys. A: Math. Gen. 14, L453* | Pierwsze SR w klimatologii |
| Black & Scholes | 1973 | *J. Pol. Econ. 81, 637* | GBM w finansach (ME z natury) |
        """)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROCESY LINIOWE
# ════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(f"## 📈 Procesy Liniowe: {'Ornstein-Uhlenbeck' if linear_model=='ou' else 'GBM'}")

    from modules.stochastic_errors import run_linear_process

    with st.spinner("Symulacja ścieżek..."):
        res = run_linear_process(
            model=linear_model,
            n_paths=n_paths,
            T=T_sim,
            dt=dt_opt,
            mu=mu_param,
            theta=theta_param,
            sigma_base=sigma_base,
            sigma_ae=sigma_ae,
            sigma_me=sigma_me,
            x0=1.0 if linear_model == "gbm" else 0.0,
            seed=int(seed),
        )

    times = res.times
    N_show = min(40, n_paths)  # ogranicz wyświetlane ścieżki

    # ── Główny wykres ścieżek ─────────────────────────────────────────────
    fig_paths = make_subplots(rows=1, cols=3,
                              subplot_titles=["🔵 Czysty (bez błędu)", "🟢 + Błąd Addytywny (AE)", "🔴 + Błąd Multiplikatywny (ME)"])

    def add_fan_chart(fig, paths, times, color_hex, col):
        """Dodaje fan chart z mediana i percentylami."""
        p5  = np.percentile(paths, 5,  axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        p95 = np.percentile(paths, 95, axis=0)

        # Przykładowe ścieżki
        for i in range(min(15, paths.shape[0])):
            fig.add_trace(go.Scatter(x=times, y=paths[i],
                                     line=dict(color=color_hex, width=0.4),
                                     opacity=0.25, showlegend=False), row=1, col=col)
        # Percentyle
        fig.add_trace(go.Scatter(x=np.concatenate([times, times[::-1]]),
                                  y=np.concatenate([p95, p5[::-1]]),
                                  fill="toself", fillcolor=hex_to_rgba(color_hex, 0.10),
                                  line=dict(width=0), showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(x=np.concatenate([times, times[::-1]]),
                                  y=np.concatenate([p75, p25[::-1]]),
                                  fill="toself", fillcolor=hex_to_rgba(color_hex, 0.20),
                                  line=dict(width=0), showlegend=False), row=1, col=col)
        # Mediana
        fig.add_trace(go.Scatter(x=times, y=p50, line=dict(color=color_hex, width=2.5),
                                  name="Mediana", showlegend=False), row=1, col=col)

    add_fan_chart(fig_paths, res.paths_clean, times, "#aaaaaa", 1)
    add_fan_chart(fig_paths, res.paths_ae,    times, CLR_AE,    2)
    add_fan_chart(fig_paths, res.paths_me,    times, CLR_ME,    3)

    fig_paths.update_layout(
        height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig_paths.update_xaxes(gridcolor="#1c1c2e", title_text="Czas t")
    fig_paths.update_yaxes(gridcolor="#1c1c2e", title_text="X(t)")
    st.plotly_chart(fig_paths, use_container_width=True)

    # ── Histogram końcowy ─────────────────────────────────────────────────
    st.markdown("### 📊 Rozkład końcowy X(T)")
    final_clean = res.paths_clean[:, -1]
    final_ae    = res.paths_ae[:, -1]
    final_me    = res.paths_me[:, -1]

    fig_hist = go.Figure()
    for vals, name, color in [
        (final_clean, "Czysty",  CLR_CLEAN),
        (final_ae,    "+ AE",    CLR_AE),
        (final_me,    "+ ME",    CLR_ME),
    ]:
        fig_hist.add_trace(go.Histogram(
            x=vals, nbinsx=50, name=name,
            marker_color=color, opacity=0.6, histnorm="probability density"
        ))
        # KDE smooth curve
        if len(vals) > 10 and vals.std() > 1e-6:
            x_kde = np.linspace(vals.min(), vals.max(), 300)
            kde = gaussian_kde(vals)
            fig_hist.add_trace(go.Scatter(
                x=x_kde, y=kde(x_kde), mode="lines",
                line=dict(color=color, width=2.5), showlegend=False
            ))

    fig_hist.update_layout(
        barmode="overlay",
        xaxis=dict(title="X(T)", gridcolor="#1c1c2e"),
        yaxis=dict(title="Gęstość prawdopodobieństwa", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), height=340,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=50, r=20, t=40, b=50)
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Metryki statystyczne ──────────────────────────────────────────────
    st.markdown("### 📋 Statystyki porównawcze")
    s = res.stats
    col1, col2, col3, col4 = st.columns(4)
    def _delta(v, ref): return f"+{v-ref:.4f}" if v >= ref else f"{v-ref:.4f}"

    with col1:
        st.metric("Bias AE", f"{s['bias_ae']:.4f}", help="Odchylenie E[X_AE] - E[X_czysty]")
        st.metric("Bias ME", f"{s['bias_me']:.4f}")
    with col2:
        st.metric("Var AE",  f"{s['variance_ae']:.4f}", help="Wariancja końcowego rozkładu")
        st.metric("Var ME",  f"{s['variance_me']:.4f}")
    with col3:
        st.metric("Skośność AE", f"{s['ae']['skewness']:.3f}",  help="Asymetria rozkładu (0=brak)")
        st.metric("Skośność ME", f"{s['me']['skewness']:.3f}")
    with col4:
        st.metric("Kurtoza AE",  f"{s['ae']['kurtosis']:.3f}",  help="Nadmiar grubości ogonów (0=normalny)")
        st.metric("Kurtoza ME",  f"{s['me']['kurtosis']:.3f}")

    # ── Wyjaśnienie ────────────────────────────────────────────────────────
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>💡 Interpretacja wyników dla {("Ornstein-Uhlenbeck" if linear_model=="ou" else "GBM")}</div>
    {'<b>OU z AE:</b> Dodanie stałego szumu poszerza rozkład stacjonarny symetrycznie. Mediana pozostaje przy μ='+str(mu_param)+'. Wariancja rośnie o σ²_AE.<br><br><b>OU z ME:</b> Szum proporcjonalny do X powoduje <b>asymetrię</b> — duże X dostają więcej szumu. Rozkład nabiera skośności. Korekcja Itô przesuwa średnią.' if linear_model=="ou" else
    '<b>GBM z AE:</b> Bazowy GBM jest już multiplikatywny (σ·S·dW). Dodanie AE wprowadza <b>addytywny szum bezwzględny</b> — przy małych cenach dominuje, przy dużych zanika. Ogon lewostronny grubieje.<br><br><b>GBM z ME:</b> Podwójny szum multiplikatywny — efektywna zmienność rośnie do √(σ²_base + σ²_ME)·S. Prawy ogon dramatycznie grubieje (log-normal z wyższą wariancją).'}
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROCESY NIELINIOWE
# ════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## 🌀 Procesy Nieliniowe")
    subtab_nl = st.tabs(["🔵 Bistabilny Potencjał", "🎢 Duffing Oscillator"])

    with subtab_nl[0]:
        st.markdown("### 🔵 Bistabilny Potencjał (Double-Well)")
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>📐 Model: Podwójna Studnia Potencjału</div>
        <div style='{FORMULA}'>V(x) = -a/2·x² + b/4·x⁴ &nbsp;|&nbsp; dX = (ax - bx³)dt + σ·dW</div>
        Parametry: a={a_param:.1f}, b={b_param:.1f} → minima potencjału przy x* = ±{np.sqrt(a_param/b_param):.2f}<br><br>
        <b>Kluczowe odkrycie Horsthemke & Lefever (1984):</b><br>
        Błąd addytywny <b>poszerza</b> rozkład, ale zachowuje bimodalność.<br>
        Błąd multiplikatywny może <b>zlikwidować bimodalność</b> — przejście fazowe wywołane szumem,
        mimo że deterministyczne parametry a, b pozostają niezmienione!
        </div>""", unsafe_allow_html=True)

        from modules.stochastic_errors import run_bistable_process

        with st.spinner("Symulacja bistabilnego procesu..."):
            res_bi = run_bistable_process(
                n_paths=min(n_paths, 200),
                T=max(T_sim, 8.0),
                dt=dt_opt,
                a=a_param, b=b_param,
                sigma_base=sigma_base,
                sigma_ae=sigma_ae,
                sigma_me=sigma_me,
                seed=int(seed),
            )

        col1, col2 = st.columns([1.6, 1])

        with col1:
            # Fan chart dla bistabilnego
            fig_bi = go.Figure()
            t_bi = res_bi.times
            show = min(30, n_paths)

            for paths, color, name in [
                (res_bi.paths_clean, CLR_CLEAN, "Czysty"),
                (res_bi.paths_ae,    CLR_AE,    "+ AE"),
                (res_bi.paths_me,    CLR_ME,    "+ ME"),
            ]:
                p50 = np.percentile(paths, 50, axis=0)
                p25 = np.percentile(paths, 25, axis=0)
                p75 = np.percentile(paths, 75, axis=0)
                for i in range(min(8, show)):
                    fig_bi.add_trace(go.Scatter(x=t_bi, y=paths[i],
                                                 line=dict(color=color, width=0.5),
                                                 opacity=0.2, showlegend=False))
                fig_bi.add_trace(go.Scatter(
                    x=np.concatenate([t_bi, t_bi[::-1]]),
                    y=np.concatenate([p75, p25[::-1]]),
                    fill="toself", fillcolor=hex_to_rgba(color, 0.15),
                    line=dict(width=0), showlegend=False
                ))
                fig_bi.add_trace(go.Scatter(x=t_bi, y=p50,
                                             line=dict(color=color, width=2.5), name=name))
            # Poziome linie minimów
            x_star = np.sqrt(a_param / b_param)
            fig_bi.add_hline(y=x_star,  line_dash="dot", line_color="#ffea00", line_width=1,
                             annotation_text=f"+x*={x_star:.2f}", annotation_font_size=10)
            fig_bi.add_hline(y=-x_star, line_dash="dot", line_color="#ffea00", line_width=1,
                             annotation_text=f"-x*={x_star:.2f}", annotation_font_size=10)
            fig_bi.add_hline(y=0, line_dash="dot", line_color="#444", line_width=1)

            fig_bi.update_layout(
                title="Trajektorie procesu bistabilnego",
                xaxis=dict(title="Czas t", gridcolor="#1c1c2e"),
                yaxis=dict(title="X(t)", gridcolor="#1c1c2e"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Inter"), height=420,
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
                margin=dict(l=50, r=20, t=50, b=50)
            )
            st.plotly_chart(fig_bi, use_container_width=True)

        with col2:
            # Histogramy końcowe (kluczowa wizualizacja!)
            fig_bi_hist = go.Figure()
            for paths, name, color in [
                (res_bi.paths_clean, "Czysty",  CLR_CLEAN),
                (res_bi.paths_ae,    "+ AE",    CLR_AE),
                (res_bi.paths_me,    "+ ME",    CLR_ME),
            ]:
                final = paths[:, -1]
                if final.std() > 1e-6:
                    x_k = np.linspace(final.min() - 0.5, final.max() + 0.5, 300)
                    kde = gaussian_kde(final, bw_method=0.2)
                    fig_bi_hist.add_trace(go.Scatter(
                        x=kde(x_k), y=x_k, mode="lines", name=name,
                        line=dict(color=color, width=2.5),
                        fill="tozerox", fillcolor=hex_to_rgba(color, 0.10)
                    ))
            fig_bi_hist.add_hline(y=x_star,  line_dash="dot", line_color="#ffea00", line_width=1)
            fig_bi_hist.add_hline(y=-x_star, line_dash="dot", line_color="#ffea00", line_width=1)

            fig_bi_hist.update_layout(
                title="Rozkład X(T) — poziome PDF",
                xaxis=dict(title="Gęstość p(x)", gridcolor="#1c1c2e"),
                yaxis=dict(title="X", gridcolor="#1c1c2e"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Inter"), height=420,
                legend=dict(x=0.5, y=0.01, bgcolor="rgba(0,0,0,0.3)"),
                margin=dict(l=50, r=20, t=50, b=50)
            )
            st.plotly_chart(fig_bi_hist, use_container_width=True)

        # Wykres potencjału V(x)
        x_pot = np.linspace(-2.5, 2.5, 300)
        V_pot = -a_param/2 * x_pot**2 + b_param/4 * x_pot**4
        fig_pot = go.Figure()
        fig_pot.add_trace(go.Scatter(x=x_pot, y=V_pot, mode="lines",
                                      line=dict(color="#00ccff", width=3),
                                      fill="tozeroy", fillcolor="rgba(0,204,255,0.05)"))
        x_star = np.sqrt(a_param / b_param)
        V_star = -a_param/2 * x_star**2 + b_param/4 * x_star**4
        fig_pot.add_trace(go.Scatter(
            x=[-x_star, x_star], y=[V_star, V_star],
            mode="markers", marker=dict(color="#ffea00", size=12, symbol="circle"),
            name="Minima x*"
        ))
        fig_pot.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers",
            marker=dict(color="#ff1744", size=10, symbol="x"),
            name="Max niestabilny"
        ))
        fig_pot.update_layout(
            title=f"Potencjał V(x) = -a/2·x² + b/4·x⁴ (a={a_param:.1f}, b={b_param:.1f})",
            xaxis=dict(title="x", gridcolor="#1c1c2e"),
            yaxis=dict(title="V(x)", gridcolor="#1c1c2e"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=280,
            legend=dict(x=0.7, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_pot, use_container_width=True)

        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>🔬 Noise-Induced Transitions (Horsthemke & Lefever 1984)</div>
        Dla modelu bistabilnego z błędem multiplikatywnym, rozkład stacjonarny (Fokker-Planck) to:<br>
        <div style='{FORMULA}'>P_inf(x) ∝ 1/(σ_base² + σ²_ME·x²) · exp(2∫(ax-bx³)/(σ_base²+σ²_ME·x²)dx)</div>
        <b>Kluczowy wynik:</b> Gdy σ_ME przekroczy próg krytyczny <b>σ*_ME = √(2a/(2b+1))</b>,
        rozkład przechodzi z <b>bimodalnego → unimodalnego</b> (skupia się przy x=0!).<br><br>
        To jest <b>przejście fazowe I rodzaju wywołane szumem</b> — bez żadnej zmiany w deterministycznej
        dynamice (a, b niezmienione). Tylko szum ME decyduje o topologii rozkładu.<br><br>
        <b>Próg krytyczny:</b> σ*_ME = {np.sqrt(2*a_param/(2*b_param+1)):.3f} dla (a={a_param:.1f}, b={b_param:.1f})
        {' → 🔴 Jesteś <b>powyżej progu</b>: rozkład dąży do unimodalnego!' if sigma_me > np.sqrt(2*a_param/(2*b_param+1)) else ' → ✅ Jesteś <b>poniżej progu</b>: rozkład pozostaje bimodalny.'}
        </div>""", unsafe_allow_html=True)

    with subtab_nl[1]:
        st.markdown("### 🎢 Duffing Oscillator z Szumem")
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>📐 Model Duffinga</div>
        <div style='{FORMULA}'>dX = V·dt &nbsp;|&nbsp; dV = (-δV - αX - βX³ + F·cos(ωt))dt + noise·dW</div>
        To nieliniowy oscylator wymuszony — przy pewnych parametrach zachowuje się <b>chaotycznie</b>.<br>
        Szum AE: stały → może stabilizować lub destabilizować chaos.<br>
        Szum ME (∝|X|): zmienia strukturę orbit fazowych proporcjonalnie do amplitudy drgań.
        </div>""", unsafe_allow_html=True)

        c_d1, c_d2, c_d3 = st.columns(3)
        with c_d1:
            delta_d = st.slider("Tłumienie δ", 0.1, 1.0, 0.3, 0.05)
            alpha_d = st.slider("α (liniowy)", -2.0, 0.0, -1.0, 0.1)
        with c_d2:
            beta_d  = st.slider("β (kubiczny)", 0.1, 2.0, 1.0, 0.1)
            F_d     = st.slider("Amplituda F", 0.0, 0.8, 0.3, 0.05)
        with c_d3:
            omega_d = st.slider("Częstość ω", 0.5, 2.0, 1.2, 0.1)

        from modules.stochastic_errors import run_duffing_process

        with st.spinner("Symulacja Duffinga..."):
            dr = run_duffing_process(
                n_paths=1,
                T=min(T_sim * 6, 50.0),
                dt=0.005,
                delta=delta_d, alpha=alpha_d, beta=beta_d,
                omega=omega_d, F=F_d,
                sigma_ae=sigma_ae,
                sigma_me=sigma_me,
                seed=int(seed)
            )

        t_d = dr["times"]
        fig_duff = make_subplots(rows=1, cols=2,
                                  subplot_titles=["Trajektoria X(t)", "Orbita Fazowa (X, V)"])

        for key, color, name in [("clean", CLR_CLEAN, "Brak szumu"),
                                   ("ae",    CLR_AE,    "+ AE"),
                                   ("me",    CLR_ME,    "+ ME")]:
            X, V = dr[f"X_{key}"], dr[f"V_{key}"]
            skip = len(t_d) // 4  # pomiń transient
            fig_duff.add_trace(go.Scatter(x=t_d[skip:], y=X[skip:],
                                           line=dict(color=color, width=1.5), name=name), row=1, col=1)
            fig_duff.add_trace(go.Scatter(x=X[skip:], y=V[skip:],
                                           mode="lines", line=dict(color=color, width=0.8),
                                           showlegend=False), row=1, col=2)

        fig_duff.update_layout(
            height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=50, r=20, t=60, b=50)
        )
        fig_duff.update_xaxes(gridcolor="#1c1c2e")
        fig_duff.update_yaxes(gridcolor="#1c1c2e")
        fig_duff.update_xaxes(title_text="Czas t", row=1, col=1)
        fig_duff.update_xaxes(title_text="X", row=1, col=2)
        fig_duff.update_yaxes(title_text="X(t)", row=1, col=1)
        fig_duff.update_yaxes(title_text="V = dX/dt", row=1, col=2)
        st.plotly_chart(fig_duff, use_container_width=True)

        lyap_clean = dr["lyap_clean"]
        lyap_ae    = dr["lyap_ae"]
        lyap_me    = dr["lyap_me"]

        col_l1, col_l2, col_l3 = st.columns(3)
        def lyap_color(l): return "#00e676" if l < 0 else "#ff1744"
        with col_l1:
            st.markdown(f"""<div style='{CARD}'>
            <div style='{H3}'>🔵 Brak szumu</div>
            Lyapunov: <b style='color:{lyap_color(lyap_clean)}'>{lyap_clean:.4f}</b><br>
            {'🔴 Chaotyczny' if lyap_clean > 0 else '✅ Regularny'}</div>""", unsafe_allow_html=True)
        with col_l2:
            st.markdown(f"""<div style='{CARD_AE}'>
            <div style='{H3_AE}'>🟢 + Błąd Addytywny</div>
            Lyapunov: <b style='color:{lyap_color(lyap_ae)}'>{lyap_ae:.4f}</b><br>
            {'🔴 Chaotyczny' if lyap_ae > 0 else '✅ Regularny'}</div>""", unsafe_allow_html=True)
        with col_l3:
            st.markdown(f"""<div style='{CARD_ME}'>
            <div style='{H3_ME}'>🔴 + Błąd Multiplikatywny</div>
            Lyapunov: <b style='color:{lyap_color(lyap_me)}'>{lyap_me:.4f}</b><br>
            {'🔴 Chaotyczny' if lyap_me > 0 else '✅ Regularny'}</div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>💡 Wykładnik Lapunova a szum</div>
        Wykładnik Lapunova λ mierzy szybkość rozejścia się bliskich trajektorii:<br>
        <div style='{FORMULA}'>|δX(t)| ≈ |δX(0)| · e^(λt)</div>
        <b>λ &lt; 0:</b> układ stabilny, perturbacje zanikają → determinizm zachowany<br>
        <b>λ &gt; 0:</b> układ chaotyczny, trajektorie rozbiegają się wykładniczo<br><br>
        Szum ME zmienia efektywny wykładnik Lapunova: może <b>wzmacniać</b> lub <b>tłumić</b> chaos
        w zależności od parametrów — efekt znany z teorii losowych układów dynamicznych (Arnold 1998).
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — STOCHASTIC RESONANCE
# ════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🔔 Stochastic Resonance")
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>🔔 Odkrycie: Benzi, Sutera & Vulpiani (1981)</div>
    <b>Stochastic Resonance (SR)</b> to zjawisko, w którym <b>optymalny poziom szumu wzmacnia sygnał</b>
    zamiast go niszczyć. System bistabilny z subprogowym sygnałem wejściowym może synchronizować
    się z sygnałem TYLKO jeśli szum ma odpowiednią amplitudę.<br><br>
    <div style='{FORMULA}'>dX = (aX - bX³ + A·sin(ωt))dt + σ·dW &nbsp;&nbsp;&nbsp; [gdzie A &lt; Próg skokowy]</div>
    <b>Kluczowy wynik:</b> SNR(σ) ma <b>maksimum przy σ*</b> — ani za mały (brak przeskoków),
    ani za duży (totalne zagłuszenie) szum nie jest optymalny. <b>Właśnie to jest SR.</b>
    </div>""", unsafe_allow_html=True)

    c_sr1, c_sr2, c_sr3 = st.columns(3)
    with c_sr1:
        omega_sr = st.slider("Częstość sygnału ω", 0.05, 0.5, 0.15, 0.01)
        A_sr     = st.slider("Amplituda sygnału A", 0.1, 0.95, 0.75, 0.05,
                             help="Subprogowy: A < próg przeskoku studni ≈ 1")
    with c_sr2:
        n_sigma_sr = st.slider("Rozdzielczość σ", 10, 40, 20)
        sigma_max_sr = st.slider("Max σ", 0.5, 4.0, 2.0, 0.1)
    with c_sr3:
        T_sr = st.slider("Długość symulacji T", 50, 500, 150, 50)

    run_sr = st.button("▶ Oblicz Stochastic Resonance", type="primary", use_container_width=True)

    if run_sr or "sr_result" not in st.session_state:
        from modules.stochastic_errors import run_stochastic_resonance, get_sr_sample_trajectories
        with st.spinner("Obliczanie SNR dla zakresu σ... (może potrwać chwilę)"):
            sr = run_stochastic_resonance(
                omega_signal=omega_sr,
                A_signal=A_sr,
                a=a_param, b=b_param,
                T=float(T_sr),
                dt=0.01,
                n_sigma=n_sigma_sr,
                sigma_max=sigma_max_sr,
                seed=int(seed),
            )
            st.session_state["sr_result"] = sr
    else:
        sr = st.session_state["sr_result"]

    # ── Krzywa SNR(σ) ─────────────────────────────────────────────────────
    fig_snr = go.Figure()
    fig_snr.add_hline(y=sr.snr_clean, line_dash="dot", line_color="#888",
                       annotation_text=f"SNR bez szumu ({sr.snr_clean:.1f} dB)",
                       annotation_font_size=10)
    fig_snr.add_trace(go.Scatter(
        x=sr.sigma_range, y=sr.snr_ae,
        mode="lines+markers", name="AE: σ·dW",
        line=dict(color=CLR_AE, width=2.5),
        marker=dict(size=5)
    ))
    fig_snr.add_trace(go.Scatter(
        x=sr.sigma_range, y=sr.snr_me,
        mode="lines+markers", name="ME: σ·|X|·dW",
        line=dict(color=CLR_ME, width=2.5),
        marker=dict(size=5)
    ))
    # Zaznacz optima
    fig_snr.add_vline(x=sr.opt_sigma_ae, line_dash="dash", line_color=CLR_AE,
                       annotation_text=f"σ*_AE={sr.opt_sigma_ae:.2f}",
                       annotation_font_color=CLR_AE, annotation_font_size=10)
    fig_snr.add_vline(x=sr.opt_sigma_me, line_dash="dash", line_color=CLR_ME,
                       annotation_text=f"σ*_ME={sr.opt_sigma_me:.2f}",
                       annotation_font_color=CLR_ME, annotation_font_size=10)
    fig_snr.update_layout(
        title=f"Stochastic Resonance — SNR(σ) dla ω={omega_sr:.2f}, A={A_sr:.2f}",
        xaxis=dict(title="Amplituda szumu σ", gridcolor="#1c1c2e"),
        yaxis=dict(title="SNR [dB]", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), height=380,
        legend=dict(x=0.7, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=60, r=20, t=60, b=50)
    )
    st.plotly_chart(fig_snr, use_container_width=True)

    # ── Trajektorie przy 3 poziomach σ ────────────────────────────────────
    st.markdown("### 📡 Trajektorie dla: za mały / optymalny / za duży σ (AE)")
    from modules.stochastic_errors import get_sr_sample_trajectories

    sigma_low  = sr.sigma_range[0]
    sigma_opt  = sr.opt_sigma_ae
    sigma_high = sr.sigma_range[-1]

    sr_traj = get_sr_sample_trajectories(
        sigma_low=sigma_low, sigma_opt=sigma_opt, sigma_high=sigma_high,
        omega_signal=omega_sr, A_signal=A_sr,
        a=a_param, b=b_param,
        T=min(T_sr, 60.0), dt=0.01, seed=int(seed)
    )

    fig_traj = make_subplots(rows=3, cols=1, shared_xaxes=True,
                              subplot_titles=[
                                  f"σ = {sigma_low:.2f} (za mały — brak przeskoków)",
                                  f"σ = {sigma_opt:.2f} (optymalny — SR!)",
                                  f"σ = {sigma_high:.2f} (za duży — zagłuszenie)"
                              ],
                              vertical_spacing=0.08)

    t_sr = sr_traj["times"]
    sig  = sr_traj["signal"]
    colors_sr = ["#00ccff", CLR_AE, "#ff9800"]

    for row, (label, color) in enumerate([("low", colors_sr[0]), ("opt", colors_sr[1]), ("high", colors_sr[2])], 1):
        x_traj = sr_traj[label]
        fig_traj.add_trace(go.Scatter(x=t_sr, y=sig * 0.5, name="Sygnał",
                                       line=dict(color="#888", width=1, dash="dot"), showlegend=(row==1)), row=row, col=1)
        fig_traj.add_trace(go.Scatter(x=t_sr, y=x_traj, name=f"X(t) σ={getattr(sr, f'opt_sigma_ae') if label=='opt' else (sigma_low if label=='low' else sigma_high):.2f}",
                                       line=dict(color=color, width=1.5), showlegend=False), row=row, col=1)
        # Próg
        x_star = np.sqrt(a_param / b_param)
        fig_traj.add_hline(y=x_star,  line_dash="dot", line_color="rgba(255, 234, 32, 0.31)", row=row, col=1)
        fig_traj.add_hline(y=-x_star, line_dash="dot", line_color="rgba(255, 234, 32, 0.31)", row=row, col=1)

    fig_traj.update_layout(
        height=580, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        margin=dict(l=50, r=20, t=60, b=50)
    )
    fig_traj.update_xaxes(gridcolor="#1c1c2e", title_text="Czas t", row=3, col=1)
    fig_traj.update_yaxes(gridcolor="#1c1c2e")
    st.plotly_chart(fig_traj, use_container_width=True)

    opt_ae_v = sr.opt_sigma_ae
    opt_me_v = sr.opt_sigma_me
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>💡 Dlaczego σ*_AE ≠ σ*_ME?</div>
    Optymalne σ dla Stochastic Resonance odpowiada warunkowi Kramera (Kramers 1940):<br>
    Szybkość przeskoków R ≈ R₀·exp(-ΔV/D), gdzie D = efektywna dyfuzja.<br><br>
    Dla AE: D = σ²/2 → σ*_AE ≈ √(2ΔV) ≈ <b>{opt_ae_v:.2f}</b><br>
    Dla ME: D(x) = (σ·x)²/2 zależy od x → optymum przesuwa się, bo szum jest
    słabszy w pobliżu niestabilnej równowagi (x≈0) niż na minimach.<br>
    Wynik: <b>σ*_ME = {opt_me_v:.2f}</b> — dla ME potrzeba więcej szumu żeby osiągnąć SR!<br><br>
    <i>Zastosowania SR: detekcja sygnałów biologicznych (receptory czuciowe, neurony), 
    inżynieria (progi ADC), klimatologia (epoki lodowe jako SR Ziemi).</i>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — FOKKER-PLANCK
# ════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## 📐 Równanie Fokker-Planck")
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>📐 Fundamentalne Równanie Ewolucji Gęstości</div>
    Fokker-Planck opisuje jak zmienia się <b>rozkład prawdopodobieństwa P(x,t)</b> w czasie:<br>
    <div style='{FORMULA}'>∂P/∂t = -∂/∂x[f(x)·P] + (1/2)·∂²/∂x²[D²(x)·P]</div>
    Człon dryfu: <code>-∂/∂x[f·P]</code> — przesuwa rozkład zgodnie z deterministyczną dynamiką<br>
    Człon dyfuzji: <code>+(1/2)∂²/∂x²[D²(x)·P]</code> — szerzy rozkład<br><br>
    <b>Kluczowe różnice AE vs ME:</b><br>
    • <b>AE:</b> D(x) = σ = const → rozkład szerzy się <b>jednakowo wszędzie</b><br>
    • <b>ME:</b> D(x) = σ·|x| → rozkład szerzy się <b>silniej tam, gdzie x jest duże</b> → asymetria, bimodalność
    </div>""", unsafe_allow_html=True)

    fp_model = st.radio("Model Fokker-Planck:", ["OU (Ornstein-Uhlenbeck)", "Bistabilny"],
                        horizontal=True)
    fp_type = "ou" if "OU" in fp_model else "bistable"

    from modules.stochastic_errors import compute_fokker_planck_stationary, compute_fp_time_evolution

    with st.spinner("Obliczanie Fokker-Planck..."):
        fp = compute_fokker_planck_stationary(
            f_type=fp_type,
            theta=theta_param, mu=mu_param,
            a=a_param, b=b_param,
            sigma_base=sigma_base,
            sigma_ae=sigma_ae,
            sigma_me=sigma_me,
            x_range=(-3.5, 3.5),
            n_points=600,
        )

    # ── Rozkłady stacjonarne ──────────────────────────────────────────────
    col_fp1, col_fp2 = st.columns([1.6, 1])

    with col_fp1:
        fig_fp = go.Figure()
        for y_vals, name, color, dash in [
            (fp.p_stationary, "Czysty P∞(x)",    CLR_CLEAN, "solid"),
            (fp.p_ae,         "P∞ z AE",           CLR_AE,    "dash"),
            (fp.p_me,         "P∞ z ME",           CLR_ME,    "dot"),
        ]:
            fig_fp.add_trace(go.Scatter(
                x=fp.x_grid, y=y_vals, name=name, mode="lines",
                line=dict(color=color, width=2.5, dash=dash),
                fill="tozeroy" if name == "P∞ z ME" else None,
                fillcolor="rgba(255,23,68,0.06)"
            ))
        x_star = np.sqrt(a_param / b_param) if fp_type == "bistable" else None
        if x_star:
            fig_fp.add_vline(x=x_star,  line_dash="dot", line_color="#ffea00", line_width=1)
            fig_fp.add_vline(x=-x_star, line_dash="dot", line_color="#ffea00", line_width=1)

        fig_fp.update_layout(
            title="Rozkłady Stacjonarne P∞(x) — Fokker-Planck",
            xaxis=dict(title="x", gridcolor="#1c1c2e"),
            yaxis=dict(title="P∞(x)", gridcolor="#1c1c2e"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=400,
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_fp, use_container_width=True)

    with col_fp2:
        # Statystyki rozkładów
        def fp_stats(x, p):
            dx = x[1] - x[0]
            mean = np.trapezoid(x * p, x)
            var  = np.trapezoid((x - mean)**2 * p, x)
            m3   = np.trapezoid((x - mean)**3 * p, x)
            m4   = np.trapezoid((x - mean)**4 * p, x)
            skew = m3 / max(var**1.5, 1e-12)
            kurt = m4 / max(var**2, 1e-12) - 3.0
            return mean, np.sqrt(var), skew, kurt

        mc, sc, sk_c, ku_c = fp_stats(fp.x_grid, fp.p_stationary)
        ma, sa, sk_a, ku_a = fp_stats(fp.x_grid, fp.p_ae)
        mm, sm, sk_m, ku_m = fp_stats(fp.x_grid, fp.p_me)

        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>📊 Momenty Rozkładu</div>
        <table style='width:100%;font-size:12px;color:#ddd;border-collapse:collapse'>
        <tr style='background:#0a0b10'>
            <th style='padding:6px;text-align:left'>Miara</th>
            <th style='padding:6px;text-align:center;color:{CLR_CLEAN}'>Czysty</th>
            <th style='padding:6px;text-align:center;color:{CLR_AE}'>AE</th>
            <th style='padding:6px;text-align:center;color:{CLR_ME}'>ME</th>
        </tr>
        <tr><td style='padding:5px'>Średnia</td>
            <td style='text-align:center'>{mc:.3f}</td>
            <td style='text-align:center'>{ma:.3f}</td>
            <td style='text-align:center'>{mm:.3f}</td></tr>
        <tr><td style='padding:5px'>Std Dev</td>
            <td style='text-align:center'>{sc:.3f}</td>
            <td style='text-align:center'>{sa:.3f}</td>
            <td style='text-align:center'>{sm:.3f}</td></tr>
        <tr><td style='padding:5px'>Skośność</td>
            <td style='text-align:center'>{sk_c:.3f}</td>
            <td style='text-align:center'>{sk_a:.3f}</td>
            <td style='text-align:center'>{sk_m:.3f}</td></tr>
        <tr><td style='padding:5px'>Kurtoza</td>
            <td style='text-align:center'>{ku_c:.3f}</td>
            <td style='text-align:center'>{ku_a:.3f}</td>
            <td style='text-align:center'>{ku_m:.3f}</td></tr>
        </table></div>""", unsafe_allow_html=True)

        # Bimodalność (tylko bistabilny)
        if fp_type == "bistable":
            peaks_me = np.where(
                (fp.p_me[1:-1] > fp.p_me[:-2]) & (fp.p_me[1:-1] > fp.p_me[2:])
            )[0]
            n_peaks_me = len(peaks_me)
            bimodal_me = n_peaks_me >= 2
            st.markdown(f"""<div style='{"background:linear-gradient(135deg,#1a120f,#281c1a);border:1px solid #ff174440" if bimodal_me else CARD};border-radius:12px;padding:14px;margin-top:8px'>
            <b>Bimodalność P∞_ME:</b><br>
            {'🔴 <b>Tak!</b> ME wywoła Noise-Induced Transition' if bimodal_me else '✅ Nie — rozkład unimodalny'}
            <br>Liczba pików: {n_peaks_me}
            </div>""", unsafe_allow_html=True)

    # ── Ewolucja czasowa ──────────────────────────────────────────────────
    st.markdown("### ⏱️ Ewolucja P(x,t) w czasie (Fokker-Planck numeryczny)")

    with st.spinner("Numeryczne rozwiązanie Fokker-Planck..."):
        fp_evol = compute_fp_time_evolution(
            f_type=fp_type,
            a=a_param, b=b_param,
            sigma_base=sigma_base,
            sigma_ae=sigma_ae,
            sigma_me=sigma_me,
            theta=theta_param,
            mu=mu_param,
            x_range=(-3.0, 3.0),
            n_x=150,
            n_t_snapshots=6,
            T_total=4.0,
        )

    fig_evol = make_subplots(rows=1, cols=2,
                              subplot_titles=["Ewolucja P(x,t) — AE", "Ewolucja P(x,t) — ME"])
    n_snaps = len(fp_evol["snaps_ae"])
    colors_t = [f"hsl({int(270 * i/(n_snaps-1))},80%,60%)" for i in range(n_snaps)]

    for col_idx, (snaps, label) in enumerate([(fp_evol["snaps_ae"], "AE"), (fp_evol["snaps_me"], "ME")], 1):
        for i, (p, t_val) in enumerate(zip(snaps, fp_evol["times"])):
            show_leg = (col_idx == 1)
            fig_evol.add_trace(go.Scatter(
                x=fp_evol["x_grid"], y=p,
                mode="lines",
                line=dict(color=colors_t[i], width=1.5 if i < n_snaps - 1 else 2.5),
                name=f"t={t_val:.1f}" if show_leg else None,
                showlegend=show_leg,
                opacity=0.5 + 0.5 * i / n_snaps,
            ), row=1, col=col_idx)

    fig_evol.update_layout(
        height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)", font_size=10),
        margin=dict(l=50, r=20, t=60, b=50)
    )
    fig_evol.update_xaxes(gridcolor="#1c1c2e", title_text="x")
    fig_evol.update_yaxes(gridcolor="#1c1c2e", title_text="P(x,t)")
    st.plotly_chart(fig_evol, use_container_width=True)

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>💡 Jak czytać wykresy ewolucji?</div>
    Każda linia to P(x,t) w kolejnym momencie czasu (od fioletu = t=0 do jasnoniebieskiego = t=T).<br><br>
    <b>AE (lewy panel):</b> Rozkład rozszerza się symetrycznie i jednostajnie. Dyfuzja jednorodna w całej przestrzeni.<br>
    <b>ME (prawy panel):</b> Rozkład rozszerza się <b>nierównomiernie</b> — przy dużych |x| szybciej (D²∝x²).
    Ogony stają się cięższe, a w modelu bistabilnym może nastąpić zmiana topologii (znikanie minimum centralnego).
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — KONSEKWENCJE PRAKTYCZNE
# ════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## ⚡ Konsekwencje Praktyczne")

    subtab_c = st.tabs(["🏦 Finanse", "🔬 Biologia / Neurony", "📡 Pomiary & Inżynieria", "📋 Decyzyjna Tabela"])

    with subtab_c[0]:
        st.markdown("### 🏦 Portfel z Błędem Modelowym")
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>💹 Kontekst Finansowy</div>
        <b>Błąd addytywny (AE) w finansach:</b><br>
        • Bid-ask spread (stały koszt transakcyjny, niezależny od kursu)<br>
        • Błąd zaokrąglania ceny (tick size)<br>
        • Microstructure noise (Market Microstructure Theory, O'Hara 1995)<br><br>
        <b>Błąd multiplikatywny (ME) w finansach:</b><br>
        • Błąd estymacji zmienności (vol estimation error) — 5% błąd w σ → 5% błąd w całym portfelu<br>
        • Dźwignia finansowa (leverage) — amplifikuje zyski I straty proporcjonalnie<br>
        • Stochastic vol models (Heston 1993): vol-of-vol = ME na zmienności
        </div>""", unsafe_allow_html=True)

        c_fin1, c_fin2 = st.columns([1, 2])
        with c_fin1:
            n_yr_fin = st.slider("Horyzont inwestycji (lata)", 5, 40, 20)
            mu_fin   = st.slider("Oczekiwany dryf μ (%/rok)", 3, 15, 7) / 100
            sig_fin  = st.slider("Zmienność σ (%/rok)", 5, 35, 18) / 100
            ae_fin   = st.slider("Szum AE (microstructure)", 0.0, 0.05, 0.02, 0.005)
            me_fin   = st.slider("Szum ME (vol estimation)", 0.0, 0.15, 0.05, 0.01)

        from modules.stochastic_errors import simulate_portfolio_with_errors
        with st.spinner("Symulacja portfela..."):
            port = simulate_portfolio_with_errors(
                n_years=n_yr_fin,
                mu_true=mu_fin,
                sigma_true=sig_fin,
                sigma_ae=ae_fin,
                sigma_me=me_fin,
                initial_value=100_000.0,
                n_paths=300,
                seed=int(seed),
            )

        with col2:
            fig_port = go.Figure()
            t_y = port["times_y"]
            for paths, name, color in [
                (port["V_clean"], "Brak błędu",  CLR_CLEAN),
                (port["V_ae"],    "+ AE",         CLR_AE),
                (port["V_me"],    "+ ME",         CLR_ME),
            ]:
                p5  = np.percentile(paths, 5,  axis=0)
                p50 = np.percentile(paths, 50, axis=0)
                p95 = np.percentile(paths, 95, axis=0)
                fig_port.add_trace(go.Scatter(
                    x=np.concatenate([t_y, t_y[::-1]]),
                    y=np.concatenate([p95, p5[::-1]]),
                    fill="toself", fillcolor=hex_to_rgba(color, 0.10), line=dict(width=0), showlegend=False
                ))
                fig_port.add_trace(go.Scatter(x=t_y, y=p50, line=dict(color=color, width=2.5), name=name))

            fig_port.update_layout(
                title=f"Wartość Portfela — {n_yr_fin} lat symulacji",
                xaxis=dict(title="Czas (lata)", gridcolor="#1c1c2e"),
                yaxis=dict(title="Wartość [PLN]", gridcolor="#1c1c2e", tickformat=",.0f"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Inter"), height=400,
                legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)"),
                margin=dict(l=60, r=20, t=50, b=50)
            )
            st.plotly_chart(fig_port, use_container_width=True)

        sm_f = port["summary"]
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.metric("Mediana bez błędu", f"{sm_f['clean_median']:,.0f} PLN")
        with col_f2:
            loss_ae = sm_f["expected_loss_ae"]
            st.metric("Strata przez AE", f"{loss_ae:,.0f} PLN",
                      delta=f"{loss_ae/sm_f['clean_median']*100:.1f}%",
                      delta_color="inverse")
        with col_f3:
            loss_me = sm_f["expected_loss_me"]
            st.metric("Strata przez ME", f"{loss_me:,.0f} PLN",
                      delta=f"{loss_me/sm_f['clean_median']*100:.1f}%",
                      delta_color="inverse")

        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>💡 Dlaczego ME jest groźniejszy dla portfela?</div>
        Błąd addytywny <b>rozkłada się równomiernie</b> przez cały okres — jest jak stały koszt transakcyjny.
        Jego wpływ na wartość końcową jest przewidywalny i ograniczony.<br><br>
        Błąd multiplikatywny <b>narasta wraz z wartością portfela</b>:
        gdy portfel wzrośnie do 500k PLN, błąd 5% vol = 25k PLN rocznie.
        Po 20 latach ME może być <b>wielokrotnie groźniejszy</b> niż AE tej samej nominalnej wartości σ.<br><br>
        <i>Wniosek: Dokładna estymacja zmienności jest ważniejsza niż minimalizacja kosztów transakcyjnych!</i>
        </div>""", unsafe_allow_html=True)

    with subtab_c[1]:
        st.markdown("### 🔬 Sygnały Biologiczne")
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>🧠 Szum w Układach Biologicznych</div>
        Biologia dostarcza przykładów obu typów szumu w procesach stochastycznych:<br><br>
        <b>Szum addytywny (AE) — stały szum tła:</b><br>
        • Termiczny szum Johnsona w kanałach jonowych<br>
        • Brownowski ruch neurotransmiterów w synapsie<br>
        • Szum wzmacniaczy pomiarowych (EEG, MEG)<br><br>
        <b>Szum multiplikatywny (ME) — skalujący się:</b><br>
        • Demograficzny szum w populacji: dN = rN·dt + σ·N·dW (Verhulst-Stratonovich)<br>
        • Ekspresja genów: fluktuacje proporcjonalne do liczby cząsteczek<br>
        • Hodgkin-Huxley z szumem kanałów: im więcej kanałów, tym proporcjonalnie więcej szumu<br><br>
        <b>Stochastic Resonance w neurobiologii:</b><br>
        Receptor czuciowy (mechanoreceptor Pacciniego) wykazuje SR — subprogowy bodziec jest
        wykrywany przy odpowiednim poziomie szumu sensorycznego (Douglass et al. 1993, Science).
        </div>""", unsafe_allow_html=True)

        # Model Verhulst z ME
        st.markdown("#### 🦠 Dynamika Populacji: Verhulst z AE vs ME")
        col_b1, col_b2 = st.columns([1, 2])
        with col_b1:
            r_pop = st.slider("Wskaźnik wzrostu r", 0.1, 2.0, 0.5, 0.1)
            K_pop = st.slider("Pojemność środowiska K", 100, 2000, 500, 50)
            N0    = st.slider("Populacja początkowa N₀", 10, 200, 50, 10)

        rng_bio = np.random.default_rng(int(seed) + 99)
        T_bio, dt_bio = 15.0, 0.05
        n_bio = int(T_bio / dt_bio)
        t_bio = np.linspace(0, T_bio, n_bio + 1)
        sqrt_dt_bio = np.sqrt(dt_bio)

        def sim_pop(sigma_noise, use_me):
            N = np.zeros(n_bio + 1)
            N[0] = N0
            for i in range(n_bio):
                dW = rng_bio.standard_normal() * sqrt_dt_bio
                drift = r_pop * N[i] * (1 - N[i] / K_pop)
                noise = (sigma_noise * N[i] if use_me else sigma_noise) * dW
                N[i+1] = max(1.0, N[i] + drift * dt_bio + noise)
            return N

        with col_b2:
            fig_pop = go.Figure()
            # Deterministyczny Verhulst
            N_det = np.zeros(n_bio + 1)
            N_det[0] = N0
            for i in range(n_bio):
                N_det[i+1] = N_det[i] + r_pop * N_det[i] * (1 - N_det[i] / K_pop) * dt_bio
            fig_pop.add_trace(go.Scatter(x=t_bio, y=N_det, name="Deterministyczny",
                                          line=dict(color=CLR_CLEAN, width=2, dash="dot")))

            for j in range(5):
                N_ae = sim_pop(sigma_base * K_pop * 0.1, False)
                N_me = sim_pop(sigma_base * 0.5, True)
                showl = (j == 0)
                fig_pop.add_trace(go.Scatter(x=t_bio, y=N_ae, name="+ AE",
                                              line=dict(color=CLR_AE, width=1),
                                              opacity=0.5, showlegend=showl))
                fig_pop.add_trace(go.Scatter(x=t_bio, y=N_me, name="+ ME (demograficzny)",
                                              line=dict(color=CLR_ME, width=1),
                                              opacity=0.5, showlegend=showl))

            fig_pop.add_hline(y=K_pop, line_dash="dot", line_color="#888",
                              annotation_text=f"K={K_pop}", annotation_font_size=10)
            fig_pop.update_layout(
                title="Dynamika populacji Verhulst (logistyczna) z szumem",
                xaxis=dict(title="Czas", gridcolor="#1c1c2e"),
                yaxis=dict(title="Populacja N(t)", gridcolor="#1c1c2e"),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="Inter"), height=370,
                legend=dict(x=0.6, y=0.1, bgcolor="rgba(0,0,0,0.3)"),
                margin=dict(l=60, r=20, t=50, b=50)
            )
            st.plotly_chart(fig_pop, use_container_width=True)

    with subtab_c[2]:
        st.markdown("### 📡 Modele Pomiarowe")
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>⚙️ Inżynieria Pomiarowa: Kiedy AE, kiedy ME?</div>
        <div style='{FORMULA}'>
        AE: Y = X + ε &nbsp;&nbsp;&nbsp; [błąd stały: ε ~ N(0, σ²_AE)]<br>
        ME: Y = X·(1 + η) &nbsp; [błąd relatywny: η ~ N(0, σ²_ME)]
        </div>
        </div>""", unsafe_allow_html=True)

        x_vals = np.linspace(0.1, 10.0, 300)
        sigma_ae_demo = sigma_ae if sigma_ae > 0 else 0.2
        sigma_me_demo = sigma_me if sigma_me > 0 else 0.1

        err_ae = sigma_ae_demo * np.ones_like(x_vals)
        err_me = sigma_me_demo * x_vals
        err_ae_rel = err_ae / x_vals * 100
        err_me_rel = err_me / x_vals * 100

        fig_meas = make_subplots(rows=1, cols=2,
                                  subplot_titles=["Błąd bezwzględny |δY|", "Błąd relatywny |δY/X| [%]"])
        for color, ae_v, me_v, row in [
            (None, err_ae, err_me, 1),
            (None, err_ae_rel, err_me_rel, 2),
        ]:
            fig_meas.add_trace(go.Scatter(x=x_vals, y=ae_v, name="AE",
                                           line=dict(color=CLR_AE, width=2.5),
                                           showlegend=(row == 1)), row=1, col=row)
            fig_meas.add_trace(go.Scatter(x=x_vals, y=me_v, name="ME",
                                           line=dict(color=CLR_ME, width=2.5),
                                           showlegend=(row == 1)), row=1, col=row)

        fig_meas.update_layout(
            height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"),
            margin=dict(l=50, r=20, t=60, b=50),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.3)")
        )
        fig_meas.update_xaxes(title_text="Wartość mierzona X", gridcolor="#1c1c2e")
        fig_meas.update_yaxes(gridcolor="#1c1c2e")
        st.plotly_chart(fig_meas, use_container_width=True)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(f"""<div style='{CARD_AE}'>
            <div style='{H3_AE}'>✅ Typowe urządzenia z błędem AE:</div>
            • ADC (przetwornik A/C) — błąd kwantyzacji stały<br>
            • Termometr RTD — offset stały (kalibracja punktu 0)<br>
            • Wagi laboratoryjne — szum tary<br>
            • Giroskop MEMS — bias stały (zero-rate offset)<br>
            • Mikrofon — szum termiczny przy ciszy
            </div>""", unsafe_allow_html=True)

        with col_m2:
            st.markdown(f"""<div style='{CARD_ME}'>
            <div style='{H3_ME}'>⚠️ Typowe urządzenia z błędem ME:</div>
            • Termometr termopara — nieliniowość ∝ T<br>
            • Oscyloskop — błąd przy wysokich napięciach (% od wartości)<br>
            • GPS — błąd efemeryd skaluje się z odległością<br>
            • Radar — błąd odbicia ∝ odległość² (radar equation)<br>
            • Akcelerometr przy dużych przyspieszeniach
            </div>""", unsafe_allow_html=True)

    with subtab_c[3]:
        st.markdown("### 📋 Kiedy używać jakiego modelu błędu?")
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>🎯 Tabela decyzyjna: AE vs ME</div>
        <table style='width:100%;font-size:13px;color:#ddd;border-collapse:collapse'>
        <tr style='background:#0a0b10'>
            <th style='padding:8px;text-align:left;border-bottom:1px solid #2a2a3a'>Pytanie diagnostyczne</th>
            <th style='padding:8px;text-align:center;border-bottom:1px solid #2a2a3a;color:{CLR_AE}'>→ AE</th>
            <th style='padding:8px;text-align:center;border-bottom:1px solid #2a2a3a;color:{CLR_ME}'>→ ME</th>
        </tr>
        <tr><td style='padding:7px;border-bottom:1px solid #1c1c2e'>Czy błąd rośnie z wartością X?</td>
            <td style='text-align:center'>❌ Nie (stały)</td>
            <td style='text-align:center'>✅ Tak (relatywny)</td></tr>
        <tr><td style='padding:7px;border-bottom:1px solid #1c1c2e'>Typ spec. przyrządu?</td>
            <td style='text-align:center'>±X (absolutny)</td>
            <td style='text-align:center'>±X% (procentowy)</td></tr>
        <tr><td style='padding:7px;border-bottom:1px solid #1c1c2e'>Rozkład błędu przy X→0?</td>
            <td style='text-align:center'>Ten sam co dla dużych X</td>
            <td style='text-align:center'>Zanika (szum→0)</td></tr>
        <tr><td style='padding:7px;border-bottom:1px solid #1c1c2e'>Błąd na log-skali?</td>
            <td style='text-align:center'>Zmienia się z X</td>
            <td style='text-align:center'>Stały (to jest definicja ME)</td></tr>
        <tr><td style='padding:7px;border-bottom:1px solid #1c1c2e'>Histogram residuów Y-X:</td>
            <td style='text-align:center'>Symetryczny gaussian</td>
            <td style='text-align:center'>Skośny / log-normalny</td></tr>
        <tr><td style='padding:7px;border-bottom:1px solid #1c1c2e'>Test Breusch-Pagan?</td>
            <td style='text-align:center'>p > 0.05 (homosked.)</td>
            <td style='text-align:center'>p < 0.05 (heterosked.)</td></tr>
        <tr><td style='padding:7px'>Główny efekt błędu?</td>
            <td style='text-align:center'>Poszerza rozkład symetrycznie</td>
            <td style='text-align:center'>Zniekształca kształt, bias, grube ogony</td></tr>
        </table></div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>⚠️ Konsekwencje złego wyboru modelu błędu</div>
        <table style='width:100%;font-size:13px;color:#ddd;border-collapse:collapse'>
        <tr style='background:#0a0b10'>
            <th style='padding:8px;text-align:left;border-bottom:1px solid #2a2a3a'>Błąd modelowania</th>
            <th style='padding:8px;text-align:left;border-bottom:1px solid #2a2a3a'>Konsekwencja</th>
            <th style='padding:8px;text-align:center;border-bottom:1px solid #2a2a3a'>Waga</th>
        </tr>
        <tr><td style='padding:7px;border-bottom:1px solid #1c1c2e'>Założenie AE gdy rzeczywisty ME</td>
            <td>Niedoszacowanie ryzyka przy dużych X → <b>nagłe straty</b></td>
            <td style='text-align:center'>🔴 Krytyczna</td></tr>
        <tr><td style='padding:7px;border-bottom:1px solid #1c1c2e'>Założenie ME gdy rzeczywisty AE</td>
            <td>Przeszacowanie ryzyka przy małych X → <b>zbędne koszty hedgingu</b></td>
            <td style='text-align:center'>🟡 Umiarkowana</td></tr>
        <tr><td style='padding:7px;border-bottom:1px solid #1c1c2e'>Ignorowanie Itô correction dla ME</td>
            <td>Systematyczny błąd dryfu → <b>błędne oczekiwania wartości</b></td>
            <td style='text-align:center'>🔴 Krytyczna</td></tr>
        <tr><td style='padding:7px;border-bottom:1px solid #1c1c2e'>GBM z AE = stały szum abs.</td>
            <td>Ujemne ceny możliwe przy małych S → <b>niefizyczny model</b></td>
            <td style='text-align:center'>🔴 Krytyczna</td></tr>
        <tr><td style='padding:7px'>Brak korekty ME na log-skali</td>
            <td>Wnioski o regresji na log(Y) z błędem ME: <b>retransformacja błędu</b></td>
            <td style='text-align:center'>🟡 Umiarkowana</td></tr>
        </table></div>""", unsafe_allow_html=True)

        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>📌 Złote zasady</div>
        <ol style='color:#ddd;line-height:1.8'>
        <li>Zawsze sprawdzaj <b>heteroskedastyczność residuów</b> — jeśli rośnie z Y, masz ME</li>
        <li>Dla danych ekonomicznych i biologicznych <b>domyślnie zakładaj ME</b> (proporcjonalność)</li>
        <li>Korekcja Itô jest <b>obowiązkowa</b> dla ME — pominięcie = systematyczny bias E[X]</li>
        <li>Stochastic Resonance: szum AE ma inne optymum niż ME → <b>dobieraj typ szumu do układu</b></li>
        <li>Noise-Induced Transitions (<b>tylko ME</b>) — monitoruj σ_ME jako parametr bifurkacji</li>
        </ol>
        </div>""", unsafe_allow_html=True)

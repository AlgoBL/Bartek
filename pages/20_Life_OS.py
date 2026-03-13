import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from modules.styling import apply_styling

st.set_page_config(page_title="Life OS — Algorytm Łowcy", page_icon="🎯", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;margin-bottom:4px'>🎯 Life OS — Algorytm Łowcy Wielorybów</h1>
<p style='text-align:center;color:#6b7280;font-size:14px;margin-bottom:24px'>
System Operacyjny oparty na Ekonomii Ergodycznej · Neurobiologii · Teorii Gier · Prawie Potęgowym
</p>
""", unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px 20px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"
NOTE = "color:#6b7280;font-size:12px;line-height:1.6"

# ═══════════════════════════════════════════════════════════
# SEKCJA 1 — PRAWA POTĘGOWE
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📈 Sekcja 1 — Świat Potęgowy vs Gaussowski")

col1, col2 = st.columns([3, 2])

with col1:
    # Power Law vs Normal Distribution
    x = np.linspace(0.01, 10, 500)
    alpha = st.slider("Parametr α prawa potęgowego (Pareto)", 0.5, 3.0, 1.16, 0.05, key="alpha_sl")
    normal_y = (1/(np.sqrt(2*np.pi)*2)) * np.exp(-0.5*((x-5)/2)**2)
    power_y = (alpha * 0.01**alpha) / (x**( alpha+1))
    power_y = power_y / power_y.max() * normal_y.max()

    fig_pl = go.Figure()
    fig_pl.add_trace(go.Scatter(x=x, y=normal_y, name="Rozkład Normalny (Mediocristan)",
        line=dict(color="#3498db", width=2.5), fill="tozeroy",
        fillcolor="rgba(52,152,219,0.08)"))
    fig_pl.add_trace(go.Scatter(x=x, y=power_y, name=f"Prawo Potęgowe α={alpha:.2f} (Extremistan)",
        line=dict(color="#00e676", width=2.5), fill="tozeroy",
        fillcolor="rgba(0,230,118,0.08)"))
    fig_pl.update_layout(
        title="Rozkład Normalny vs Prawo Potęgowe",
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), legend=dict(x=0.4, y=0.95),
        xaxis=dict(title="Wynik", gridcolor="#1c1c2e"),
        yaxis=dict(title="Prawdopodobieństwo", gridcolor="#1c1c2e"),
        margin=dict(l=40,r=20,t=50,b=40)
    )
    fig_pl.add_vline(x=8.5, line_dash="dash", line_color="#ff1744",
        annotation_text="Outlier (Czarny Łabędź)", annotation_font_color="#ff1744")
    st.plotly_chart(fig_pl, use_container_width=True)

with col2:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>📐 Matematyka Pareto</div>
    <p style='{NOTE}'>
    W rozkładzie normalnym 80% wyników skupia się wokół średniej.<br><br>
    W środowisku potęgowym <b style='color:#00e676'>1% uczestników generuje 99% wyniku</b>.
    Jeden Czarny Łabędź przewyższa sumę wszystkich "normalnych" zdarzeń.<br><br>
    <b style='color:#ffea00'>Wzór:</b> P(X > x) = (x_min/x)^α<br><br>
    Im mniejsze α, tym "grubszy ogon" — tym bardziej liczy się outlier.<br><br>
    <b style='color:#ff1744'>Błąd liniowy:</b> Optymalizowanie pod "średni wynik" to strategia przegrana w Extremistanie.
    </p></div>""", unsafe_allow_html=True)

# Monte Carlo — funkcja schodkowa
st.markdown("### 🎲 Symulacja Monte Carlo — Funkcja Schodkowa")
np.random.seed(42)
n_agents = 200
n_periods = 100
proba_success = st.slider("Prawdopodobieństwo sukcesu na okres (%)", 1, 15, 3, key="mc_prob") / 100

paths = np.zeros((n_agents, n_periods))
for i in range(n_agents):
    wealth = 1.0
    for t in range(n_periods):
        roll = np.random.random()
        if roll < proba_success:
            wealth *= np.random.uniform(5, 50)
        else:
            wealth *= np.random.uniform(0.90, 0.99)
        paths[i, t] = wealth

fig_mc = go.Figure()
show_n = min(50, n_agents)
for i in range(show_n):
    color = "#00e676" if paths[i, -1] > 10 else "rgba(100,100,120,0.3)"
    width = 2.0 if paths[i, -1] > 10 else 0.5
    fig_mc.add_trace(go.Scatter(
        x=list(range(n_periods)), y=np.log1p(paths[i]),
        mode="lines", line=dict(color=color, width=width),
        showlegend=False, hoverinfo="skip"
    ))
winners = sum(1 for i in range(n_agents) if paths[i, -1] > 10)
fig_mc.update_layout(
    title=f"Ścieżki {n_agents} agentów (log skala) — Wieloryby: {winners}/{n_agents} ({winners/n_agents*100:.1f}%)",
    height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", family="Inter"),
    xaxis=dict(title="Okres (prób)", gridcolor="#1c1c2e"),
    yaxis=dict(title="Wartość (log)", gridcolor="#1c1c2e"),
    margin=dict(l=40,r=20,t=50,b=40)
)
st.plotly_chart(fig_mc, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# SEKCJA 2 — ERGODYCZNOŚĆ
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## ⚡ Sekcja 2 — Ergodyczność i Przetrwanie")

col_e1, col_e2 = st.columns(2)

with col_e1:
    np.random.seed(7)
    n_sim = 1000
    ens_results = []
    for _ in range(n_sim):
        w = 1.0
        for _ in range(20):
            if np.random.random() < 0.5:
                w *= 1.5
            else:
                w *= 0.6
        ens_results.append(w)

    time_path = [1.0]
    for _ in range(20):
        if np.random.random() < 0.5:
            time_path.append(time_path[-1] * 1.5)
        else:
            time_path.append(time_path[-1] * 0.6)

    fig_erg = make_subplots(rows=1, cols=2,
        subplot_titles=("Ensemble: 1000 osób × 1 próba", "Time: 1 osoba × 20 prób"))
    fig_erg.add_trace(go.Histogram(x=[min(r, 20) for r in ens_results],
        marker_color="#3498db", name="Ensemble"), row=1, col=1)
    fig_erg.add_trace(go.Scatter(x=list(range(21)), y=time_path,
        line=dict(color="#00e676", width=2.5), name="Ścieżka czasowa"), row=1, col=2)
    fig_erg.add_hline(y=1.0, line_dash="dash", line_color="#ff1744", row=1, col=2,
        annotation_text="Start=1")
    fig_erg.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white", family="Inter"),
        showlegend=False, margin=dict(l=30,r=20,t=60,b=40))
    fig_erg.update_xaxes(gridcolor="#1c1c2e")
    fig_erg.update_yaxes(gridcolor="#1c1c2e")
    st.plotly_chart(fig_erg, use_container_width=True)

with col_e2:
    ruin_threshold = st.slider("Próg ruiny (% kapitału początkowego)", 5, 50, 20, key="ruin_thr") / 100
    np.random.seed(99)
    n_paths = 300
    T = 60
    survival = []
    ruin_times = []
    for _ in range(n_paths):
        w = 1.0
        ruined = False
        for t in range(T):
            w *= np.random.choice([1.3, 0.75], p=[0.45, 0.55])
            if w < ruin_threshold:
                ruin_times.append(t)
                ruined = True
                break
        if not ruined:
            survival.append(w)

    ruin_pct = len(ruin_times) / n_paths * 100
    fig_surv = go.Figure()
    fig_surv.add_trace(go.Histogram(x=ruin_times, name="Moment ruiny",
        marker_color="#ff1744", nbinsx=20))
    fig_surv.update_layout(
        title=f"Czas do Ruiny — {ruin_pct:.0f}% agentów zbankrutowało",
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        xaxis=dict(title="Okres", gridcolor="#1c1c2e"),
        yaxis=dict(title="Liczba agentów", gridcolor="#1c1c2e"),
        margin=dict(l=40,r=20,t=50,b=40)
    )
    st.plotly_chart(fig_surv, use_container_width=True)
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>⚠️ Zasada Przetrwania (Ergodicity Economics)</div>
    <p style='{NOTE}'>
    Przy progu ruiny <b style='color:#ff1744'>{ruin_threshold*100:.0f}%</b> kapitału: <br>
    <b style='color:#ff1744'>{ruin_pct:.0f}%</b> agentów odpada zanim zobaczą wieloryba.<br><br>
    <b style='color:#00e676'>Wniosek Ole Petersa:</b> EV jest mylące w systemie nieergodycznym.
    Priorytet = uniknięcie "absorbing state" (punktu bez powrotu).
    </p></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SEKCJA 3 — DOPAMINA I RPE
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🧠 Sekcja 3 — Neurobiologia: Dopamina i Błąd Przewidywania Nagrody (RPE)")

col_d1, col_d2 = st.columns([3,2])
with col_d1:
    events = ["Brak sygnału", "Brak sygnału", "Oczekiwanie nagrody",
              "Nagroda otrzymana ✅", "Brak sygnału", "Oczekiwanie nagrody",
              "Prawie sukces ❌", "Brak sygnału", "Niespodziewana nagroda 🎉",
              "Brak sygnału", "Oczekiwanie nagrody", "Nagroda otrzymana ✅"]
    dopamine_outcome = [0, 0, 0.6, 1.0, 0, 0.6, -0.8, -0.3, 1.5, 0.1, 0.5, 0.9]
    dopamine_process = [0.3, 0.3, 0.5, 0.6, 0.4, 0.5, 0.4, 0.3, 0.7, 0.4, 0.5, 0.6]
    colors_out = ["#00e676" if v >= 0 else "#ff1744" for v in dopamine_outcome]

    fig_dop = go.Figure()
    fig_dop.add_trace(go.Bar(x=list(range(len(events))), y=dopamine_outcome,
        name="Gracz WYNIKOWY (Outcome)", marker_color=colors_out,
        customdata=events,
        hovertemplate="<b>%{customdata}</b><br>RPE: %{y:.1f}<extra></extra>"))
    fig_dop.add_trace(go.Scatter(x=list(range(len(events))), y=dopamine_process,
        name="Gracz PROCESOWY (Process KPI)",
        line=dict(color="#ffea00", width=2.5, dash="dash"),
        mode="lines+markers"))
    fig_dop.add_hline(y=0, line_color="#555", line_width=1)
    fig_dop.update_layout(
        title="Błąd Przewidywania Nagrody (RPE) — Wynik vs Proces",
        height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        xaxis=dict(ticktext=events, tickvals=list(range(len(events))),
                   tickangle=-30, gridcolor="#1c1c2e", tickfont=dict(size=9)),
        yaxis=dict(title="Poziom Dopaminy (RPE)", gridcolor="#1c1c2e"),
        legend=dict(x=0.01, y=0.99), margin=dict(l=40,r=20,t=50,b=80)
    )
    st.plotly_chart(fig_dop, use_container_width=True)

with col_d2:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>🔬 Neurobiologia Wolframa Schultza</div>
    <p style='{NOTE}'>
    Dopamina = waluta motywacji, NIE nagrody.<br><br>
    <b style='color:#ff1744'>Negative RPE</b> ("prawie sukces") = poziom dopaminy spada PONIŻEJ bazy.<br>
    Efekt: fizyczny ból, apatia, wypalenie.<br><br>
    <b style='color:#ffea00'>Hack:</b> Przepnij KPI z wyniku na <b>wykonanie procedury</b>.<br><br>
    Gracz procesowy ma stabilny poziom dopaminy niezależnie od rynkowego szumu.
    Gracz wynikowy jeździ na dopaminowej kolejce górskiej — i zazwyczaj <b style='color:#ff1744'>odpada przed wielorybem</b>.
    </p></div>
    <div style='{CARD}'>
    <div style='{H3}'>✅ Twoje Process KPI</div>
    <p style='{NOTE}'>
    ✔ Czy przeprowadziłem dziś głęboką sesję budowania pozycji?<br>
    ✔ Czy zachowałem dyscyplinę emocjonalną w kontakcie?<br>
    ✔ Czy moja rezerwa kapitałowa jest bezpieczna?<br>
    ✔ Czy nie wykonałem ruchu z pozycji desperacji?
    </p></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SEKCJA 4 — TEORIA GIER
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## ♟️ Sekcja 4 — Teoria Gier i Sygnalizacja")

col_g1, col_g2 = st.columns(2)

with col_g1:
    months_reserve = st.slider("Twoja rezerwa gotówkowa (miesiące przetrwania)", 0, 36, 12, key="batna")
    batna_score = min(100, months_reserve * 4.5 + (20 if months_reserve > 6 else 0))
    deal_quality = min(100, months_reserve * 2.8)
    desperation = max(0, 100 - months_reserve * 7)
    leverage = min(100, months_reserve * 3.5 + 10)

    fig_batna = go.Figure(go.Bar(
        x=["BATNA Score", "Jakość Dealów", "Desperacja ❌", "Leverage Negocjacyjny"],
        y=[batna_score, deal_quality, desperation, leverage],
        marker_color=["#00e676", "#3498db", "#ff1744", "#a855f7"],
        text=[f"{v:.0f}" for v in [batna_score, deal_quality, desperation, leverage]],
        textposition="outside", textfont=dict(color="white", size=12)
    ))
    fig_batna.update_layout(
        title=f"BATNA Leverage — {months_reserve} miesięcy rezerwy",
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        yaxis=dict(range=[0,115], gridcolor="#1c1c2e"),
        xaxis=dict(gridcolor="#1c1c2e"),
        margin=dict(l=20,r=20,t=50,b=40), showlegend=False
    )
    st.plotly_chart(fig_batna, use_container_width=True)

with col_g2:
    urgency = ["Niska", "Średnia", "Wysoka", "Ekstremalna"]
    availability = ["Rzadki", "Umiarkowany", "Łatwo_dostępny", "Desperacki"]
    z = [[90, 75, 45, 10],
         [80, 65, 35,  5],
         [60, 50, 25,  3],
         [30, 20, 10,  1]]
    fig_heat = go.Figure(go.Heatmap(
        z=z, x=availability, y=urgency,
        colorscale=[[0,"#ff1744"],[0.5,"#ffea00"],[1,"#00e676"]],
        text=[[f"{v}" for v in row] for row in z],
        texttemplate="%{text}",
        hovertemplate="Pilność: %{y}<br>Dostępność: %{x}<br>Postrzegana Wartość: %{z}<extra></extra>"
    ))
    fig_heat.update_layout(
        title="Wartość Postrzegana = f(Pilność × Dostępność)",
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        xaxis=dict(title="Twoja Dostępność / Desperacja"),
        yaxis=dict(title="Pilność Twojego Ruchu"),
        margin=dict(l=70,r=20,t=50,b=60)
    )
    fig_heat.add_annotation(x="Desperacki", y="Ekstremalna",
        text="☠️ Wartość=1<br>TU NIE IDZIE", showarrow=True, arrowcolor="#ff1744",
        font=dict(color="#ff1744", size=10))
    fig_heat.add_annotation(x="Rzadki", y="Niska",
        text="👑 Wartość=90<br>TU CHCESZ BYĆ", showarrow=True, arrowcolor="#00e676",
        font=dict(color="#00e676", size=10), ax=60, ay=30)
    st.plotly_chart(fig_heat, use_container_width=True)

# Game Theory Payoff Matrix
st.markdown("### 🎯 Macierz Wypłat — Cierpliwy vs Desperacki")
strategies = ["Cierpliwy (#1)", "Cierpliwy (#2)", "Desperacki (#1)", "Desperacki (#2)"]
payoffs = pd.DataFrame({
    "Wieloryb Otwiera Rozmowę": [95, 85, 40, 20],
    "Wieloryb Milczy": [70, 60, 5, 2],
    "Wieloryb Testuje": [80, 75, 15, 8],
}, index=strategies)

fig_matrix = go.Figure(go.Heatmap(
    z=payoffs.values, x=payoffs.columns, y=payoffs.index,
    colorscale=[[0,"#ff1744"],[0.5,"#ffea00"],[1,"#00e676"]],
    text=payoffs.values, texttemplate="%{text}",
))
fig_matrix.update_layout(
    title="Oczekiwana Wypłata (0-100) w każdym scenariuszu",
    height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", family="Inter"),
    margin=dict(l=130,r=20,t=50,b=60)
)
st.plotly_chart(fig_matrix, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# SEKCJA 5 — KELLY CRITERION I BARBELL
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📊 Sekcja 5 — Kelly Criterion i Barbell Strategy")

col_k1, col_k2 = st.columns(2)

with col_k1:
    win_prob = st.slider("Prawdopodobieństwo sukcesu p (%)", 1, 60, 15, key="kelly_p") / 100
    win_mult = st.slider("Mnożnik wygranej (odds b)", 2, 100, 20, key="kelly_b")
    win_mult = max(0.1, win_mult)
    kelly_f = (win_prob * win_mult - (1 - win_prob)) / win_mult
    kelly_f = max(0, kelly_f)
    half_kelly = kelly_f / 2
    
    bet_sizes = np.linspace(0, min(1.0, kelly_f * 3 if kelly_f > 0 else 0.5), 200)
    growth_rates = [win_prob * np.log(1 + win_mult * b) + (1-win_prob) * np.log(max(1e-9, 1 - b)) for b in bet_sizes]
    
    fig_kelly = go.Figure()
    fig_kelly.add_trace(go.Scatter(x=bet_sizes*100, y=growth_rates,
        line=dict(color="#00e676", width=2.5), name="Wzrost logarytmiczny"))
        
    fig_kelly.add_vline(x=kelly_f*100, line_dash="solid", line_color="#ffea00",
        annotation_text=f"Kelly={kelly_f*100:.1f}%", annotation_font_color="#ffea00")
        
    if kelly_f > 0:
        fig_kelly.add_vline(x=half_kelly*100, line_dash="dash", line_color="#3498db",
            annotation_text=f"½K={half_kelly*100:.1f}%", annotation_font_color="#3498db")
        ruin_zone = kelly_f * 2
        if ruin_zone <= 1.0:
            idx_ruin = next((i for i,b in enumerate(bet_sizes) if b > ruin_zone), len(bet_sizes)-1)
            fig_kelly.add_vrect(x0=ruin_zone*100, x1=bet_sizes[-1]*100,
                fillcolor="rgba(255,23,68,0.1)", line_width=0, annotation_text="Strefa ruiny")
                
    fig_kelly.update_layout(
        title=f"Kelly Criterion — p={win_prob*100:.0f}%, b={win_mult}x → f*={kelly_f*100:.1f}%",
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        xaxis=dict(title="Stawka (% kapitału)", gridcolor="#1c1c2e"),
        yaxis=dict(title="Oczekiwany wzrost log", gridcolor="#1c1c2e"),
        margin=dict(l=40,r=20,t=50,b=40)
    )
    st.plotly_chart(fig_kelly, use_container_width=True)
    st.markdown(f"""<div style='background:#0f111a;border:1px solid #2a2a3a;border-radius:10px;padding:12px'>
    <b style='color:#ffea00'>Optymalna stawka Kelly: {kelly_f*100:.1f}%</b>&nbsp;&nbsp;
    <b style='color:#3498db'>½ Kelly (bezpieczna): {half_kelly*100:.1f}%</b><br>
    <span style='color:#6b7280;font-size:12px'>Overbetting powyżej 2×Kelly = matematyczna droga do ruiny</span>
    </div>""", unsafe_allow_html=True)

with col_k2:
    safe_pct = st.slider("% w Bezpiecznym (Barbell)", 50, 95, 90, key="barbell_safe")
    risky_pct = 100 - safe_pct
    years = 20
    t = np.arange(years+1)
    safe_ret = 0.05
    risky_exp_ret = 0.0
    risky_std = 2.5
    np.random.seed(42)
    risky_outcomes = np.random.lognormal(risky_exp_ret, risky_std, 500)

    barbell_final = (safe_pct/100) * (1+safe_ret)**years + (risky_pct/100) * np.percentile(risky_outcomes, 90)
    middle_final = (1 + 0.09)**years
    safe_final = (1 + safe_ret)**years

    fig_bb = go.Figure()
    categories = ["Pure Safe\n(5%/rok)", f"Barbell\n({safe_pct}%+{risky_pct}%)", "Middle Ground\n(9%/rok)"]
    values = [safe_final, barbell_final, middle_final]
    colors = ["#3498db", "#00e676", "#f39c12"]
    for cat, val, col in zip(categories, values, colors):
        fig_bb.add_trace(go.Bar(x=[cat], y=[val], marker_color=col,
            text=[f"{val:.1f}x"], textposition="outside",
            textfont=dict(color="white", size=14), name=cat))
    fig_bb.update_layout(
        title=f"Barbell vs Middle Ground — po {years} latach (p90 scenariusz)",
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        yaxis=dict(title="Mnożnik kapitału (×)", gridcolor="#1c1c2e"),
        xaxis=dict(gridcolor="#1c1c2e"),
        showlegend=False, margin=dict(l=40,r=20,t=50,b=40)
    )
    st.plotly_chart(fig_bb, use_container_width=True)
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>🏋️ Zasada Sztangi (Taleb)</div>
    <p style='{NOTE}'>
    <b style='color:#00e676'>{safe_pct}% Bezpieczne</b> (gotówka, obligacje, rezerwa) —
    daje Ci "Fuck You Money" i pozycję czasową.<br><br>
    <b style='color:#ff1744'>{risky_pct}% Asymetryczne</b> (opcje, wieloryby, moonshots) —
    nieograniczony potencjał wzrostu.<br><br>
    <b style='color:#f39c12'>Unikaj ŚRODKA</b> — "umiarkowane ryzyko" to najgorsza strategia.
    </p></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SEKCJA 6 — TEORIA SIECI
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🌐 Sekcja 6 — Teoria Sieci: Dziury Strukturalne (Ronald Burt)")

try:
    import networkx as nx
    G = nx.Graph()
    cluster_A = ["Tech_1", "Tech_2", "Tech_3", "Tech_4"]
    cluster_B = ["Finance_1", "Finance_2", "Finance_3"]
    cluster_C = ["Art_1", "Art_2", "Art_3"]
    you = "YOU (Broker)"
    for nodes in [cluster_A, cluster_B, cluster_C]:
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                G.add_edge(nodes[i], nodes[j])
    for c in cluster_A:
        G.add_edge(you, c)
    for c in cluster_B:
        G.add_edge(you, c)
    for c in cluster_C:
        G.add_edge(you, c)

    pos = nx.spring_layout(G, seed=42)
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node == you:
            node_colors.append("#00e676"); node_sizes.append(30)
        elif node in cluster_A:
            node_colors.append("#3498db"); node_sizes.append(12)
        elif node in cluster_B:
            node_colors.append("#a855f7"); node_sizes.append(12)
        else:
            node_colors.append("#f39c12"); node_sizes.append(12)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_labels = list(G.nodes())

    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(255,255,255,0.1)", width=1), hoverinfo="skip"))
    fig_net.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
        text=node_labels, textposition="top center",
        textfont=dict(size=9, color="white"),
        marker=dict(color=node_colors, size=node_sizes, line=dict(color="white",width=1)),
        hovertemplate="%{text}<extra></extra>"))
    fig_net.update_layout(
        title="Graf Sieci — Ty jako Broker Dziury Strukturalnej",
        height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20,r=20,t=50,b=20)
    )
    cola, colb = st.columns([3,2])
    with cola:
        st.plotly_chart(fig_net, use_container_width=True)
    with colb:
        betweenness = nx.betweenness_centrality(G)
        you_bc = betweenness.get(you, 0)
        avg_bc = np.mean([v for k,v in betweenness.items() if k != you])
        st.markdown(f"""<div style='{CARD}'>
        <div style='{H3}'>📡 Twoja Centralność (Burt)</div>
        <p style='{NOTE}'>
        <b style='color:#00e676'>Betweenness Centrality: {you_bc:.3f}</b><br>
        Średnia pozostałych węzłów: {avg_bc:.3f}<br>
        Twoja przewaga: <b style='color:#ffea00'>{you_bc/max(avg_bc,0.001):.0f}×</b><br><br>
        Broker na "dziurze strukturalnej" kontroluje przepływ informacji między klastrami.
        Wartość rośnie <b>wykładniczo</b> z każdym kolejnym klasterm który łączysz.
        </p></div>
        <div style='{CARD}'>
        <div style='{H3}'>🎯 Klastry w Twoim Grafie</div>
        <p style='{NOTE}'>
        🔵 Tech — {len(cluster_A)} węzłów<br>
        🟣 Finance — {len(cluster_B)} węzłów<br>
        🟠 Art — {len(cluster_C)} węzłów<br><br>
        Każda nieznana osoba w nowym klastrze to potencjalna opcja call.
        </p></div>""", unsafe_allow_html=True)

        clusters_connected = np.array([1, 2, 3, 4, 5])
        broker_value = 2 ** clusters_connected
        
        fig_exp = go.Figure()
        fig_exp.add_trace(go.Scatter(x=clusters_connected, y=broker_value, mode='lines+markers',
            line=dict(color="#00e676", width=3), marker=dict(size=10, color="#ffea00")))
        fig_exp.update_layout(
            title="Exponential Value of Brokerage",
            height=200, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"),
            xaxis=dict(title="Połączone Klastry", gridcolor="#1c1c2e", dtick=1),
            yaxis=dict(title="Wartość Dziury (2^n)", gridcolor="#1c1c2e"),
            margin=dict(l=40,r=20,t=40,b=20)
        )
        st.plotly_chart(fig_exp, use_container_width=True)
except ImportError:
    st.info("Zainstaluj networkx: `pip install networkx`")

# ═══════════════════════════════════════════════════════════
# SEKCJA 7 — DAILY OS ALGORITHM
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📅 Sekcja 7 — Daily OS Algorithm — Twój Codzienny Protokół")

tabs = st.tabs(["🌅 Rano", "☀️ W ciągu dnia", "🌙 Wieczorem", "⚖️ 3 Testy Decyzyjne"])

with tabs[0]:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>🌅 PROTOKÓŁ PORANNY</div>
    </div>""", unsafe_allow_html=True)
    morning_items = [
        ("📵 Zero Reaktywności", "Nie sprawdzaj maili/wiadomości przez pierwsze 60 minut. Twój mózg ma najwyższy poziom kreatywności tuż po przebudzeniu — nie zarzynaj go reactive noise."),
        ("🏋️ Trening Fizyczny", "Obniżenie kortyzolu, wzrost BDNF (neuroplastyczność). Min. 20-30 min. To inwestycja, nie koszt czasu."),
        ("🧘 Blok Głębokiej Pracy", "2-4 godziny Deep Work (Cal Newport) nad JEDNĄ rzeczą budującą pozycję. Nie gaszenie pożarów."),
        ("☕ Bez Taniej Dopaminy", "Social media, powiadomienia, małe 'quick wins' — blokują receptory D2. Jesteś na dopaminowym poście rano."),
    ]
    for title, desc in morning_items:
        c1, c2 = st.columns([1,7])
        with c1:
            st.checkbox("", key=f"m_{title}")
        with c2:
            st.markdown(f"**{title}**  \n<span style='color:#6b7280;font-size:12px'>{desc}</span>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>☀️ PROTOKÓŁ DZIENNY — Polowanie</div>
    </div>""", unsafe_allow_html=True)
    day_items = [
        ("🔍 Filtruj Szum", "Każde spotkanie/mail przepuść przez filtr: czy ma potencjał nieliniowego zwrotu? Jeśli NIE → deleguj lub usuń."),
        ("👂 Asymetria Informacji", "Słuchaj więcej niż mówisz. Kto zadaje pytania, ten kontroluje rozmowę i zbiera wywiad."),
        ("🩺 Bądź Lekarzem", "Nie sprzedawaj. Diagnozuj. Pytaj 'Co jest Twoim największym problemem?', nie 'Czy mogę Ci coś zaproponować?'"),
        ("⚠️ Sygnał Desperacji", "Jeśli czujesz presję 'muszę to zamknąć' — STOP. Wycofaj się. To sygnał utraty pozycji negocjacyjnej."),
        ("🔗 Buduj Mosty", "Każdego dnia: czy połączyłem dwie osoby z różnych klastrów? Każde połączenie to opcja call na przyszłość."),
    ]
    for title, desc in day_items:
        c1, c2 = st.columns([1,7])
        with c1:
            st.checkbox("", key=f"d_{title}")
        with c2:
            st.markdown(f"**{title}**  \n<span style='color:#6b7280;font-size:12px'>{desc}</span>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>🌙 AUDYT WIECZORNY — Ocena Dnia</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("**Oceń swój dzień (tylko procesy, nie wyniki):**")
    c_deep = st.slider("Głęboka Praca (Deep Work)", 0, 100, 70, format="%d%%", key="ev_deep")
    c_emo = st.slider("Dyscyplina Emocjonalna", 0, 100, 80, format="%d%%", key="ev_emo")
    c_batna = st.slider("Ochrona BATNA / Rezerwy", 0, 100, 90, format="%d%%", key="ev_batna")
    c_pos = st.slider("Budowanie Pozycji", 0, 100, 60, format="%d%%", key="ev_pos")

    day_score = (c_deep + c_emo + c_batna + c_pos) / 4
    color = "#00e676" if day_score >= 70 else "#f39c12" if day_score >= 50 else "#ff1744"
    fig_day = go.Figure(go.Indicator(
        mode="gauge+number", value=day_score,
        number={"font": {"size": 48, "color": color}, "suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.15},
            "steps": [{"range":[0,50],"color":"rgba(255,23,68,0.15)"},
                      {"range":[50,70],"color":"rgba(255,234,0,0.15)"},
                      {"range":[70,100],"color":"rgba(0,230,118,0.15)"}],
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
        }
    ))
    fig_day.update_layout(height=200, margin=dict(l=30,r=30,t=20,b=20),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white", family="Inter"))
    st.plotly_chart(fig_day, use_container_width=True)
    verdict = "✅ Solidny dzień łowcy!" if day_score>=70 else "🟡 Dobry, ale jest margines." if day_score>=50 else "🔴 Zresetuj protokół jutro."
    st.markdown(f"<center><b style='color:{color};font-size:18px'>{verdict}</b></center><br><br>", unsafe_allow_html=True)

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>⏱️ PROGRESS TRACKER — Alokacja Czasu</div>
    </div>""", unsafe_allow_html=True)
    colp1, colp2, colp3 = st.columns(3)
    with colp1:
        time_dw = st.number_input("Deep Work (h)", 0.0, 16.0, 4.0, step=0.5)
    with colp2:
        time_reg = st.number_input("Regeneracja (h)", 0.0, 16.0, 2.0, step=0.5)
    with colp3:
        time_bw = st.number_input("Busy Work (h)", 0.0, 16.0, 3.0, step=0.5)
        
    if (time_dw + time_reg + time_bw) > 0:
        fig_pie = go.Figure(data=[go.Pie(labels=["Deep Work", "Regeneracja", "Busy Work"],
                                         values=[time_dw, time_reg, time_bw],
                                         hole=.4,
                                         marker=dict(colors=["#00e676", "#3498db", "#ff1744"]))])
        fig_pie.update_layout(height=260, margin=dict(l=20,r=20,t=20,b=20), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_pie, use_container_width=True)

with tabs[3]:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>⚖️ 3 TESTY DECYZYJNE — Algorytm Przed Każdym Ruchem</div>
    </div>""", unsafe_allow_html=True)

    action = st.text_input("Opisz akcję którą rozważasz:", placeholder="np. 'Wysłać follow-up do X po 3 dniach ciszy'", key="action_input")

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(f"""<div style='background:#0d0f1a;border:2px solid #ff1744;border-radius:12px;padding:16px;height:280px'>
        <div style='color:#ff1744;font-weight:700;font-size:13px;margin-bottom:8px'>🛡️ TEST 1: PRZETRWANIE</div>
        <p style='color:#aaa;font-size:12px;line-height:1.5'>
        "Czy to działanie naraża mnie na ryzyko <b>wypadnięcia z gry</b> przed wystąpieniem zdarzenia X?"<br><br>
        Czy to zużywa:<br>
        • Rezerwę kapitałową?<br>
        • Reputację?<br>
        • Energię potrzebną na kolejne 12 miesięcy ciszy?
        </p>
        </div>""", unsafe_allow_html=True)
        t1_ans = st.radio("Odpowiedź:", ["TAK → STOP", "NIE → Kontynuuj test 2"], key="t1")

    with t2:
        st.markdown(f"""<div style='background:#0d0f1a;border:2px solid #ffea00;border-radius:12px;padding:16px;height:280px'>
        <div style='color:#ffea00;font-weight:700;font-size:13px;margin-bottom:8px'>📈 TEST 2: ASYMETRIA</div>
        <p style='color:#aaa;font-size:12px;line-height:1.5'>
        "Czy to działanie buduje <b>pozycję</b> zwiększającą ekspozycję na Czarne Łabędzie, czy jest tylko liniową pracą?"<br><br>
        • Potencjał 10x-100x? → Inwestuj<br>
        • Tylko "stawka godzinowa"? → Deleguj<br>
        • Skalowalne bez limitu? → Inwestuj bez oczekiwań
        </p>
        </div>""", unsafe_allow_html=True)
        t2_ans = st.radio("Odpowiedź:", ["LINIOWE → Minimalizuj", "POTĘGOWE → Investuj w proces"], key="t2")

    with t3:
        st.markdown(f"""<div style='background:#0d0f1a;border:2px solid #a855f7;border-radius:12px;padding:16px;height:280px'>
        <div style='color:#a855f7;font-weight:700;font-size:13px;margin-bottom:8px'>♟️ TEST 3: SYGNALIZACJA</div>
        <p style='color:#aaa;font-size:12px;line-height:1.5'>
        "Czy wykonuję ten ruch z <b>pozycji siły</b> (mogę odejść) czy <b>desperacji</b> (muszę to mieć)?"<br><br>
        • Czy mogę powiedzieć "nie spieszy mi się"?<br>
        • Czy mam BATNA wystarczające do czekania?<br>
        • Despacja zeruje prawdopodobieństwo w grach o wysoką stawkę.
        </p>
        </div>""", unsafe_allow_html=True)
        t3_ans = st.radio("Odpowiedź:", ["DESPERACJA → Czekaj/Zwiększ BATNA", "SIŁA → Wykonaj ruch"], key="t3")

    if action:
        go_no_go = (t1_ans == "NIE → Kontynuuj test 2" and
                    t2_ans == "POTĘGOWE → Investuj w proces" and
                    t3_ans == "SIŁA → Wykonaj ruch")
        verdict2 = "✅ GO — Wykonaj ruch z dyscypliną procesu." if go_no_go else "🛑 NO GO — Zmień podejście lub poczekaj."
        color2 = "#00e676" if go_no_go else "#ff1744"
        st.markdown(f"""<div style='background:rgba(0,0,0,0.4);border:2px solid {color2};
            border-radius:12px;padding:16px;text-align:center;margin-top:12px'>
            <span style='font-size:20px;font-weight:700;color:{color2}'>{verdict2}</span><br>
            <span style='color:#6b7280;font-size:12px'>Akcja: "{action}"</span>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SEKCJA 8 — CHRONOBIOLOGIA I RYTMY ULTRADIANE
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🧬 Sekcja 8 — Chronobiologia i Rytmy Ultradiane (Zarządzanie Energią)")

col_c1, col_c2 = st.columns([3, 2])

with col_c1:
    peak_time = st.slider("Szczyt Melatoniny (Środek snu) - zazwyczaj 3:00 - 4:00 rano", 0.0, 24.0, 3.5, 0.5, format="%.1f h", key="melatonin_peak_time")
    # Generowanie danych do wykresu polarnego
    hours = np.linspace(0, 24, 240)
    theta = hours * 15 # 360/24
    
    # Symulacja kortyzolu i melatoniny
    # Kortyzol pikuje około 30-45 min po przebudzeniu (CAR - Cortisol Awakening Response)
    # Przebudzenie zakładamy ok. 4h po szczycie melatoniny
    wake_time = (peak_time + 4) % 24
    
    # Funkcja do generowania krzywych dobowych
    def circadian_curve(x, peak, width, amplitude, base):
        dist = np.minimum(np.abs(x - peak), 24 - np.abs(x - peak))
        return base + amplitude * np.exp(-0.5 * (dist / width)**2)
        
    melatonin = circadian_curve(hours, peak_time, 2.0, 80, 5)
    
    # Kortyzol ma dwa piki (rano silny, po południu słabszy)
    cortisol_morning_peak = (wake_time + 1) % 24
    cortisol = circadian_curve(hours, cortisol_morning_peak, 1.5, 70, 10) + circadian_curve(hours, (cortisol_morning_peak + 8)%24, 3.0, 30, 0)
    
    fig_circ = go.Figure()
    fig_circ.add_trace(go.Scatterpolar(r=melatonin, theta=theta, name="Melatonina (Odpoczynek/Naprawa)", line_color="#a855f7", fill="toself", opacity=0.6))
    fig_circ.add_trace(go.Scatterpolar(r=cortisol, theta=theta, name="Kortyzol (Akcja/Skupienie)", line_color="#f39c12", fill="toself", opacity=0.6))
    
    # Złote okno (Deep work window) - 2-4h po przebudzeniu
    deep_work_start = (wake_time + 2) % 24
    deep_work_end = (deep_work_start + 3) % 24
    
    fig_circ.update_layout(
        title="Zegar Dobowy (Panda & Huberman)",
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 100]),
            angularaxis=dict(tickfont_size=11, tickmode="array", tickvals=np.arange(0, 360, 45), ticktext=["0:00", "3:00", "6:00", "9:00", "12:00", "15:00", "18:00", "21:00"], direction="clockwise")
        ),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=400, margin=dict(t=40, b=40, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_circ, use_container_width=True)

with col_c2:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>⏱️ Twój Rytm Dobowy</div>
    <p style='{NOTE}'>
    Biologia dyktuje, kiedy powinieneś robić najtrudniejsze rzeczy.<br><br>
    🌙 <b>Pobudka (Szacowana):</b> ~{int(wake_time):02d}:{int((wake_time%1)*60):02d}<br>
    🔥 <b>Złote Okno (Deep Work):</b> {int(deep_work_start):02d}:{int((deep_work_start%1)*60):02d} - {int(deep_work_end):02d}:{int((deep_work_end%1)*60):02d}<br>
    W tym czasie Twój poziom kortyzolu i dopaminy naturalnie tworzy optymalne warunki dla neuroplastyczności.<br><br>
    <b style='color:#00e676'>Zasada 90/20 (BRAC):</b> Mózg utrzymuje wysoką frekwencję (Beta/Gamma) max przez 90 min, po czym potrzebuje 20 min w stanie Alfa (NSDR / relaks) na wypłukanie adenozyny.
    </p></div>""", unsafe_allow_html=True)
    
    ultradian_min = st.slider("Czas trwania obecnego bloku pracy (min)", 0, 180, 45, key="ultradian_min")
    adenosine = (ultradian_min / 90) * 100 if ultradian_min <= 90 else 100 + ((ultradian_min - 90) * 1.5)
    perf = 100 - (max(0, ultradian_min - 90) * 1.5)
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        st.metric("Skupienie / Wydajność", f"{max(0, min(100, perf)):.0f}%", delta="-⚠️ Wypalenie" if ultradian_min > 90 else None, delta_color="inverse")
    with col_u2:
        st.metric("Dług Adenozynowy", f"{min(200, adenosine):.0f}%", delta="⚡ Za wysoki!" if adenosine > 100 else None, delta_color="inverse")


# ═══════════════════════════════════════════════════════════
# SEKCJA 9 — ARCHITEKTURA PRZEPŁYWU
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🌊 Sekcja 9 — Architektura Przepływu (Flow State)")

col_f1, col_f2 = st.columns([2, 3])

with col_f1:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>Trójkąt Przepływu (Csíkszentmihályi)</div>
    <p style='{NOTE}'>
    Kanał Flow znajduje się dokładnie tam, gdzie wyzwanie minimalnie przekracza Twoje obecne umiejętności (~4%).<br><br>
    1️⃣ Ustaw swoje <b>Umiejętności</b> w danym zadaniu.<br>
    2️⃣ Ustaw jego obiektywną <b>Trudność</b>.<br>
    3️⃣ Zobacz swój stan kognitywny na mapie.
    </p></div>""", unsafe_allow_html=True)
    
    skills_level = st.slider("Poziom Umiejętności (Wiedza/Praktyka)", 0, 100, 60, key="flow_skills")
    challenge_level = st.slider("Poziom Wyzwania (Trudność/Stawka)", 0, 100, 75, key="flow_challenge")
    
    if challenge_level > skills_level + 20:
        state_name, state_col = "Niepokój / Lęk (Anxiety) 🔴", "#ff1744"
        hack = "Hack: Zmniejsz trudność (podziel na małe kroki) lub dobierz wsparcie/wiedzę."
    elif skills_level > challenge_level + 20:
        state_name, state_col = "Nuda (Boredom) 🥱 / Relaks", "#3498db"
        hack = "Hack: Skróć czas na wykonanie o połowę. Stwórz sztuczną presję lub zwiększ standardy."
    elif challenge_level < 30 and skills_level < 30:
        state_name, state_col = "Apatia ⚪", "#888888"
        hack = "Zmień całkowicie cel zadania. Brakuje w nim i Twojej ambicji, i kompetencji."
    else:
        state_name, state_col = "⚡ STAN FLOW (Transient Hypofrontality)", "#00e676"
        hack = "Idealna ścieżka! Kora przedczołowa wyciszona. Działasz płynnie i instynktownie."

    st.markdown(f"""<div style='background:rgba(0,0,0,0.4);border:1px solid {state_col};border-radius:8px;padding:12px;margin-top:10px'>
    <b style='color:{state_col}'>{state_name}</b><br><span style='font-size:12px;color:#aaa'>{hack}</span>
    </div>""", unsafe_allow_html=True)

with col_f2:
    # Generowanie heatmapy stanów Flow
    sc, ch = np.meshgrid(np.linspace(0, 100, 100), np.linspace(0, 100, 100))
    Z = np.zeros_like(sc)
    for i in range(100):
        for j in range(100):
            s_val, c_val = sc[i, j], ch[i, j]
            if c_val > s_val + 20: Z[i, j] = 1 # Anxiety
            elif s_val > c_val + 20: Z[i, j] = 2 # Boredom
            elif s_val < 30 and c_val < 30: Z[i, j] = 3 # Apathy
            else: Z[i, j] = 4 # Flow
            
    colorscale_flow = [[0, "#ff1744"], [0.33, "#3498db"], [0.66, "#2a2a3a"], [1, "#00e676"]]
    
    fig_flow = go.Figure(go.Heatmap(z=Z, x=sc[0,:], y=ch[:,0], colorscale=colorscale_flow, showscale=False, opacity=0.4))
    fig_flow.add_trace(go.Scatter(x=[skills_level], y=[challenge_level], mode="markers", marker=dict(size=18, color="#ffffff", line=dict(color=state_col, width=3)), name="Twój Stan"))
    
    fig_flow.update_layout(
        title="Mapa Kognitywna", height=350,
        xaxis=dict(title="Umiejętności", range=[0, 100]), yaxis=dict(title="Wyzwanie", range=[0, 100]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=40, r=20, t=40, b=40)
    )
    fig_flow.add_annotation(x=80, y=80, text="FLOW ZONE", showarrow=False, font=dict(color="#00e676", size=16, weight="bold"))
    fig_flow.add_annotation(x=20, y=80, text="ANXIETY", showarrow=False, font=dict(color="#ff1744", size=14))
    fig_flow.add_annotation(x=80, y=20, text="BOREDOM", showarrow=False, font=dict(color="#3498db", size=14))
    st.plotly_chart(fig_flow, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# SEKCJA 10 — HORMEZA I ALOSTATEZA
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🛡️ Sekcja 10 — Hormeza i Antykruchość Biologiczna (M. Mattson)")

col_h1, col_h2 = st.columns(2)

with col_h1:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>🔥 Kalkulator Stresorów (Allostatic Load)</div>
    <p style='{NOTE}'>
    Krótki, ostry stres (Eustres) buduje twój organizm poprzez procesy nadkompensacji (np. autofagia).<br>
    Lekki, ciągły stres w tle (Distres) wyczerpuje twój układ nerwowy i opornościowy.
    </p></div>""", unsafe_allow_html=True)
    
    acute_stress_val = st.slider("Ostre Stresory (Zimno, Spriny, Sauna, Post/Głodówka)", 0, 100, 70, key="acute_stress")
    chronic_stress_val = st.slider("Przewlekłe Stresory (Maile, Przerywany Sen, Social Media, Inflamacja)", 0, 100, 30, key="chronic_stress")
    
    # Model Hormezy Inverted-U
    x_dose = np.linspace(0, 100, 100)
    health_response = 2.5 * x_dose - 0.025 * x_dose**2
    
    current_dose = acute_stress_val - (chronic_stress_val * 1.5)
    user_hx = max(0, min(100, 50 + current_dose/2))
    user_hy = 2.5 * user_hx - 0.025 * user_hx**2
    m_color_h = "#00e676" if user_hy > 0 else "#ff1744"
    
    fig_hormesis = go.Figure()
    fig_hormesis.add_trace(go.Scatter(x=x_dose, y=health_response, mode='lines', line=dict(color='#a855f7', width=3), name='Reakcja Adaptacyjna'))
    fig_hormesis.add_hline(y=0, line_dash="dash", line_color="#555")
    fig_hormesis.add_trace(go.Scatter(x=[user_hx], y=[user_hy], mode='markers', marker=dict(size=14, color=m_color_h), name='Twój Stan'))
    
    fig_hormesis.update_layout(
        title="Krzywa Hormetyczna (Wika Marka Mattsona)", height=300,
        xaxis=dict(title="Całkowity Poziom Stresu (Dose)", showticklabels=False), yaxis=dict(title="Wzrost Formy vs Degeneracja", showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False
    )
    st.plotly_chart(fig_hormesis, use_container_width=True)

with col_h2:
    st.markdown("### 💉 Protokół Antykruchości Fizjologicznej")
    st.markdown(f"""<div style='background:#111;border-left:4px solid #00e676;padding:12px;margin-bottom:12px'>
    <b>1. Ekspozycja na Skrajności (Barbell):</b><br>
    Spędzaj czas albo w totalnym głębokim relaksie (System Przywspółczulny, 0% stresu), albo w ekstremalnym, ultra-krótkim wysiłku (System Współczulny, 100% stresu). 
    <span style='color:#ff1744'>Środek sztangi niszczy serce i mózg.</span>
    </div>
    <div style='background:#111;border-left:4px solid #3498db;padding:12px;margin-bottom:12px'>
    <b>2. Okna Autofagii:</b><br>
    Post (>14-16h) uruchamia głęboki komórkowy recykling niszczący stare białka i naprawiający DNA.
    </div>
    <div style='background:#111;border-left:4px solid #a855f7;padding:12px'>
    <b>3. Szok Termiczny:</b><br>
    Zimno generuje CSP (Cold Shock Proteins), a gorąco (Sauna) HSP (Heat Shock Proteins), uodparniając białka i układ nerwowy na starzenie.
    </div>
    """, unsafe_allow_html=True)
    if user_hy < 0:
        st.error("⚠️ ALERT ALOSTATYCZNY: System jest przeciążony w tle! Drastycznie obetnij _chronic_ stressors (sen, bodźce). Dodawanie teraz treningów czy zimna tylko pogorszy sytuację.")
    elif user_hy < 40:
        st.warning("⚠️ Twój organizm znajduje się w strefie lenistwa. Wymaga mocniejszych bodźców i ekstremów by stymulować wzrost.")
    else:
        st.success("✅ OPTIMUM HORMETYCZNE. Twój układ nerwowy znajduje się na szczycie krzywej adaptacyjnej nadkompensacji.")


# ═══════════════════════════════════════════════════════════
# SEKCJA 11 — FOGG BEHAVIOR MODEL
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## ⚙️ Sekcja 11 — Fizyka Nawyków (B.J. Fogg Behavior Model)")

col_b1, col_b2 = st.columns([1, 1])

with col_b1:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>B = MAP (Behavior = Motivation × Ability × Prompt)</div>
    <p style='{NOTE}'>
    Siła woli jest zasobem, który wyczerpuje się w ciągu dnia (Ego Depletion).<br>
    Nigdy nie operuj bazując tylko na sile woli czy "motywacji". Maksymalizuj Łatwość (Ability) na wyłapanym Sygnale (Prompt).
    </p></div>""", unsafe_allow_html=True)
    
    b_mot = st.slider("Motywacja do Nawyku / Opór Wewnętrzny (1=Znikoma, 10=Żrąca Skuteczność)", 1, 10, 4, key="fbm_mot")
    b_abi = st.slider("Łatwość Akcji (1=Wymaga logistyki i wysiłku, 10=Dwu-minutowa sprawa w zasięgu ręki)", 1, 10, 3, key="fbm_abi")
    
    action_threshold = 30
    cur_score = b_mot * b_abi
    
    if cur_score >= action_threshold:
        st.success("✅ Akcja (Behavior) ma miejsce. Przebito krzywą akceptacji.")
    else:
        st.error("❌ Nawyk nie zaskoczy. NIE podkręcaj bezsensownie Motywacji. Zwiększ jego Łatwość (Ability), aż stanie się absurdalnie wręcz trywialny do wykonania.")

with col_b2:
    x_ab = np.linspace(1.0, 10.0, 100)
    y_mo = action_threshold / x_ab
    
    fig_fbm = go.Figure()
    fig_fbm.add_trace(go.Scatter(x=x_ab, y=y_mo, mode='lines', line=dict(color='#ff1744', width=2), name='Action Line'))
    fig_fbm.add_trace(go.Scatter(x=[b_abi], y=[b_mot], mode='markers', marker=dict(size=16, color="#00e676" if cur_score>=action_threshold else "#f39c12"), name="Twój Nawyk"))
    
    fig_fbm.update_layout(
        title="Fazy Adaptacji Zmiany (Fogg's Action Line)", height=300,
        xaxis=dict(title="Łatwość Wykonania (Ability) ➜", range=[0, 10]), 
        yaxis=dict(title="Motywacja ➜", range=[0, 10]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=40, r=20, t=40, b=40)
    )
    fig_fbm.add_annotation(x=8, y=8, text="✅ DZIAŁAJĄCA RUTYNA", showarrow=False, font=dict(color="#00e676"))
    fig_fbm.add_annotation(x=2, y=2, text="❌ MARTWA STREFA", showarrow=False, font=dict(color="#ff1744"))
    st.plotly_chart(fig_fbm, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# SEKCJA 12 — DYSKONTO HIPERBOLICZNE
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📉 Sekcja 12 — Dyskonto Hiperboliczne i Default Nudges")

col_n1, col_n2 = st.columns([3, 2])

with col_n1:
    days_array = np.arange(0, 365, 5)
    reward_value = 1000.0
    
    # Modele dyskontowania
    k_exp = 0.005 # Racjonalne (wykładnicze)
    k_hyp = 0.04  # Behawioralne (hiperboliczne - bardzo strome cięcie z bliska)
    
    val_exp = reward_value * np.exp(-k_exp * days_array)
    val_hyp = reward_value / (1 + k_hyp * days_array)
    
    fig_disc = go.Figure()
    fig_disc.add_trace(go.Scatter(x=days_array, y=val_exp, mode="lines", name="Racjonalny Homo Economicus", line=dict(color="#3498db")))
    fig_disc.add_trace(go.Scatter(x=days_array, y=val_hyp, mode="lines", name="Twój gadzi mózg (Hiperbola)", line=dict(color="#ff1744", width=3)))
    
    fig_disc.update_layout(
        title="Twój Mózg: Deprecjacja Wartości w Czasie", height=280,
        xaxis=dict(title="Dni oczekiwania na nagrodę (np. zdrowie, gotówka)"), yaxis=dict(title="Odczuwana 'Wartość' Nagrody"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white", size=11), margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(x=0.4, y=0.9, font=dict(size=10))
    )
    st.plotly_chart(fig_disc, use_container_width=True)

with col_n2:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>Odysseus Pact (Kontrakt Odyseusza)</div>
    <p style='{NOTE}'>
    Dyskonto Hiperboliczne oznacza, że 10 PLN dzisiaj zyskuje u Ciebie w mózgu przewagę nad 100 PLN za tydzień. Dlatego wybierasz fast food zamiast kaloryferu na brzuchu z przyszłości.<br><br>
    <b style='color:#ffea00'>Architektura Wyboru (R. Thaler):</b>
    Aby pokonać dyskonto, projektuj środowisko tak, by "dobra rzecz" była wartością domyślną (default nudge), a jej pominięcie wymagało absurdalnego wysiłku.
    </p></div>""", unsafe_allow_html=True)
    
    st.markdown("<p style='font-size:14px;color:#00e676;font-weight:bold'>Twój Kontrakt Odyseusza (Test tarcia zmysłów):</p>", unsafe_allow_html=True)
    st.checkbox("Czy Twój telefon jest w innym fizycznym pokoju, kiedy pracujesz (deep work)?", key="chk_od1")
    st.checkbox("Czy przelewasz nadwyżki finansowe w pełni automatycznie w dniu wypłaty?", key="chk_od2")
    st.checkbox("Czy usunąłeś wszystkie powiadomienia, ikony z pulpitu i bodźce wzrokowe?", key="chk_od3")

# ═══════════════════════════════════════════════════════════
# SEKCJA 13 — EWOLUCJA WSPÓŁPRACY (ZASZUMIONY DYLEMAT WIĘŹNIA)
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🤝 Sekcja 13 — Ewolucja Współpracy w Zaszumionym Środowisku (IPD)")

col_ipd1, col_ipd2 = st.columns([3, 2])

with col_ipd1:
    noise_level = st.slider("Poziom szumu środowiskowego (Błędy interpretacji %)", 0, 30, 5, 1, key="ipd_noise") / 100.0
    
    # Symulacja wpływu szumu na wypłaty dominujących strategii
    tft_score = max(0, 100 - noise_level * 300)
    wsls_score = max(0, 100 - noise_level * 100)
    gtft_score = max(0, 100 - noise_level * 150)
    alld_score = min(100, 20 + noise_level * 200)
    
    strategies_ipd = ["Tit-for-Tat (TFT)", "Generous TFT", "Win-Stay Lose-Shift (WSLS)", "Always Defect (ALLD)"]
    scores_ipd = [tft_score, gtft_score, wsls_score, alld_score]
    colors_ipd = ["#3498db", "#a855f7", "#00e676", "#ff1744"]
    
    fig_ipd = go.Figure()
    fig_ipd.add_trace(go.Bar(
        x=strategies_ipd, y=scores_ipd, marker_color=colors_ipd,
        text=[f"{v:.1f}" for v in scores_ipd], textposition="outside",
        textfont=dict(color="white", size=12)
    ))
    fig_ipd.update_layout(
        title=f"Odporność Strategii przy {noise_level*100:.0f}% Szumu",
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        yaxis=dict(range=[0, 115], gridcolor="#1c1c2e", title="Wypłata Asymptotyczna"),
        xaxis=dict(gridcolor="#1c1c2e"),
        margin=dict(l=20,r=20,t=50,b=40)
    )
    st.plotly_chart(fig_ipd, use_container_width=True)

with col_ipd2:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>🔄 Spirale Zdrady i Stabilność (Axelrod, Nowak)</div>
    <p style='{NOTE}'>
    W klasycznej Teorii Gier (bez szumu) <b style='color:#3498db'>Tit-for-Tat</b> dominuje.
    Ale rzeczywistość jest zaszumiona. Błędnie interpretujemy intencje i losowe zdarzenia jako ataki.<br><br>
    W szumie TFT wpada w nieskończoną spiralę wzajemnych retorsji (zdrada za rzekomą zdradę).<br><br>
    <b style='color:#00e676'>Win-Stay, Lose-Shift (Pawłow)</b> jest asymptotycznie stabilne: powtarza ruch, jeśli wypłata była wysoka, zmienia jeśli niska. Potrafi automatycznie wybaczyć po obustronnym błędzie i powrócić do kooperacji, jednocześnie bezlitośnie eksploatując naiwnych.
    </p></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SEKCJA 14 — MECHANISM DESIGN
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## ⚙️ Sekcja 14 — Projektowanie Mechanizmów (Incentive-Compatible Nudges)")

col_md1, col_md2 = st.columns([1, 1])

with col_md1:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>Revelation Principle & Asymetria Informacji</div>
    <p style='{NOTE}'>
    Problem: Twoje planujące "Ja" (Pryncypał) narzuca jeden cel. Twoje wykonawcze "Ja" (Agent) w danym dniu ma ukryte, zmienne parametry energii (niska/wysoka) i często sabotuje sztywny system (dezercja).<br><br>
    Rozwiązanie z <b>Mechanism Design</b>: Stwórz menu opcji (Screening Contracts), które sprawi, że dominującą strategią agenta będzie ujawnienie prawdy o swoim stanie i asymilacja do dopasowanego celu, chroniąc system przed "Wszystko albo Nic".
    </p></div>""", unsafe_allow_html=True)
    
    vitality = st.selectbox("Zgłoś swój dzisiejszy obiektywny stan witalności (Typ Agenta):", 
        ["Niski (Wyczerpanie, Stres)", "Średni (Standard)", "Wysoki (Flow, Energia)"], index=1, key="md_vitality")

with col_md2:
    if "Niski" in vitality:
        contract = "Zobowiązanie Minimum (Tylko Nawyk Zębowy / Utrzymanie Linii)"
        payout = "Ochrona przed pęknięciem systemu, spójność tożsamości (0 do 1)"
        color_md = "#ff1744"
    elif "Średni" in vitality:
        contract = "Zobowiązanie Standardowe (2-3h Deep Work)"
        payout = "Liniowy, stabilny progres (10% postępu dziennie)"
        color_md = "#3498db"
    else:
        contract = "Zobowiązanie Ekstremalne (Moonshot, Ryzyko, Outbound Do Dużych Graczy)"
        payout = "Opcja Call, Szansa na Nieliniowy Zwrot (Czarne Łabędzie)"
        color_md = "#00e676"
        
    st.markdown(f"""<div style='background:rgba(0,0,0,0.4);border:2px solid {color_md};border-radius:12px;padding:20px'>
    <h4 style='color:{color_md};margin-bottom:10px'>Przyjęty Kontrakt (Typ Zgodny z Bodźcami)</h4>
    <b>Zadanie:</b> {contract}<br>
    <b>Oczekiwany Zwrot:</b> {payout}<br><br>
    <span style='color:#aaa;font-size:12px'>Brak kary lub poczucia winy za wybór opcji Niskiej. Opcja Wysoka premiowana szansą na przewrót statusu quo. System zabezpiecza Condition of Individual Rationality.</span>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SEKCJA 15 — EXPLORATION VS EXPLOITATION
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🎲 Sekcja 15 — Eksploracja vs Eksploatacja & Reguła 37% (Multi-Armed Bandit)")

col_ee1, col_ee2 = st.columns([3, 2])

with col_ee1:
    total_time = st.slider("Horyzont Czasowy (np. całkowita pula dni na znalezienie celu)", 10, 100, 30, key="ee_time")
    
    explore_phase = int(total_time * 0.37)
    x_time = np.arange(1, total_time + 1)
    
    fig_ee = go.Figure()
    
    # Zaznaczanie obszarów eksploracji i eksploatacji
    fig_ee.add_vrect(x0=1, x1=explore_phase, fillcolor="rgba(52,152,219,0.2)", line_width=0, 
                     annotation_text="Faza Obserwacji (Gather Data)", annotation_position="top left", annotation_font_color="#3498db")
    fig_ee.add_vrect(x0=explore_phase, x1=total_time, fillcolor="rgba(0,230,118,0.2)", line_width=0, 
                     annotation_text="Faza Decyzyjna (Eksploatacja)", annotation_position="top right", annotation_font_color="#00e676")
    
    fig_ee.add_vline(x=explore_phase, line_dash="solid", line_color="#ffea00")
    
    fig_ee.update_layout(
        title=f"Optimal Stopping (Reguła 37%) — Przełączenie trybu w {explore_phase}. kroku",
        height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        xaxis=dict(title="Dostępny Czas (T)", gridcolor="#1c1c2e"),
        yaxis=dict(showticklabels=False, gridcolor="#1c1c2e", range=[0, 1]),
        margin=dict(l=20,r=20,t=50,b=40),
        showlegend=False
    )
    st.plotly_chart(fig_ee, use_container_width=True)

with col_ee2:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>Złoty Podział Zatrzymania Czasu</div>
    <p style='{NOTE}'>
    Matematyka problemu optymalnego zatrzymania wskazuje jasno: zbierz dane przez pierwsze <b>37% czasu</b> (Zasada Sekreterki). Odrzuć wszystkie pierwsze zapytania, precyzyjnie kalibrując własne oczekiwania i rozpoznając rozkład rynku.<br><br>
    Następnie zmień tryb na eksploatację: zaatakuj i przyjmij <b>pierwszą opcję</b>, która jest lepsza od najwyższej, jaką widziałeś podczas fazy kalibracyjnej.
    </p></div>""", unsafe_allow_html=True)
    st.markdown(f"""<div style='background:#111;border-left:4px solid #f39c12;padding:12px'>
    <b>Upper Confidence Bound (UCB) w Life OS:</b><br>
    Na początku nowej ścieżki (duży zapas czasu) maksymalizuj wejście z optymizmem w nieznane. Gdy czas paruje – zamykaj się z opcjami sprawdzonymi historycznie.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SEKCJA 16 — TEORIA SYGNAŁÓW W ERZE AI 
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📡 Sekcja 16 — Teoria Sygnałów & Zarządzanie Wiarygodnością (Post-AGI)")

col_sig1, col_sig2 = st.columns([3, 2])

with col_sig1:
    ai_penetration = st.slider("Poziom Adopcji Generatywnego AI w Pracy i Komunikacji", 0.0, 1.0, 0.8, 0.05, key="sig_ai")
    
    signals = ["Cyfrowy Raport / Opracowanie", "Cold Email / Pitch", "Odręczne Pismo / List", "Spotkanie Twarzą w Twarz", "Skin in the Game (Zakład własnymi pieniędzmi)"]
    cost = [10, 5, 50, 80, 100]
    
    # AI degraduje wiarygodność tekstu. Fizyka zostaje nienaruszona
    cred_text = max(5, 80 - ai_penetration * 75)
    cred_email = max(5, 60 - ai_penetration * 55)
    
    credibility = [cred_text, cred_email, 65 + ai_penetration*10, 90, 100]
    colors_sig = ["#ff1744", "#ff1744", "#f39c12", "#00e676", "#a855f7"]
    
    fig_sig = go.Figure()
    fig_sig.add_trace(go.Scatter(
        x=cost, y=credibility, mode="markers+text",
        text=signals, textposition="top center",
        marker=dict(size=[15, 12, 20, 30, 40], color=colors_sig, opacity=0.8),
        textfont=dict(color="white", size=10)
    ))
    
    fig_sig.update_layout(
        title="Portfel Sygnałów (Costly Signaling Theory)",
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"),
        xaxis=dict(title="Wysiłek, Koszt i Fizyczność (Skin in the Game)", gridcolor="#1c1c2e", range=[0, 115]),
        yaxis=dict(title="Postrzegana Wiarygodność u Odbiorcy", gridcolor="#1c1c2e", range=[0, 115]),
        margin=dict(l=40,r=40,t=40,b=40), showlegend=False
    )
    st.plotly_chart(fig_sig, use_container_width=True)

with col_sig2:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>Zapadnięcie Się Tanich Sygnałów</div>
    <p style='{NOTE}'>
    W ekonomii ewolucyjnej sygnalizacja jest wiarygodna tylko, gdy oszustowi zbytnio nie opłaca się jej podrobić (pawi ogon).<br><br>
    Sztuczna Inteligencja sprowadza koszt wygenerowania pięknego raportu, skryptu lub kurtuazyjnego emaila do zera. Cykl domyka się – stają się one dla odbiorcy "Cheap Talk", potęgując jedynie tzw. "karę za AI".<br><br>
    <b style='color:#00e676'>Nowa Waluta (Proof of Work):</b> W erze wyzerowanych kosztów kognitywnych, wiarygodność przenosi się w domenę atomów i weryfikowalnego ryzyka. Trzonem Twojego portfela w `Life OS` musi być fizyczna obecność, autentyczność wystawiająca na zranienie i uwiązany kapitał.
    </p></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SEKCJA 17 — DUALIZM JAŹNI I SAMOKONTROLA
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🧠 Sekcja 17 — Wewnątrzpersonalna Teoria Gier (Planista vs Wykonawca)")

col_ds1, col_ds2 = st.columns(2)

with col_ds1:
    penalty = st.slider("Wielkość Nałożonej Kary za Odstępstwo (Commitment Device w PLN)", 0, 5000, 1000, 100, key="ds_pen")
    
    opt_a_impulse = 80 # Szybka jednorazowa dopamina z nagrody dziś
    opt_b_impulse = 20
    opt_b_future = 90  # Długofalowa dywidenda z pracy
    
    # Narzucona sztuczna dewalucja opcji poprzez karę finansową/reputacyjną
    penalty_util = penalty / 50.0  
    opt_a_mod = max(0, opt_a_impulse - penalty_util)
    
    fig_ds = go.Figure()
    fig_ds.add_trace(go.Bar(name="Naturalny Odruch (Gadzi)", x=["Rozrywka/Sofa (Zysk Natychmiast)", "Dyscyplina (Inwestycja)"], y=[opt_a_impulse, opt_b_impulse], marker_color="#ff1744"))
    fig_ds.add_trace(go.Bar(name="Z Kontraktem Zobowiązującym", x=["Rozrywka/Sofa (Zysk Natychmiast)", "Dyscyplina (Inwestycja)"], y=[opt_a_mod, opt_b_impulse + opt_b_future], marker_color="#00e676"))
    
    fig_ds.update_layout(
        title="Manipulacja Oczekiwaną Użytecznością Systemu 1",
        barmode='group', height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), margin=dict(l=20,r=20,t=40,b=40),
        xaxis=dict(gridcolor="#1c1c2e"), yaxis=dict(title="Psychologiczna Użyteczność dla Wykonawcy", gridcolor="#1c1c2e"),
        legend=dict(x=0.0, y=1.2, orientation="h", font=dict(size=10))    
    )
    st.plotly_chart(fig_ds, use_container_width=True)

with col_ds2:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>Dual-Self Model (Planista i Wykonawca)</div>
    <p style='{NOTE}'>
    Nie stanowisz jednolitego ja. Jesteś zespołem rotujących w czasie agentów.<br><br>
    <b>Planista (System 2: Kora przedczołowa)</b> dąży do potęgowania skumulowanych zysków w życiu przez dekady.<br>
    <b>Wykonawca (System 1: Ciało migdałowate)</b> preferuje maksymalizację dopaminy o godzinie 19:00.<br><br>
    <b style='color:#00e676'>Zarządzanie poprzez Zobowiązanie:</b>
    Planista musi "związać sobie ręce". Ustawiając bolesną opłatę za rezygnację (np. oddanie na partię polityczną której nie lubisz {penalty} PLN), zmieniasz Równowagę Nasha wewnątrz umysłu – Wykonawca w strachu ulega racjonalizując drogę Dyscypliny za korzystniejszą w danym momencie.
    </p></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SEKCJA 18 — BACKWARD INDUCTION
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## ♟️ Sekcja 18 — Backward Induction w Trajektorii Życiowej")

col_bi1, col_bi2 = st.columns([3, 2])

with col_bi1:
    st.markdown("<span style='color:#aaa;font-size:12px'>Metoda symulowania optymalnej przestrzeni kroków posiłkując się Subgame-perfect Nash equilibrium.</span>", unsafe_allow_html=True)
    
    node_x = [0, 1, 1, 2, 2, 2, 2]
    node_y = [0, 1, -1, 1.5, 0.5, -0.5, -1.5]
    node_text = ["Dziś (t=0)", "Projekt A (Skalowalny)", "Model B (Etat)", "Cel: Niezależność (Z=100)", "Rozwój Boczny (Z=40)", "Awans (Z=30)", "Zwolnienie (Z=-10)"]
    colors_bi = ["#a855f7", "#00e676", "#ff1744", "#00e676", "#555", "#555", "#ff1744"]
    
    edge_x = [0, 1, None, 0, 1, None, 1, 2, None, 1, 2, None, 1, 2, None, 1, 2, None]
    edge_y = [0, 1, None, 0, -1, None, 1, 1.5, None, 1, 0.5, None, -1, -0.5, None, -1, -1.5, None]

    fig_bi = go.Figure()
    fig_bi.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#333", width=2), hoverinfo="skip"))
    fig_bi.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", 
        text=node_text, textposition="bottom center",
        marker=dict(size=22, color=colors_bi, line=dict(color="white", width=2)),
        textfont=dict(color="white", size=11)
    ))
    
    fig_bi.update_layout(
        title="Drzewo Trajektorii (Krytyczna Ścieżka Indukcji Wstecznej)",
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, 2]),
        margin=dict(l=20,r=20,t=40,b=20)
    )
    fig_bi.add_trace(go.Scatter(x=[2, 1, 0], y=[1.5, 1, 0], mode="lines", line=dict(color="#00e676", width=4, dash="dot"), name="Optymalna Trasa Z"))
    st.plotly_chart(fig_bi, use_container_width=True)

with col_bi2:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>Złudzenie Pierwszego Kroku</div>
    <p style='{NOTE}'>
    Myślenie uwięzione w czasie t=1 szuka najwyższej natychmiastowej wypłaty (np. bezpieczny fotel z dobrą wypłatą natychmiastową na Modelu B). Ale Model B na poziomie węzłów t=2 prowadzi do całkowitego spłycenia opcji przyszłości (ślepy zaułek - pułapka lokalnego maksimum).<br><br>
    <b style='color:#00e676'>Mechanizm Myślenia Wstecznego (Backward Induction):</b> Rozwiązuje grę od jej zakończenia (t=T). Definiujesz stan T (Z=100), sprawdzasz warunek wymagany na T-1 doprowadzający do T z najwyższym prawdopodobieństwem i iterujesz cofasz aż do dzisiaj.<br><br>
    Tylko myślenie wsteczne ukazuje <b>prawdziwy koszt alternatywny</b> obecnej wygody.
    </p></div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SEKCJA 19 — ANTYKRUCHOŚĆ I NIERÓWNOŚĆ JENSENA 
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🗻 Sekcja 19 — Wypukłość Wypłat (Antykruchość Czasu/Projektów)")

col_ak1, col_ak2 = st.columns([2, 3])

with col_ak1:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>Zysk Z Niepewności (Wypukłość Jensena)</div>
    <p style='{NOTE}'>
    Antykruchość to system, który kwitnie w warunkach stresu.<br>
    Twój Life OS musi asymilować losowość, odcinając z góry ryzyko finansowego czy projektowego zera absolutnego (Via Negativa), i otwierając system na nieograniczone wypłaty dodatnie z małych, odważnych zakładów z Prawego Ogona (Opcje Realne).
    </p></div>""", unsafe_allow_html=True)
    
    sys_type = st.radio("Zdefiniuj ekspozycję swojej formacji zawodowej na zmienność rynku:", 
        ["Kruchy (Brak planu B, uzależnienie od 1 klienta)", "Wytrzymały (Standardowa redundancja i etaty)", "Antykruchy (Barbell, Dźwignie IP/Media, Opcje)"], index=2, key="ak_type")

with col_ak2:
    v = np.linspace(-5, 5, 100)
    if "Kruchy" in sys_type:
        y_val = -0.5 * v**2 + 5
        color_ak = "#ff1744"
        fill_col = "rgba(255, 23, 68, 0.2)"
        ak_title = "Kruchość (Wklęsłość) – Pętla na szyi, pęka przy wstrząsie."
    elif "Wytrzymały" in sys_type:
        y_val = np.ones_like(v) * 2
        color_ak = "#3498db"
        fill_col = "rgba(52, 152, 219, 0.2)"
        ak_title = "Wytrzymałość (Linear) – Zmieność ignorowana, ale asymetria ujemna."
    else:
        y_val = 0.5 * np.maximum(0, v)**2 - 0.5
        color_ak = "#00e676"
        fill_col = "rgba(0, 230, 118, 0.2)"
        ak_title = "Antykruchość (Wypukłość) – Bezpośrednia monetyzacja rynkowego Chaosu."
        
    fig_ak = go.Figure()
    fig_ak.add_trace(go.Scatter(x=v, y=y_val, mode="lines", line=dict(color=color_ak, width=4), fill="tozeroy", fillcolor=fill_col)) 
    
    fig_ak.update_layout(
        title=ak_title,
        height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), showlegend=False,
        xaxis=dict(title="Zmienność Rynku (Stres, Zwolnienia, Pandemia)", gridcolor="#1c1c2e"),
        yaxis=dict(title="Zwrot/Rozwój Osobisty (Wypłata)", gridcolor="#1c1c2e", range=[-8, 15]),
        margin=dict(l=40,r=20,t=40,b=40)
    )
    fig_ak.add_vline(x=0, line_dash="dash", line_color="#555")
    st.plotly_chart(fig_ak, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# SEKCJA 20 — ECONOMIA UWAGI I PRZECIĄŻENIE ZARZĄDCZE
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 👁️ Sekcja 20 — Ekonomia Uwagi jako Krytyczna Infrastruktura")

col_attn1, col_attn2 = st.columns([1, 1])

with col_attn1:
    governance_load = st.slider("Szacunkowa objętość decyzji i wątków mikrozarządzania dziennie (Load)", 10, 200, 110, key="at_load")
    
    capacity = 100
    x_at = np.arange(10, 200, 5)
    y_at = [max(0, 100 - (max(0, val - capacity) ** 1.5) / 2) for val in x_at]
    
    eff_current = max(0, 100 - (max(0, governance_load - capacity) ** 1.5) / 2)
    col_ef = "#00e676" if eff_current > 80 else ("#f39c12" if eff_current > 40 else "#ff1744")
    
    fig_at = go.Figure()
    fig_at.add_trace(go.Scatter(x=x_at, y=y_at, mode="lines", fill="tozeroy", line=dict(color="#3498db", width=3), fillcolor="rgba(52, 152, 219, 0.1)"))
    fig_at.add_trace(go.Scatter(x=[governance_load], y=[eff_current], mode="markers", marker=dict(size=18, color=col_ef, line=dict(color="white", width=2))))
    
    fig_at.update_layout(
        title="Przeciążenie Zarządcze (Governance Overload)",
        height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", family="Inter"), showlegend=False,
        xaxis=dict(title="Ilość Równoległych Pętli Zadaniowych", gridcolor="#1c1c2e"),
        yaxis=dict(title="Dostępny Budżet Decyzyjny Siły Woli (%)", gridcolor="#1c1c2e"),
        margin=dict(l=40,r=20,t=40,b=40)
    )
    fig_at.add_vline(x=capacity, line_dash="solid", line_color="#ffea00", annotation_text="Kres Pojemności Systemu 2")
    st.plotly_chart(fig_at, use_container_width=True)

with col_attn2:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>Złudzenie Multitaskingu</div>
    <p style='{NOTE}'>
    Nieuwaga jest równie potężnym narzędziem w zarządzaniu co skupienie. Gdy Twój Load pożera Capacity (<span style='color:#ffea00'>wartość > {capacity}</span>), doznajesz obciążenia poznawczego, czyli "zmęczenia decyzyjnego". Mózg, nie mogąc poprawnie kalkulować strategii Life OS, bezpowrotnie opada do poziomu najsłabszego Systemu 1 (impulsów, apatii).<br><br>
    ✅ <b>Racjonalna Nieuwaga:</b> Ucz się ignorować e-maile i szumy o asymetrycznie zerowej wartości. Projektuj powierzchnię kontrolną tak, by nie przekraczała naturalnej elastyczności mózgu wspieranego zewnętrznymi modułami automatyzacji rutyn.
    </p></div>""", unsafe_allow_html=True)
    
    if eff_current == 0:
        st.error("🔴 CAŁKOWITY PARALIŻ. Jesteś w stanie dekompensacji poznawczej. Automatyczny powrót mózgu na tanie znieczulacze zagwarantowany. Eliminuj obowiązki natychmiast.")
    elif eff_current < 70:
        st.warning("🟡 Pojawiają się koszty spadku zdolności samokontroli.")


st.markdown("---")
st.markdown(f"""<div style='text-align:center;color:#2a2a3a;font-size:11px;padding:12px'>
Life OS v3.0 · Advanced Quant Platform Ecosystem · Integrujący twardą powtarzalność praw nauk systemowych:<br>
Ergodicity (Peters), RPE (Schultz), Network Centrality (Burt), Barbell & Extremistan (Taleb), Kelly Criterion, <br>
Chronobiology (Panda), Flow State (Csíkszentmihályi), Action Line (Fogg), Nudges (Kahneman & Thaler), <br>
Mechanism Design, Evolutionary Game Theory (Axelrod/Nowak), Multi-Armed Bandit, Costly Signaling.
</div>""", unsafe_allow_html=True)


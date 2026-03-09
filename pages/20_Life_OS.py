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

st.markdown("---")
st.markdown(f"""<div style='text-align:center;color:#2a2a3a;font-size:11px;padding:12px'>
Life OS v1.0 · Barbell Strategy Quant · Oparty na: Ergodicity Economics (Peters), 
Reward Prediction Error (Schultz), Structural Holes (Burt), Barbell Strategy (Taleb), 
Kelly Criterion (Kelly 1956)
</div>""", unsafe_allow_html=True)

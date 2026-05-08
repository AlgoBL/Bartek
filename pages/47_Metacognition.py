"""47_Metacognition.py — Superforecasting, Brier Score, Kalibracja Predykcji"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.styling import apply_styling, module_header

st.markdown(apply_styling(), unsafe_allow_html=True)
st.markdown(module_header(
    title="Superforecasting — Metapoznanie",
    subtitle="Brier Score · Kalibracja · Dunning-Kruger · Tetlock's Superforecaster Rules",
    icon="🔭", badge="Epistemologia"
), unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"

tabs = st.tabs([
    "📊 Brier Score & Kalibracja", "🧠 Dunning-Kruger",
    "🎯 Superforecaster Test", "📐 Zasady Tetlocka"
])

# ── TAB 1: BRIER SCORE ────────────────────────────────────────────────────
with tabs[0]:
    st.markdown("### 📊 Brier Score — Miara Jakości Predykcji")
    st.caption("Brier (1950): BS = (1/N)·Σ(f_i - o_i)². 0=perfekcyjny, 1=najgorszy. Niższy = lepszy.")

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>📐 Interpretacja Brier Score</div>
    <b>BS = 0.0:</b> Perfekcyjne predykcje (niemożliwe w praktyce)<br>
    <b>BS = 0.25:</b> Poziom losowego forecasting (p=0.5 zawsze)<br>
    <b>BS ≤ 0.15:</b> Superforecaster (top 2% Tetlock GJP)<br>
    <b>BS ≤ 0.20:</b> Expert forecaster<br>
    <b>BS > 0.25:</b> Poniżej poziomu losowego (ostrożnie z negacją predykcji!)
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 🧪 Twoje Predykcje Finansowe (Wprowadź dane)")
    n_predictions = st.slider("Liczba predykcji do oceny", 3, 20, 8)

    pred_data = []
    col_h1, col_h2, col_h3 = st.columns([3, 2, 1])
    col_h1.markdown("**Predykcja (pytanie)**")
    col_h2.markdown("**Twoje p (0-100%)**")
    col_h3.markdown("**Wynik (1/0)**")

    default_predictions = [
        "S&P500 > 5000 na koniec Q2 2025",
        "Inflacja USA > 3% za 6 mies.",
        "Fed obniżka stóp w Q3 2025",
        "EUR/USD > 1.10 za 3 mies.",
        "Recesja USA w 2025",
        "Złoto > 3000 USD do końca roku",
        "BTC > 100k USD do końca 2025",
        "VIX > 30 w ciągu 6 mies.",
    ]

    for i in range(n_predictions):
        c1, c2, c3 = st.columns([3, 2, 1])
        q = c1.text_input(f"Predykcja {i+1}", default_predictions[i] if i < len(default_predictions) else f"Pytanie {i+1}", key=f"pred_q_{i}", label_visibility="collapsed")
        p = c2.slider("", 0, 100, 50, 1, key=f"pred_p_{i}", label_visibility="collapsed") / 100
        o = c3.checkbox("Tak", key=f"pred_o_{i}")
        pred_data.append({"question": q, "prob": p, "outcome": int(o)})

    if pred_data:
        df_pred = pd.DataFrame(pred_data)
        df_pred["brier_item"] = (df_pred["prob"] - df_pred["outcome"]) ** 2
        bs_total = df_pred["brier_item"].mean()
        bs_random = 0.25

        # Resolution (variance of forecasts)
        resolution = df_pred["prob"].var()
        # Reliability (calibration error)
        reliability = df_pred["brier_item"].mean() - (df_pred["prob"].mean() - df_pred["outcome"].mean())**2

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Brier Score", f"{bs_total:.4f}",
                       f"{(bs_total - bs_random):+.4f} vs random",
                       delta_color="inverse")
        col_m2.metric("Skill Score", f"{1 - bs_total/bs_random:.2%}",
                       help="Pozytywny = lepszy od losowego")
        col_m3.metric("Resolution", f"{resolution:.4f}",
                       help="Wyższy = bardziej zróżnicowane predykcje")
        col_m4.metric("N Predykcji", n_predictions)

        # Calibration plot
        bins = np.linspace(0, 1, 6)
        bin_labels = []
        obs_rates = []
        pred_means = []

        for b in range(len(bins) - 1):
            mask = (df_pred["prob"] >= bins[b]) & (df_pred["prob"] < bins[b + 1])
            if mask.sum() > 0:
                bin_labels.append(f"{bins[b]:.0%}–{bins[b+1]:.0%}")
                obs_rates.append(df_pred[mask]["outcome"].mean())
                pred_means.append(df_pred[mask]["prob"].mean())

        fig_cal = go.Figure()
        fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                      name="Perfect Calibration",
                                      line=dict(color="#aaa", dash="dash")))
        if obs_rates:
            fig_cal.add_trace(go.Scatter(x=pred_means, y=obs_rates, mode="markers+lines",
                                          name="Twoja Kalibracja",
                                          marker=dict(size=14, color="#00e676"),
                                          line=dict(color="#00e676", width=2.5)))
        fig_cal.update_layout(
            title="Krzywa Kalibracji (Reliability Diagram)",
            xaxis=dict(title="Podane Prawdopodobieństwo", gridcolor="#1c1c2e", range=[-0.05, 1.05]),
            yaxis=dict(title="Zaobserwowana Częstość", gridcolor="#1c1c2e", range=[-0.05, 1.05]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=380,
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_cal, use_container_width=True)

        level = ("🏆 SUPERFORECASTER (top 2%)" if bs_total <= 0.15 else
                 "⭐ Expert Forecaster" if bs_total <= 0.20 else
                 "📊 Przeciętny Forecaster" if bs_total <= 0.25 else
                 "⚠️ Poniżej poziomu losowego")
        st.markdown(f"""<div style='{CARD}'>
        <b>Poziom:</b> <span style='font-size:18px'>{level}</span><br>
        Brier Score = {bs_total:.4f} | Random baseline = {bs_random:.4f}
        </div>""", unsafe_allow_html=True)


# ── TAB 2: DUNNING-KRUGER ─────────────────────────────────────────────────
with tabs[1]:
    st.markdown("### 🧠 Dunning-Kruger — Quantitative Model")
    st.caption("Kruger & Dunning (1999): pewność siebie a kompetencja. Efekt Mt. Stupid i Doliny Rozpaczy.")

    col1, col2 = st.columns([1, 2])
    with col1:
        skill_level = st.slider("Twój poziom wiedzy/umiejętności (0-100)", 0, 100, 30)
        confidence_level = st.slider("Twój poziom pewności siebie (0-100)", 0, 100, 75)

    # Dunning-Kruger curve
    skill_x = np.linspace(0, 100, 500)
    # Characteristic DK shape: high confidence at low skill, dip in middle, rise with expertise
    dk_confidence = (
        85 * np.exp(-0.5 * ((skill_x - 15) / 12)**2) +    # Peak of Mt. Stupid
        20 * np.exp(-0.5 * ((skill_x - 40) / 8)**2) * (-1) +  # Valley of Despair (subtract)
        30 + 0.4 * skill_x +                                  # Expert rise
        10 * np.exp(-0.5 * ((skill_x - 35) / 10)**2) * (-1)   # Dip
    )
    dk_confidence = np.clip(dk_confidence, 10, 100)
    actual_skill = skill_x  # linear

    with col2:
        fig_dk = go.Figure()
        fig_dk.add_trace(go.Scatter(x=skill_x, y=dk_confidence, mode="lines",
                                     name="Pewność Siebie (DK curve)",
                                     line=dict(color="#ff1744", width=3)))
        fig_dk.add_trace(go.Scatter(x=skill_x, y=actual_skill, mode="lines",
                                     name="Rzeczywiste Umiejętności",
                                     line=dict(color="#00e676", width=2.5, dash="dash")))

        # Annotate phases
        fig_dk.add_annotation(x=15, y=95, text="Mt. Stupid<br>(Urojona<br>Kompetencja)", showarrow=False,
                               font=dict(color="#ffea00", size=10), align="center")
        fig_dk.add_annotation(x=38, y=30, text="Dolina<br>Rozpaczy", showarrow=False,
                               font=dict(color="#3498db", size=10))
        fig_dk.add_annotation(x=80, y=72, text="Plateau<br>Eksperta", showarrow=False,
                               font=dict(color="#00e676", size=10))

        # User position
        dk_val_at_skill = np.interp(skill_level, skill_x, dk_confidence)
        fig_dk.add_trace(go.Scatter(x=[skill_level], y=[confidence_level], mode="markers",
                                     name="Twoja Pozycja",
                                     marker=dict(size=18, color="#a855f7",
                                                 line=dict(color="white", width=2))))
        fig_dk.add_trace(go.Scatter(x=[skill_level], y=[dk_val_at_skill], mode="markers",
                                     name="DK Predykcja",
                                     marker=dict(size=14, color="#ffea00", symbol="diamond")))

        fig_dk.update_layout(
            title="Efekt Dunning-Kruger — Pewność vs Kompetencja",
            xaxis=dict(title="Poziom Umiejętności", gridcolor="#1c1c2e", range=[0, 100]),
            yaxis=dict(title="Pewność Siebie (%)", gridcolor="#1c1c2e", range=[0, 110]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="Inter"), height=420,
            legend=dict(x=0.5, y=0.99, xanchor="center"),
            margin=dict(l=50, r=20, t=50, b=50)
        )
        st.plotly_chart(fig_dk, use_container_width=True)

    calibration_error = abs(confidence_level - skill_level)
    overconfident = confidence_level > dk_val_at_skill

    if skill_level < 25 and confidence_level > 70:
        phase = "⛰️ SZCZYT GŁUPOTY (Mt. Stupid) — wysokie ryzyko kosztownych błędów!"
        color = "#ff1744"
    elif skill_level > 60 and confidence_level < 50:
        phase = "🌑 Syndrom Imposter — zbyt mała wiara we własne umiejętności"
        color = "#3498db"
    elif skill_level > 70 and confidence_level > 65:
        phase = "🏔️ Plateau Eksperta — kalibracja coraz lepsza"
        color = "#00e676"
    else:
        phase = "🕳️ Dolina Rozpaczy — uczysz się swoich ograniczeń"
        color = "#ffea00"

    st.markdown(f"""<div style='{CARD}'>
    <b>Faza:</b> <span style='color:{color}'>{phase}</span><br>
    <b>Błąd kalibracji:</b> {calibration_error:.0f}pp |
    {'Nadkonfidencja ⚠️' if overconfident else 'Niedokonfidencja (Imposter)'}<br>
    <b>DK Predykcja pewności na tym poziomie:</b> {dk_val_at_skill:.0f}%
    </div>""", unsafe_allow_html=True)


# ── TAB 3: SUPERFORECASTER TEST ───────────────────────────────────────────
with tabs[2]:
    st.markdown("### 🎯 Test Superforecastera (GJP)")
    st.caption("Na podstawie projektu Good Judgment Project (Tetlock & Gardner 2015)")

    questions = [
        {
            "q": "Jakie jest prawdopodobieństwo, że stopa inflacji CPI USA spadnie poniżej 2.5% w ciągu następnych 12 miesięcy?",
            "ref": "50%",
            "considerations": "Fed prognozuje powrót do 2%. Rynek mieszkaniowy wciąż wysoki. Rynek pracy silny."
        },
        {
            "q": "Jakie jest prawdopodobieństwo, że S&P 500 zakończy rok co najmniej 10% wyżej niż obecnie?",
            "ref": "35%",
            "considerations": "Historycznie: rynek rośnie >10%/rok w ~55% przypadków. Bieżące wyceny (P/E) wysokie."
        },
        {
            "q": "Jakie jest prawdopodobieństwo recesji technicznej (2 kwartały ujemnego PKB) w USA w ciągu 12 miesięcy?",
            "ref": "20%",
            "considerations": "Yield curve częściowo uninwertowana. Rynek pracy solidny. Historyczne: po inwersji recesja w ~70% w 18m."
        },
    ]

    for i, q_data in enumerate(questions):
        with st.expander(f"Pytanie {i+1}: {q_data['q'][:60]}..."):
            st.markdown(f"**Pełne pytanie:** {q_data['q']}")
            st.markdown(f"**Do rozważenia:** {q_data['considerations']}")
            user_p = st.slider("Twoje prawdopodobieństwo (%)", 0, 100, 50, 1, key=f"sfq_{i}")
            ref_p = int(q_data["ref"].strip("%"))
            diff = abs(user_p - ref_p)
            st.markdown(f"""
            Twoja odpowiedź: **{user_p}%** | Benchmark GJP: **{q_data["ref"]}**
            | Odchylenie: **{diff}pp** → {'✅ Dobra' if diff <= 15 else '⚠️ Sprawdź' if diff <= 30 else '❌ Duże odchylenie'}
            """)

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>🔑 10 Zasad Superforecastera (Tetlock)</div>
    1. <b>Triage</b>: Nie prognozuj niemożliwego do przewidzenia.<br>
    2. <b>Reference Class Forecasting</b>: Zacznij od bazowej stopy częstości.<br>
    3. <b>Inside vs Outside View</b>: Balansuj historię z bieżącym kontekstem.<br>
    4. <b>Think in Ranges</b>: Nigdy 0% ani 100%. 95% = odważnie.<br>
    5. <b>Granularity</b>: 60% vs 65% ma znaczenie.<br>
    6. <b>Update Bayesianly</b>: Aktualizuj małymi krokami na nowe dowody.<br>
    7. <b>Avoid Extremes</b>: Regresja do średniej jest regułą.<br>
    8. <b>Track Record</b>: Mierz Brier Score. Bez pomiaru, bez uczenia.<br>
    9. <b>Superteam</b>: Aggregation z innymi forecasterami > solo.<br>
    10. <b>Learn from Errors</b>: Najlepsi forecasterzy są skrupulatnie samokrytyczni.
    </div>""", unsafe_allow_html=True)


# ── TAB 4: TETLOCK RULES ──────────────────────────────────────────────────
with tabs[3]:
    st.markdown("### 📐 Zasady Tetlocka + Epistemic Framework")

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>🦊 Lis vs Jeż (Isaiah Berlin)</div>
    <b>Jeż:</b> Zna jedną wielką rzecz. Jeden model. Pewny siebie. Overconfident. Słabe Brier Score.<br>
    <b>Lis:</b> Zna wiele małych rzeczy. Wiele modeli. Calibrated uncertainty. Najlepsze predykcje.<br><br>
    Tetlock (20-letnie badanie forecasterów): <b>Lisy biją Jeże</b> — nawet gdy Jeże to eksperci w swojej dziedzinie.
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>📊 Dekomponujący Brier Score</div>
    <b>BS = Reliability + Resolution - Uncertainty</b><br><br>
    • <b>Reliability</b> (↓ = lepiej): Czy 70% prob → zdarzenia w 70% przypadków?<br>
    • <b>Resolution</b> (↑ = lepiej): Czy predykcje są zróżnicowane (nie zawsze 50%)?<br>
    • <b>Uncertainty</b>: Entropijny limit — im trudniejsze pytania, tym wyższy BS baseline.<br><br>
    <b>Skill Score (Brier Skill Score):</b> BSS = 1 - BS/BS_random<br>
    BSS > 0 → lepszy od losowego. BSS = 1 → perfekcja.
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>🔬 Zastosowanie w Finansach</div>
    • <b>Wycena opcji:</b> Implied probability = market forecast. Kalibracja rynku.<br>
    • <b>Prediction Markets (Polymarket):</b> Aggregate superforecasting w stawkach finansowych.<br>
    • <b>Risk Management:</b> Brier Score wewnętrznych modeli recesji vs outturns.<br>
    • <b>Stress Testing:</b> Prawdopodobieństwo scenariuszy MAD (Macroeconomic Adverse).<br>
    • <b>Analyst Ratings:</b> Śledzenie Brier Score analityków stocków → selekcja alfa.
    </div>""", unsafe_allow_html=True)

"""
28_Doradca.py — Doradca Inwestycyjny AI (Intelligent Portfolio Advisor)
=======================================================================
Moduł zbiera sygnały z:
  - GlobalPortfolio (portfel użytkownika)
  - Heartbeat Cache (VIX, YC, TED, GEX, HY, Sentiment, M2, …)
  - Reguły heurystyczne

Prezentuje:
  - Score Cards (Ochrona, Wzrost, Ryzyko, Ogółem)
  - Radar Chart 6-wymiarowy
  - Lista akcji „Co robić TERAZ" (priorytety)
  - Timeline prognozowanego portfela
  - Alert panel dla aktualnych ostrzeżeń
  - Szczegółowa analiza per aktywo
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.styling import apply_styling
from modules.global_settings import get_gs
from modules.advisor_engine import AdvisorEngine, AdvisorReport, _load_cache
from modules.background_updater import bg_engine

# ── Setup ──────────────────────────────────────────────────────────────────────
st.markdown(apply_styling(), unsafe_allow_html=True)
gs = get_gs()

# ── Kolory ─────────────────────────────────────────────────────────────────────
C_GREEN  = "#00e676"
C_YELLOW = "#ffea00"
C_RED    = "#ff1744"
C_CYAN   = "#00ccff"
C_PURPLE = "#a855f7"
C_BLUE   = "#3b82f6"

# ── Nagłówek ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
    background: linear-gradient(135deg,
        rgba(0,204,255,0.12) 0%,
        rgba(168,85,247,0.12) 50%,
        rgba(0,230,118,0.10) 100%);
    border: 1px solid rgba(0,204,255,0.3);
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
">
    <div style="position:absolute;top:-20px;right:-20px;width:150px;height:150px;
                border-radius:50%;background:radial-gradient(rgba(168,85,247,0.15),transparent);"></div>
    <h1 style="margin:0;font-size:2.1rem;letter-spacing:-0.5px;">🧭 Doradca Inwestycyjny AI</h1>
    <p style="color:#94a3b8;margin:8px 0 0 0;font-size:15px;">
        Synteza wniosków ze wszystkich modułów analitycznych · Portfel: <b style="color:{C_CYAN}">{gs.profile_name}</b>
        · Kapitał: <b style="color:{C_GREEN}">{gs.initial_capital:,.0f} PLN</b>
    </p>
    <p style="color:#6b7280;margin:6px 0 0 0;font-size:12px;">
        ⚡ Dane makro z Heartbeat Engine · Analiza oparta na regułach heurystycznych · Aktualizacja przy każdym otwarciu
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar: Horyzont + Informacje ────────────────────────────────────────────
st.sidebar.markdown("### 🎯 Parametry Analizy")
horizon = st.sidebar.slider(
    "📅 Horyzont Inwestycyjny",
    min_value=1, max_value=60, value=12, step=1,
    help="Wybierz horyzont od 1 miesiąca do 5 lat (co 1 miesiąc)",
    key="advisor_horizon"
)
hor_label = f"{horizon} mies." if horizon < 12 else (
    f"{horizon//12} rok" if horizon == 12 else
    f"{horizon//12} lata {horizon%12} mies." if horizon%12 else f"{horizon//12} lat"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 What-If (Symulator)")
st.sidebar.caption("Suwaki zmieniają analizę tymczasowo, **nie psując portfela globalnego**.")
sim_safe_raw = st.sidebar.slider(
    "Alokacja Bezpieczna (%)",
    min_value=0, max_value=100,
    value=int(gs.alloc_safe_pct*100)
)
sim_safe_pct = sim_safe_raw / 100.0
sim_risky_pct = 1.0 - sim_safe_pct

st.sidebar.markdown(f"""
<div style="background:rgba(0,204,255,0.08);border:1px solid rgba(0,204,255,0.2);
            border-radius:8px;padding:10px;font-size:12px;color:#94a3b8;">
⏱️ Wybrany horyzont: <b style="color:{C_CYAN}">{hor_label}</b><br>
📌 Profil: <b>{gs.profile_name}</b><br>
💰 Kapitał: <b>{gs.initial_capital:,.0f} PLN</b><br>
🔒 Bezpieczna (Sim): <b style="color:{C_BLUE}">{sim_safe_pct:.0%}</b><br>
⚡ Ryzykowna (Sim): <b style="color:{C_RED}">{sim_risky_pct:.0%}</b>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Pobierz najnowsze dane makro", type="primary", use_container_width=True):
    with st.spinner("🌐 Pobieranie aktualnych wskaźników z rynku (Yahoo Finance / FRED)..."):
        bg_engine.fetch_now_sync()
        st.toast("✅ Dane makro zaktualizowane!")
        st.rerun()

st.sidebar.caption("Heartbeat Cache analizuje na ustandaryzowanej lokalnej migawce. "
                   "Możesz włączyć stałe odświeżanie w tle w Ustawieniach.")
if st.sidebar.button("⚙️ Idź do Ustawień", use_container_width=True):
    st.switch_page("pages/0_Globalne_Ustawienia.py")

# ── Autostart Heartbeat jeśli brakuje danych ─────────────────────────────────
m_cache, g_cache = _load_cache()
if not m_cache:
    with st.spinner("Pierwsze wywołanie Doradcy — buduję cache rynkowy (szacowany czas: ok 10 s)..."):
        bg_engine.fetch_now_sync()
        st.rerun()

# ── Generuj raport ───────────────────────────────────────────────────────────
with st.spinner("🧠 Analizuję portfel i dane makroekonomiczne..."):
    try:
        engine = AdvisorEngine(gs=gs, sim_safe_pct=sim_safe_pct, sim_risky_pct=sim_risky_pct)
        report = engine.generate_report(horizon_months=horizon)
        data_ok = True
    except Exception as e:
        st.error(f"❌ Błąd generowania raportu: {e}")
        data_ok = False

if not data_ok:
    st.stop()

# ── Nagłówek raportu ──────────────────────────────────────────────────────────
_headline_color = C_GREEN if report.score_overall >= 65 else (C_YELLOW if report.score_overall >= 40 else C_RED)
st.markdown(f"""
<div style="background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.08);
            border-left:4px solid {_headline_color};border-radius:10px;
            padding:14px 20px;margin-bottom:20px;">
    <div style="font-size:18px;font-weight:700;color:{_headline_color};">{report.headline}</div>
    <div style="font-size:13px;color:#94a3b8;margin-top:6px;">{report.market_context}</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 1 — SCORE CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Ocena Portfela")

def _score_color(v: float, invert: bool = False) -> str:
    """Kolor score (0–100). invert=True dla ryzyka (wysokie ryzyko = czerwony)."""
    if invert:
        if v >= 65: return C_RED
        if v >= 40: return C_YELLOW
        return C_GREEN
    else:
        if v >= 65: return C_GREEN
        if v >= 40: return C_YELLOW
        return C_RED

def _gauge_fig(value: float, title: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 40, "color": color, "family": "Inter"}, "suffix": ""},
        title={"text": f"<span style='font-size:13px;color:#888'>{title}</span>",
               "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#555",
                     "tickvals": [0, 25, 50, 75, 100]},
            "bar":  {"color": color, "thickness": 0.12},
            "bgcolor": "#0a0b0e", "borderwidth": 1, "bordercolor": "#2a2a3a",
            "steps": [
                {"range": [0,  40],  "color": "rgba(255,23,68,0.15)"},
                {"range": [40, 65],  "color": "rgba(255,234,0,0.1)"},
                {"range": [65, 100], "color": "rgba(0,230,118,0.15)"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.85, "value": value},
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"})
    return fig

sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    c = _score_color(report.score_protection)
    st.plotly_chart(_gauge_fig(report.score_protection, "🛡️ Ochrona Kapitału", c),
                    use_container_width=True)
    st.markdown(f"<div style='text-align:center;font-size:11px;color:#6b7280;'>Jak dobrze portfel chroni przed stratami</div>", unsafe_allow_html=True)
with sc2:
    c = _score_color(report.score_growth)
    st.plotly_chart(_gauge_fig(report.score_growth, "📈 Potencjał Wzrostu", c),
                    use_container_width=True)
    st.markdown(f"<div style='text-align:center;font-size:11px;color:#6b7280;'>Szansa na zyski powyżej inflacji</div>", unsafe_allow_html=True)
with sc3:
    c = _score_color(report.score_risk, invert=True)
    st.plotly_chart(_gauge_fig(report.score_risk, "⚠️ Poziom Ryzyka", c),
                    use_container_width=True)
    st.markdown(f"<div style='text-align:center;font-size:11px;color:#6b7280;'>Wyższy = większe ryzyko systemowe</div>", unsafe_allow_html=True)
with sc4:
    c = _score_color(report.score_overall)
    st.plotly_chart(_gauge_fig(report.score_overall, "🎯 Ocena Ogólna", c),
                    use_container_width=True)
    st.markdown(f"<div style='text-align:center;font-size:11px;color:#6b7280;'>Syntetyczna ocena portfela i otoczenia</div>", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 2 — CO ROBIĆ TERAZ + RADAR
# ══════════════════════════════════════════════════════════════════════════════
col_actions, col_radar = st.columns([3, 2], gap="large")

with col_actions:
    st.markdown("## 🎯 Co Zrobić Teraz?")
    st.caption(f"Rekomendacje dla horyzontu: **{hor_label}** · Posortowane według priorytetu")

    PRIORITY_BADGE = {
        1: f"<span style='background:{C_RED}22;border:1px solid {C_RED}88;color:{C_RED};border-radius:4px;padding:1px 8px;font-size:10px;font-weight:700'>🚨 KRYTYCZNE</span>",
        2: f"<span style='background:{C_YELLOW}22;border:1px solid {C_YELLOW}88;color:{C_YELLOW};border-radius:4px;padding:1px 8px;font-size:10px;font-weight:700'>⚡ WAŻNE</span>",
        3: f"<span style='background:{C_GREEN}22;border:1px solid {C_GREEN}88;color:{C_GREEN};border-radius:4px;padding:1px 8px;font-size:10px;font-weight:700'>💡 OPTYMALIZACJA</span>",
    }
    CAT_COLOR = {
        "Ryzyko": C_RED, "Makro": C_YELLOW, "Alokacja": C_CYAN,
        "Dywersyfikacja": C_GREEN, "Inflacja": C_PURPLE, "Sentyment": "#f97316",
        "Portfel": C_GREEN, "Timing": C_YELLOW,
    }

    for action in report.actions[:8]:  # Max 8 akcji
        cat_c = CAT_COLOR.get(action.category, "#94a3b8")
        conf_pct = int(action.confidence * 100)
        conf_bar = f"""<div style="height:3px;border-radius:2px;background:rgba(255,255,255,0.08);margin-top:6px;">
            <div style="width:{conf_pct}%;height:3px;border-radius:2px;background:{cat_c};"></div></div>"""

        horizon_note = ""
        if action.horizon_months > 0:
            horizon_note = f"<span style='color:#6b7280;font-size:10px;'> · Istotne od ~{action.horizon_months} mies.</span>"

        st.markdown(f"""
<div style="background:rgba(15,17,26,0.8);border:1px solid rgba(255,255,255,0.06);
            border-left:3px solid {cat_c};border-radius:10px;padding:14px 16px;margin-bottom:10px;">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <span style="font-size:20px">{action.icon}</span>
        <div>
            <div style="font-weight:700;color:white;font-size:14px;">{action.title}</div>
            <div style="display:flex;gap:8px;margin-top:2px;">
                {PRIORITY_BADGE.get(action.priority, "")}
                <span style="background:{cat_c}22;border:1px solid {cat_c}55;color:{cat_c};border-radius:4px;padding:1px 6px;font-size:10px;">{action.category}</span> {horizon_note}
            </div>
        </div>
    </div>
    <div style="color:#94a3b8;font-size:13px;line-height:1.5;">{action.description}</div>
    <div style="display:flex;align-items:center;gap:8px;margin-top:8px;">
        <span style="font-size:10px;color:#6b7280;">Pewność: {conf_pct}%</span>
        {conf_bar}
    </div>
</div>
""", unsafe_allow_html=True)

with col_radar:
    st.markdown("## 📡 Radar Portfela")
    st.caption("6-wymiarowa ocena (0 = najsłabszy, 100 = najsilniejszy)")

    categories = [
        "Bezpieczeństwo", "Potencjał\nWzrostu", "Płynność",
        "Dywersyfikacja", "Ochrona przed\nInflacją", "Ochrona\nWalutowa"
    ]
    values = [
        report.radar_safety, report.radar_growth, report.radar_liquidity,
        report.radar_diversification, report.radar_inflation, report.radar_currency,
    ]
    # Close the radar
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values_closed, theta=cats_closed,
        fill="toself",
        fillcolor="rgba(0,204,255,0.15)",
        line=dict(color=C_CYAN, width=2),
        name="Twój portfel"
    ))
    # Benchmark (rynek neutralny)
    fig_radar.add_trace(go.Scatterpolar(
        r=[50, 50, 70, 50, 50, 50, 50], theta=cats_closed,
        fill="toself",
        fillcolor="rgba(168,85,247,0.06)",
        line=dict(color=C_PURPLE, width=1, dash="dot"),
        name="Benchmark (neutralny)"
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(size=9, color="#555"),
                gridcolor="rgba(255,255,255,0.05)",
                tickvals=[25, 50, 75, 100],
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#94a3b8"),
                gridcolor="rgba(255,255,255,0.06)",
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=60, r=60, t=60, b=60),
        legend=dict(font=dict(size=11, color="#94a3b8"), bgcolor="rgba(0,0,0,0)"),
        font=dict(color="white", family="Inter"),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Legenda ikon wymiarów
    for cat, val in zip(categories, values):
        c = _score_color(val)
        cat_clean = cat.replace("\n", " ")
        st.markdown(
            f"<span style='font-size:11px;color:#6b7280;'>{cat_clean}: </span>"
            f"<b style='color:{c}'>{val:.0f}/100</b>  ",
            unsafe_allow_html=True
        )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 3 — ALERT PANEL
# ══════════════════════════════════════════════════════════════════════════════
if report.alerts:
    st.markdown("## 🚨 Aktywne Alerty Makro")
    alert_cols = st.columns(min(len(report.alerts), 3))
    for i, alert in enumerate(report.alerts):
        col = alert_cols[i % 3]
        is_red = alert.startswith("🔴")
        bg = f"rgba(255,23,68,0.10)" if is_red else "rgba(255,234,0,0.08)"
        border = f"rgba(255,23,68,0.35)" if is_red else "rgba(255,234,0,0.3)"
        col.markdown(
            f"<div style='background:{bg};border:1px solid {border};border-radius:8px;"
            f"padding:10px 14px;font-size:13px;'>{alert}</div>",
            unsafe_allow_html=True
        )
    st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 4 — TIMELINE PORTFELA (MONTE CARLO)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"## 📈 Prognoza Stochastyczna (Monte Carlo) — {hor_label}")
st.caption(
    "⚠️ Symulacja z uwzględnieniem szumu Gaussa (parametryzowana przez VIX). "
    "Prezentuje zakresy: optymistyczny (P90), bazowy (P50) oraz pesymistyczny (P10)."
)

if hasattr(report, "timeline_full_data") and report.timeline_full_data:
    fd = report.timeline_full_data
    cap_start = gs.initial_capital
    cap_end = fd["p50"][-1] if fd["p50"] else cap_start
    total_ret = (cap_end / cap_start - 1) * 100 if cap_start > 0 else 0

    fig_tl = go.Figure()
    
    # Obszar ufności (P10 do P90)
    fig_tl.add_trace(go.Scatter(
        x=fd["labels"] + fd["labels"][::-1],
        y=fd["p90"] + fd["p10"][::-1],
        fill="toself",
        fillcolor="rgba(0, 204, 255, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Przedział ufności (P10-P90)"
    ))

    # Tylko TOS (Conservative)
    fig_tl.add_trace(go.Scatter(
        x=fd["labels"], y=fd["conservative"],
        fill=None, mode="lines", line=dict(color=C_BLUE, width=2, dash="dot"),
        name=f"Bezpieczna alokacja / TOS"
    ))
    
    # Scenariusz bazowy P50
    fig_tl.add_trace(go.Scatter(
        x=fd["labels"], y=fd["p50"],
        mode="lines+markers",
        line=dict(color=C_CYAN, width=3),
        marker=dict(size=6, color=C_CYAN, line=dict(color="white", width=1)),
        name="Wariant Bazowy (P50)",
    ))

    # Linia startowa
    fig_tl.add_hline(y=cap_start, line_dash="dash", line_color="rgba(255,255,255,0.2)",
                     annotation_text=f"Start: {cap_start:,.0f} PLN",
                     annotation_position="left")

    fig_tl.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.6)",
        height=380,
        yaxis_title="Wartość portfela (PLN)",
        xaxis_title="Horyzont",
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    # KPI timeline
    tl_k1, tl_k2, tl_k3, tl_k4 = st.columns(4)
    tl_k1.metric("💰 Średnia E[X]", f"{cap_end:,.0f} PLN")
    tl_k2.metric("📉 Max Drawdown", fd.get("max_drawdown", "?"))
    diff = cap_end - (fd["conservative"][-1] if fd["conservative"] else cap_start)
    tl_k3.metric("⚡ Zysk ponad infl/TOS", f"{diff:+,.0f} PLN")
    tl_k4.metric("💥 P10 (Pesymizm)", f"{fd['p10'][-1]:,.0f} PLN")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 5 — ANALIZA SYGNAŁÓW MAKRO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🔬 Szczegółowa Analiza Sygnałów Makro")
st.caption("Status poszczególnych wskaźników wpływających na rekomendacje Doradcy")

if report.signals:
    sig_data = []
    for s in report.signals:
        status_icon = "✅" if s.current_state == "ok" else ("🟡" if s.current_state == "warn" else "🔴")
        sig_data.append({
            "Wskaźnik":    s.name,
            "Wartość":     f"{s.value:.3f}" if s.value is not None else "N/A",
            "Próg OK":     f"{s.threshold_ok}",
            "Próg ALARM":  f"{s.threshold_warn}",
            "Kierunek":    "⬇️ niższy=lepszy" if s.direction == "lower_better" else "⬆️ wyższy=lepszy",
            "Status":      f"{status_icon} {s.current_state.upper()}",
            "Score":       f"{s.score():.0f}/100",
            "Waga":        f"{s.weight:.0%}",
        })
    sig_df = pd.DataFrame(sig_data)

    # Color coding w tabeli
    st.dataframe(sig_df, use_container_width=True, hide_index=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 6 — ANALIZA PORTFELA PER AKTYWO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 💼 Analiza Aktywów Portfela")
st.caption(f"Ocena poszczególnych składników portfela w kontekście horyzontu {hor_label}")

assets = gs.all_assets_unified
cap = gs.initial_capital

if assets:
    col_a1, col_a2 = st.columns(2, gap="large")

    for i, asset in enumerate(assets):
        col = col_a1 if i % 2 == 0 else col_a2
        seg = asset.get("segment", "Ryzykowny")
        typ = asset.get("type", "Inne")
        w   = asset.get("weight", 0)
        yld = asset.get("yield_pct", 0)
        amount = w / 100 * cap

        seg_color  = C_BLUE if seg == "Bezpieczny" else C_RED
        seg_icon   = "🔒" if seg == "Bezpieczny" else "⚡"
        risk_label = "Niskie" if seg == "Bezpieczny" else "Podwyższone"
        risk_color = C_GREEN if seg == "Bezpieczny" else C_YELLOW

        # Prosta ocena aktywa
        if seg == "Bezpieczny" and yld > 0:
            insight = f"Stopa {yld:.2f}% (brutto) = {yld*(1-0.19):.2f}% netto po Belce 19%. Stabilny fundament portfela."
        elif "ETF" in typ or "SPY" in asset.get("ticker","") or "QQQ" in asset.get("ticker",""):
            vix = engine.macro.get("VIX_1M", 20) or 20
            if vix > 25:
                insight = f"Uwaga: VIX={vix:.0f} — zmienność podwyższona. Unikaj powiększania pozycji przy obecnym ryzyku."
            else:
                insight = f"Środowisko WIX={vix:.0f} — neutralne. ETF w dobrej kondycji dla horyzontu {hor_label}."
        elif "BTC" in asset.get("ticker","") or "ETH" in asset.get("ticker",""):
            insight = "Aktywo spekulacyjne o wysokiej zmienności. Rekomendowany limit: max 5–10% całego portfela."
        else:
            insight = "Brak specyficznych alertów dla tego aktywa."

        yield_html = f'<span style="background:rgba(0,204,255,0.1);border:1px solid rgba(0,204,255,0.3);color:{C_CYAN};border-radius:4px;padding:1px 8px;font-size:11px;">Yield: {yld:.2f}%</span>' if yld > 0 else ''
        with col:
            st.markdown(f"""
<div style="background:rgba(15,17,26,0.8);border:1px solid rgba(255,255,255,0.06);
            border-left:3px solid {seg_color};border-radius:10px;
            padding:14px 16px;margin-bottom:10px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <div>
            <span style="font-size:16px;font-weight:700;color:white;">{seg_icon} {asset.get('name', asset.get('ticker','?'))}</span>
            <span style="color:#6b7280;font-size:12px;margin-left:8px;">({typ})</span>
        </div>
        <div style="text-align:right;">
            <div style="font-size:16px;font-weight:700;color:{C_CYAN}">{w:.1f}%</div>
            <div style="font-size:11px;color:#6b7280">{amount:,.0f} PLN</div>
        </div>
    </div>
    <div style="display:flex;gap:12px;margin-bottom:8px;">
        <span style="background:{seg_color}22;border:1px solid {seg_color}55;color:{seg_color};border-radius:4px;padding:1px 8px;font-size:11px;">{seg}</span>
        <span style="background:{risk_color}22;border:1px solid {risk_color}55;color:{risk_color};border-radius:4px;padding:1px 8px;font-size:11px;">Ryzyko: {risk_label}</span> {yield_html}
    </div>
    <div style="font-size:12px;color:#94a3b8;line-height:1.5;">💡 {insight}</div>
</div>
""", unsafe_allow_html=True)
else:
    st.info("Brak aktywów w portfelu. Skonfiguruj portfel w Globalnych Ustawieniach.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 7 — PODSUMOWANIE PORTFELA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📋 Podsumowanie Portfela")

summary_c1, summary_c2 = st.columns([3, 2])
with summary_c1:
    st.markdown(f"""
<div style="background:rgba(15,17,26,0.8);border:1px solid rgba(0,204,255,0.15);
            border-radius:12px;padding:20px;line-height:1.8;">
    {report.portfolio_assessment.replace(chr(10), '<br>')}
</div>
""", unsafe_allow_html=True)

with summary_c2:
    fig_mini_pie = go.Figure(go.Pie(
        labels=[a.get("name", a.get("ticker", "?")) for a in assets],
        values=[a["weight"] for a in assets],
        hole=0.5,
        marker_colors=[C_BLUE if a.get("segment")=="Bezpieczny" else "#f97316" for a in assets],
        texttemplate="%{label}<br>%{percent:.0%}",
        textfont_size=10,
    ))
    fig_mini_pie.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        height=250, margin=dict(t=5, b=5, l=5, r=5), showlegend=False,
    )
    st.plotly_chart(fig_mini_pie, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 8 — ASYSTENT REBALANCINGU (NOWA)
# ══════════════════════════════════════════════════════════════════════════════
if hasattr(report, "rebalancing_orders") and report.rebalancing_orders:
    st.markdown("## ⚖️ Asystent Transakcji (Rebalancing)")
    st.caption("Konkretne zlecenia generowane wg docelowej struktury. Doprowadź swój portfel do idealnych wag.")
    
    for order in report.rebalancing_orders:
        action = order["action"]
        asset = order["asset"]
        diff = order["diff_pln"]
        t = order["type"]
        
        bcolor = C_GREEN if "KUP" in action else (C_RED if "SPRZED" in action else C_CYAN)
        
        # Opcjonalny tag dla ryzykownych
        target_span = ""
        if "target_pct" in order:
            target_span = f'<span style="float:right;color:#6b7280;font-size:12px;">Cel: {order["target_pct"]:.1f}% kapitału</span>'
            
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.02); border-left:4px solid {bcolor}; 
                    padding:10px 16px; margin-bottom:8px; border-radius:6px; display:flex; 
                    justify-content:space-between; align-items:center;">
            <div>
                <b style="color:{bcolor};">{action}</b>
                <span style="color:#d1d5db; margin-left:10px; font-size:15px;">{asset}</span>
            </div>
            <div>
                <span style="font-weight:700; color:white;">{diff:,.0f} PLN</span>
                <br>{target_span}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div style="text-align:center;color:#374151;font-size:11px;padding:16px;">
    🧭 Doradca Inwestycyjny AI v1.0 · Barbell Strategy Dashboard<br>
    <span style="color:#1f2937;">
    Analiza oparta na danych makroekonomicznych z Heartbeat Engine i strukturze portfela z Globalnych Ustawień.
    Nie stanowi porady inwestycyjnej w rozumieniu przepisów prawa. Inwestuj odpowiedzialnie.
    </span>
</div>
""", unsafe_allow_html=True)

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
import datetime, json

# ── Cached helpers (F4.2 — eliminacja zbednych rekurencji) ──────────────────
@st.cache_data(ttl=300, show_spinner=False)
def _cached_sparkline(ticker: str, period: str = "3mo"):
    """Pobiera dane sparkline z cache (TTL 5 min) — eliminuje podwójne pobieranie."""
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period, interval="1d")
        if df is not None and not df.empty and "Close" in df.columns:
            return df["Close"].dropna().tolist()[-60:]
    except Exception:
        pass
    return []

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
# SEKCJA 1 — NOWOCZESNE SCORE CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📊 Ocena Portfela")

def _score_color(v: float, invert: bool = False) -> str:
    if invert:
        return C_RED if v >= 65 else (C_YELLOW if v >= 40 else C_GREEN)
    return C_GREEN if v >= 65 else (C_YELLOW if v >= 40 else C_RED)


def _gauge_fig(value: float, title: str, color: str) -> go.Figure:
    """Mini gauge chart dla sekcji emerytalnej (zakres 0–100)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 28, "color": color, "family": "Inter"}, "suffix": "/100"},
        title={"text": title, "font": {"size": 12, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#374151", "tickwidth": 1,
                     "tickvals": [0, 25, 50, 75, 100]},
            "bar":  {"color": color, "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,0.06)",
            "borderwidth": 1,
            "steps": [
                {"range": [0, 40],  "color": "rgba(255,23,68,0.10)"},
                {"range": [40, 65], "color": "rgba(255,234,0,0.08)"},
                {"range": [65, 100],"color": "rgba(0,230,118,0.10)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 2},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=180,
        margin=dict(t=40, b=10, l=20, r=20),
        font=dict(family="Inter"),
    )
    return fig

def _score_card(title: str, subtitle: str, value: float, color: str, icon: str) -> str:
    pct = int(value)
    prev_key = f"_advisor_prev_{title}"
    prev = st.session_state.get(prev_key, value)
    st.session_state[prev_key] = value
    diff = value - prev
    arrow = (f"<span style='color:{C_GREEN};font-size:11px;'>▲ {diff:+.1f}</span>" if diff > 0.5
             else f"<span style='color:{C_RED};font-size:11px;'>▼ {diff:+.1f}</span>" if diff < -0.5
             else "<span style='color:#6b7280;font-size:11px;'>— bez zmian</span>")
    pulse = f"box-shadow:0 0 18px {color}55;" if value < 35 else ""
    return f"""
<div style="background:linear-gradient(135deg,rgba(15,17,26,0.95),rgba(26,28,40,0.9));
    border:1px solid {color}44;border-top:3px solid {color};border-radius:14px;
    padding:20px 18px;{pulse}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <span style="font-size:26px;">{icon}</span>
    {arrow}
  </div>
  <div style="font-size:38px;font-weight:800;color:{color};margin:8px 0 2px 0;letter-spacing:-1px;">{value:.0f}</div>
  <div style="font-size:12px;font-weight:700;color:white;letter-spacing:0.5px;">{title}</div>
  <div style="font-size:10px;color:#6b7280;margin-top:3px;">{subtitle}</div>
  <div style="height:6px;background:rgba(255,255,255,0.07);border-radius:4px;margin-top:12px;overflow:hidden;">
    <div style="height:6px;width:{pct}%;background:linear-gradient(90deg,{color}99,{color});border-radius:4px;
      transition:width 0.8s ease;"></div>
  </div>
</div>"""

sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    st.markdown(_score_card("Ochrona Kapitału", "Jak dobrze portfel chroni",
        report.score_protection, _score_color(report.score_protection), "🛡️"), unsafe_allow_html=True)
with sc2:
    st.markdown(_score_card("Potencjał Wzrostu", "Szansa na zyski > inflacji",
        report.score_growth, _score_color(report.score_growth), "📈"), unsafe_allow_html=True)
with sc3:
    st.markdown(_score_card("Poziom Ryzyka", "Wyższy = więcej zagrożeń",
        report.score_risk, _score_color(report.score_risk, invert=True), "⚠️"), unsafe_allow_html=True)
with sc4:
    _ci = getattr(engine, '_score_ci', (report.score_overall, report.score_overall))
    _ci_txt = f"przedział ufności: {_ci[0]:.0f}–{_ci[1]:.0f}" if _ci[0] != _ci[1] else ""
    st.markdown(_score_card("Ocena Ogólna", f"Synteza makro + portfel | {_ci_txt}",
        report.score_overall, _score_color(report.score_overall), "🎯"), unsafe_allow_html=True)

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

        # XAI — evidence chain (CFA Institute 2025)
        xai_html = ""
        evidence = getattr(action, 'evidence', [])
        explanation = getattr(action, 'explanation_chain', '')
        if evidence or explanation:
            ev_tags = " ".join(f"<code style='background:rgba(255,255,255,0.06);border-radius:3px;padding:1px 5px;font-size:10px;color:#94a3b8;'>{e}</code>" for e in evidence[:3])
            xai_html = f"""
            <div style='border-top:1px solid rgba(255,255,255,0.05);margin-top:8px;padding-top:7px;'>
              <span style='font-size:10px;color:#6b7280;'>📊 Dane: </span>{ev_tags}
              {'<div style="font-size:10px;color:#64748b;margin-top:3px;">💡 ' + explanation + '</div>' if explanation else ''}
            </div>"""

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
    # Benchmark — dynamiczny wg profilu ryzyka (Konserwatywny/Zrównoważony/Agresywny)
    _bm = getattr(engine, 'radar_benchmark', [55, 55, 70, 55, 55, 50])
    _bm_closed = _bm + [_bm[0]]
    _profile_label = getattr(gs, 'profile_name', 'Benchmark')
    fig_radar.add_trace(go.Scatterpolar(
        r=_bm_closed, theta=cats_closed,
        fill="toself",
        fillcolor="rgba(168,85,247,0.06)",
        line=dict(color=C_PURPLE, width=1, dash="dot"),
        name=f"Benchmark ({_profile_label})"
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
# SEKCJA 2.5 — MAKRO TERMOMETR
# ══════════════════════════════════════════════════════════════════════════════
if report.signals:
    st.markdown("## 🌡️ Termometr Rynku")
    st.caption("Stan aktywnych wskaźników makroekonomicznych — im dłuższy pasek, tym bardziej odbiega od normy.")

    _sig_show = [s for s in report.signals if s.value is not None][:8]
    if _sig_show:
        _names = [s.name for s in _sig_show]
        _scores = [s.score() for s in _sig_show]
        _states = [s.current_state for s in _sig_show]
        _colors = [C_GREEN if st == "ok" else (C_YELLOW if st == "warn" else C_RED) for st in _states]
        _icons  = ["✅" if st == "ok" else ("🟡" if st == "warn" else "🔴") for st in _states]
        _labels = [f"{ic} {nm}: {sc:.0f}/100" for ic, nm, sc in zip(_icons, _names, _scores)]

        _fig_thermo = go.Figure(go.Bar(
            y=_labels, x=_scores,
            orientation="h",
            marker=dict(
                color=_colors,
                opacity=0.85,
                line=dict(color="rgba(0,0,0,0)", width=0)
            ),
            text=[f"{s:.0f}" for s in _scores],
            textposition="inside",
            insidetextfont=dict(color="white", size=11),
        ))
        _fig_thermo.add_vline(x=50, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                              annotation_text="Neutralny")
        _fig_thermo.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,11,20,0.5)",
            height=max(180, len(_sig_show) * 38),
            margin=dict(l=10, r=20, t=10, b=10),
            xaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.05)", title="Score (0–100)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
            showlegend=False,
        )
        st.plotly_chart(_fig_thermo, use_container_width=True)

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
        bg = "rgba(255,23,68,0.10)" if is_red else "rgba(255,234,0,0.08)"
        border = "rgba(255,23,68,0.35)" if is_red else "rgba(255,234,0,0.3)"
        col.markdown(
            f"<div style='background:{bg};border:1px solid {border};border-radius:8px;"
            f"padding:10px 14px;font-size:13px;'>{alert}</div>",
            unsafe_allow_html=True
        )
    st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 4 — MONTE CARLO (ulepszony: P5/P95 + strefa zysk/strata)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"## 📈 Prognoza Stochastyczna (Monte Carlo) — {hor_label}")
st.caption("Symulacja 500 ścieżek portfela z szumem Gaussa. P5/P95 = scenariusze ekstremalne.")

if hasattr(report, "timeline_full_data") and report.timeline_full_data:
    fd = report.timeline_full_data
    cap_start = gs.initial_capital
    cap_end = fd["p50"][-1] if fd["p50"] else cap_start

    fig_tl = go.Figure()

    # Strefa zysku (P50 > start) — zielona
    fig_tl.add_trace(go.Scatter(
        x=fd["labels"] + fd["labels"][::-1],
        y=[max(v, cap_start) for v in fd["p50"]] + [cap_start] * len(fd["labels"]),
        fill="toself", fillcolor="rgba(0,230,118,0.07)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", name="Strefa Zysku", showlegend=True
    ))
    # Strefa ekstremalna P5–P95
    if "p90" in fd and "p10" in fd:
        fig_tl.add_trace(go.Scatter(
            x=fd["labels"] + fd["labels"][::-1],
            y=fd["p90"] + fd["p10"][::-1],
            fill="toself", fillcolor="rgba(0,204,255,0.10)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", name="Przedział P10–P90"
        ))
    # TOS (bezpieczna)
    fig_tl.add_trace(go.Scatter(
        x=fd["labels"], y=fd["conservative"],
        mode="lines", line=dict(color=C_BLUE, width=2, dash="dot"),
        name="Bezpieczna / TOS"
    ))
    # P50 bazowy
    fig_tl.add_trace(go.Scatter(
        x=fd["labels"], y=fd["p50"],
        mode="lines+markers",
        line=dict(color=C_CYAN, width=3),
        marker=dict(size=6, color=C_CYAN, line=dict(color="white", width=1)),
        name="Wariant Bazowy (P50)"
    ))
    # P10 pesymistyczny
    if "p10" in fd:
        fig_tl.add_trace(go.Scatter(
            x=fd["labels"], y=fd["p10"],
            mode="lines", line=dict(color=C_RED, width=1.5, dash="dot"),
            name="Pesymistyczny (P10)"
        ))
    # Linia startowa
    fig_tl.add_hline(y=cap_start, line_dash="dash", line_color="rgba(255,255,255,0.25)",
                     annotation_text=f"Start: {cap_start:,.0f} PLN", annotation_position="left")

    fig_tl.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.6)", height=380,
        yaxis_title="Wartość portfela (PLN)", xaxis_title="Horyzont",
        legend=dict(orientation="h", y=1.05, x=0, font=dict(size=11)),
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    tl_k1, tl_k2, tl_k3, tl_k4 = st.columns(4)
    tl_k1.metric("💰 Scenariusz Bazowy (P50)", f"{cap_end:,.0f} PLN",
                 delta=f"{(cap_end/cap_start-1)*100:+.1f}%" if cap_start > 0 else None)
    tl_k2.metric("📉 Max Drawdown (est.)", fd.get("max_drawdown", "?"))
    diff = cap_end - (fd["conservative"][-1] if fd["conservative"] else cap_start)
    tl_k3.metric("⚡ Zysk ponad TOS", f"{diff:+,.0f} PLN")
    tl_k4.metric("💥 Pesymistyczny (P10)", f"{fd['p10'][-1]:,.0f} PLN")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 4.5 — WATERFALL: WKŁAD SYGNAŁÓW DO SCORE
# ══════════════════════════════════════════════════════════════════════════════
if hasattr(report, "score_contributions") and report.score_contributions:
    st.markdown("## ⚖️ Wkład Sygnałów do Oceny Portfela")
    st.caption("Waterfall: każdy słupek = ile punktów dany wskaźnik dodał (↑) lub odjął (↓) od oceny ogólnej.")
    _contribs = report.score_contributions
    _wf_names  = [c["name"] for c in _contribs]
    _wf_deltas = [c["delta"] for c in _contribs]
    _wf_colors = [C_GREEN if d >= 0 else C_RED for d in _wf_deltas]
    _wf_text   = [f"{d:+.1f}" for d in _wf_deltas]
    _fig_wf = go.Figure(go.Bar(
        x=_wf_names, y=_wf_deltas,
        marker_color=_wf_colors, opacity=0.85,
        text=_wf_text, textposition="outside",
        textfont=dict(size=10, color="white"),
    ))
    _fig_wf.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    _fig_wf.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.5)", height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Wkład [pkt]", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(tickangle=-25, gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    st.plotly_chart(_fig_wf, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SEKCJA 5 — SZCZEGÓŁOWA TABELA SYGNAŁÓW
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("🔬 Szczegółowe Dane Sygnałów Makro", expanded=False):
    if report.signals:
        sig_data = []
        for s in report.signals:
            status_icon = "✅" if s.current_state == "ok" else ("🟡" if s.current_state == "warn" else "🔴")
            sig_data.append({
                "Wskaźnik": s.name,
                "Wartość": f"{s.value:.3f}" if s.value is not None else "N/A",
                "Status": f"{status_icon} {s.current_state.upper()}",
                "Score": f"{s.score():.0f}/100",
                "Waga": f"{s.weight:.0%}",
                "Wkład": f"{(s.score()-50)*s.weight:+.2f} pkt",
            })
        st.dataframe(pd.DataFrame(sig_data), use_container_width=True, hide_index=True)

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

        # Insight
        if seg == "Bezpieczny" and yld > 0:
            insight = f"Stopa {yld:.2f}% (brutto) = {yld*(1-0.19):.2f}% netto po Belce 19%. Stabilny fundament portfela."
        elif "ETF" in typ or "SPY" in asset.get("ticker","") or "QQQ" in asset.get("ticker",""):
            vix = engine.macro.get("VIX_1M", 20) or 20
            insight = (f"Uwaga: VIX={vix:.0f} — zmienność podwyższona. Unikaj powiększania pozycji."
                       if vix > 25 else f"VIX={vix:.0f} — środowisko neutralne. ETF w dobrej kondycji dla {hor_label}.")
        elif "BTC" in asset.get("ticker","") or "ETH" in asset.get("ticker",""):
            insight = "Aktywo spekulacyjne. Rekomendowany limit: max 5–10% całego portfela."
        else:
            insight = "Brak specyficznych alertów dla tego aktywa."

        # Sparkline SVG (dane z report.sparkline_data)
        tkr = asset.get("ticker", "")
        spark_prices = report.sparkline_data.get(tkr, [])
        sparkline_html = ""
        if len(spark_prices) >= 5:
            _mn, _mx = min(spark_prices), max(spark_prices)
            _rng = max(_mx - _mn, 1e-9)
            _sw, _sh = 100, 28
            _pts = " ".join(
                f"{int((_sw/(len(spark_prices)-1))*i)},{int(_sh - _sh*(v-_mn)/_rng)}"
                for i, v in enumerate(spark_prices)
            )
            _chg = (spark_prices[-1]/spark_prices[0]-1)*100 if spark_prices[0] else 0
            _sc = C_GREEN if _chg >= 0 else C_RED
            sparkline_html = (
                f"<div style='display:flex;align-items:center;gap:8px;margin-top:6px;'>"
                f"<svg width='{_sw}' height='{_sh}'>"
                f"<polyline points='{_pts}' fill='none' stroke='{_sc}' stroke-width='1.8'/></svg>"
                f"<span style='font-size:10px;color:{_sc};'>{_chg:+.1f}% (30D)</span></div>"
            )

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
    {sparkline_html}
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
# SEKCJA 7.5 — INTELIGENTNA EMERYTURA FIRE & BARBELL
# ══════════════════════════════════════════════════════════════════════════════
if hasattr(report, "ret_readiness_score"):
    st.markdown("## 🎯 Inteligentna Emerytura (FIRE) & Barbell Integration")
    st.caption("Zaawansowana ocena szans na udaną emeryturę (uwzględnia horyzont-zależny SWR Morningstar 2025, CAPE, Glide Path i SORR).")

    rc1, rc2, rc3, rc4 = st.columns(4)

    with rc1:
        c_readiness = _score_color(report.ret_readiness_score)
        st.plotly_chart(_gauge_fig(report.ret_readiness_score, "🔥 Readiness Score", c_readiness), use_container_width=True)
        # Fidelity 2024 label
        _rrs = report.ret_readiness_score
        _rrs_lbl = ("🟢 On Track" if _rrs >= 85 else ("🟡 Likely on Track" if _rrs >= 70 else ("🟠 Somewhat on Track" if _rrs >= 50 else "🔴 Off Track")))
        st.markdown(f"<div style='text-align:center;font-size:11px;color:#94a3b8;'>{_rrs_lbl} · Fidelity 2024</div>", unsafe_allow_html=True)

    with rc2:
        c_prob = _score_color(report.ret_success_prob * 100)
        st.plotly_chart(_gauge_fig(report.ret_success_prob * 100, "🎲 Szansa Sukcesu (P50)", c_prob), use_container_width=True)
        st.markdown(f"<div style='text-align:center;font-size:11px;color:#6b7280;'>Przetrwanie kapitału z uwzgl. podatków (IKE/Zwykłe)</div>", unsafe_allow_html=True)

    with rc3:
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.metric("SWR (Morningstar 2025)", f"{report.ret_adjusted_swr*100:.2f}%",
                  help="Horyzont-zależny SWR wg tabeli Morningstar 2025 + korekta CAPE Kitces 2022.")
        st.metric("Lat do FIRE", f"{report.ret_years_to_fire:.1f} lat" if report.ret_years_to_fire < 90 else "Niedościgły cel")

    with rc4:
        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        # SORR KPI (Pfau 2024)
        _sorr_sig = next((s for s in report.signals if "SORR" in s.name), None)
        if _sorr_sig is not None:
            _sorr_val = _sorr_sig.value or 0
            _sorr_score = _sorr_sig.score()
            _sorr_state = _sorr_sig.current_state
            _sorr_c = C_GREEN if _sorr_state == "ok" else (C_YELLOW if _sorr_state == "warn" else C_RED)
            _sorr_lbl = "✅ Niskie" if _sorr_state == "ok" else ("⚠️ Umiarkowane" if _sorr_state == "warn" else "🚨 Wysokie")
            st.markdown(f"""
<div style='background:rgba(0,0,0,0.3);border:1px solid {_sorr_c}44;border-top:3px solid {_sorr_c};
            border-radius:10px;padding:14px;text-align:center;'>
  <div style='font-size:11px;color:#6b7280;'>⏳ SORR Risk (Pfau 2024)</div>
  <div style='font-size:26px;font-weight:800;color:{_sorr_c};'>{_sorr_score:.0f}/100</div>
  <div style='font-size:11px;color:{_sorr_c};'>{_sorr_lbl}</div>
  <div style='font-size:10px;color:#4b5563;margin-top:4px;'>Ryzyko Sekwencji Zwrotów</div>
</div>""", unsafe_allow_html=True)
        else:
            st.metric("SORR Risk", "N/A")



    st.markdown(f"""
    <div style="background:rgba(255,100,50,0.08);border:1px solid rgba(255,100,50,0.2);
                border-left:3px solid rgba(255,100,50,0.8);border-radius:10px;
                padding:14px 16px;margin-bottom:10px;margin-top:10px;">
        <span style="font-size:14px;color:#d1d5db;">💡 {report.ret_assessment}</span>
    </div>
    """, unsafe_allow_html=True)

    # Trajektoria
    if hasattr(report, "ret_timeline_labels") and report.ret_timeline_labels:
        fig_r_tl = go.Figure()
        fig_r_tl.add_trace(go.Scatter(
            x=report.ret_timeline_labels, y=report.ret_timeline_values,
            mode="lines+markers", line=dict(color="#ff6432", width=2, dash="dot"),
            name="Majątek na Emeryturze (P50)"
        ))
        fig_r_tl.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=180, margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"), xaxis=dict(gridcolor="rgba(255,255,255,0.05)")
        )
        st.plotly_chart(fig_r_tl, use_container_width=True)

    # ── Bucket Strategy (deterministyczny kalkulator) ─────────────────────────
    with st.expander("🪣 Bucket Strategy — Szybki Kalkulator (Daryanani 2008)", expanded=False):
        st.caption("Podziel kapitał emerytalny na 3 wiadra wg horyzontu. Eliminuje sprzedaż akcji podczas bessy.")
        _bk_cap = gs.initial_capital
        _bk_exp = gs.ret_monthly_expense * 12
        _bk_zus = gs.ret_zus_monthly * 12
        _bk_net = max(0.0, _bk_exp - _bk_zus)
        _b1 = _bk_net * 3          # 3 lata gotówka
        _b2_pv = sum((_bk_net * 1.03**y) / 1.04**y for y in range(1, 8))  # 7 lat obligacje ~4%
        _b3 = max(0.0, _bk_cap - _b1 - _b2_pv)
        _bc1, _bc2, _bc3 = st.columns(3)
        _bc1.metric("🪣 Wiadro 1 (Bezpieczne)", f"{_b1:,.0f} PLN", f"{_b1/_bk_cap*100:.0f}% | 3 lata gotówka/TOS")
        _bc2.metric("🪣 Wiadro 2 (Obligacje)", f"{_b2_pv:,.0f} PLN", f"{_b2_pv/_bk_cap*100:.0f}% | 7 lat TIPS/obligacje")
        _bc3.metric("🪣 Wiadro 3 (Akcje/ETF)", f"{_b3:,.0f} PLN", f"{_b3/_bk_cap*100:.0f}% | 10+ lat wzrost")
        st.caption("📖 Szczegółowy kalkulator + Refill Schedule → moduł **Emerytura** zakładka 🪣 Bucket Strategy")

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

# ── Eksport Raportu HTML (F4.3) ───────────────────────────────────────
@st.cache_data
def get_report_html(report, gs):
    _ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    _html_report = f"""
<!DOCTYPE html>
<html lang="pl">
<head><meta charset="UTF-8"><title>Raport Doradcy AI — {_ts}</title>
<style>
  body{{font-family:Inter,sans-serif;background:#0a0b14;color:#d1d5db;padding:32px;max-width:900px;margin:auto;}}
  h1{{color:#00ccff;}} h2{{color:#a855f7;border-bottom:1px solid #1f2937;padding-bottom:8px;}}
  .card{{background:#111827;border:1px solid #1f2937;border-radius:10px;padding:16px;margin:12px 0;}}
  .score{{font-size:36px;font-weight:900;}} .ok{{color:#00e676;}} .warn{{color:#ffea00;}} .alarm{{color:#ff1744;}}
  .action{{border-left:4px solid #00ccff;padding:10px 14px;margin:8px 0;background:#0d1117;border-radius:4px;}}
  .evidence{{font-size:11px;color:#6b7280;margin-top:4px;}}
  footer{{text-align:center;color:#374151;font-size:11px;margin-top:32px;}}
</style></head>
<body>
<h1>🧠 Raport Doradcy Inwestycyjnego AI</h1>
<p style="color:#6b7280;">Wygenerowano: {_ts} | Portfel: {gs.profile_name}</p>
<h2>📊 Oceny główne</h2>
<div class="card">
  <span class="score">{report.score_overall:.0f}/100</span> Ocena Ogólna
  • Ochrona: {report.score_protection:.0f} • Wzrost: {report.score_growth:.0f} • Ryzyko: {report.score_risk:.0f}
</div>
<h2>✅ Rekomendacje ({len(report.actions)})</h2>
{''.join(f'<div class="action"><b>{a.icon} {a.title}</b> [{a.category}]<br>{a.description}<div class="evidence">{"&nbsp;".join(getattr(a,"evidence",[])) or ""}<br><i>{getattr(a,"explanation_chain","")}</i></div></div>' for a in report.actions[:8])}
<h2>🎯 Emerytura FIRE</h2>
<div class="card">
  Readiness Score: <b>{getattr(report,'ret_readiness_score',0):.0f}/100</b>
  • Sukces MC: {getattr(report,'ret_success_prob',0):.1%}
  • SWR: {getattr(report,'ret_adjusted_swr',0)*100:.2f}%<br>
  {getattr(report,'ret_assessment','')}
</div>
<footer>Doradca AI V4.0 — Intelligent Barbell Dashboard. Nie stanowi porady inwestycyjnej.</footer>
</body></html>
"""
    return _html_report

st.divider()
_dl_col, _info_col = st.columns([1, 3])
with _dl_col:
    st.download_button(
        label="📄 Pobierz Raport HTML",
        data=get_report_html(report, gs).encode("utf-8"),
        file_name=f"doradca_raport_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html",
        help="Eksportuj pełny raport doradczy do pliku HTML — gotowy do wydruku lub udostępnienia."
    )
with _info_col:
    st.caption("💡 Raport HTML zawiera wszystkie oceny, rekomendacje z uzasadnieniami (XAI) i dane emerytalne FIRE.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center;color:#374151;font-size:11px;padding:16px;">
    🧠 Doradca Inwestycyjny AI V4.0 • Barbell Strategy Dashboard<br>
    <span style="color:#1f2937;">
    XAI • Morningstar 2025 SWR • SORR (Pfau 2024) • Bucket Strategy (Daryanani 2008) • Bootstrap CI.<br>
    Nie stanowi porady inwestycyjnej. Inwestuj odpowiedzialnie.
    </span>
</div>
""", unsafe_allow_html=True)

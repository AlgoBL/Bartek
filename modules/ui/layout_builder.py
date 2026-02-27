"""
layout_builder.py — Komponenty layoutu Streamlit dla Barbell Dashboard.

Wydziela logikę renderowania (Streamlit UI) z app.py (wzorzec MVC).
Funkcje przyjmują przetworzone dane i renderują komponenty UI.
"""

import streamlit as st
from modules.ui.charts_builder import (
    build_advanced_gauge,
    build_vix_term_structure,
    build_credit_spread_chart,
)


# ─── HELPER: GAUGE WITH TOOLTIP ───────────────────────────────────────────────

def render_gauge_with_tooltip(
    label: str,
    fig,
    help_text: str,
    overlap_margin: str = "-20px",
) -> None:
    """
    Renderuje gauge Plotly z nagłówkiem + tooltipem HTML powyżej.
    Zachowuje równomierne wyrównanie kolumn przez stałą wysokość tytułu.
    """
    safe = (
        help_text
        .replace('"', "'")
        .replace("\n", " ")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    st.markdown(
        f"""
        <div style='
            position: relative;
            z-index: 10;
            height: 22px;
            line-height: 22px;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            font-family: Inter, sans-serif;
            font-size: 12px;
            color: #aaa;
            padding: 0 12px;
            cursor: help;
            margin-bottom: {overlap_margin};
        ' title="{safe}">
            <span style='color:#555;margin-right:4px;'>&#9432;</span>{label}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── PILLAR HEADER ────────────────────────────────────────────────────────────

def render_pillar_header(label: str, color: str) -> None:
    """Renderuje nagłówek filara (h4) z kolorową ramką."""
    style = (
        "text-align:center;font-size:13px;font-weight:700;"
        "letter-spacing:1px;margin-bottom:4px;padding:4px 0;border-radius:6px;"
    )
    bg_alpha = "rgba"
    # Convert hex to rgba background
    color_map = {
        "#ff1744": "rgba(255,23,68,0.08)",
        "#3498db": "rgba(52,152,219,0.08)",
        "#2ecc71": "rgba(46,204,113,0.08)",
        "#f1c40f": "rgba(241,196,15,0.08)",
        "#a855f7": "rgba(168,85,247,0.08)",
    }
    bg_color = color_map.get(color, "rgba(100,100,100,0.08)")
    st.markdown(
        f"<h4 style='{style}color:{color};background:{bg_color};'>{label}</h4>",
        unsafe_allow_html=True,
    )


# ─── BUSINESS CYCLE CARD ──────────────────────────────────────────────────────

def render_business_cycle_card(
    phase: str,
    desc: str,
    icon: str,
    color: str,
    yield_curve: float,
    initial_claims: float | None,
) -> None:
    """Renderuje kartę fazy cyklu koniunkturalnego."""
    claims_html = ""
    if initial_claims:
        claims_html = (
            f"&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"<span style='color:#aaa;font-size:10px;'>"
            f"Claims: <b style='color:#f39c12'>{initial_claims/1000:.0f}k</b></span>"
        )
    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,#0f111a,#1a1c28);padding:18px 14px;
                    border-radius:12px;text-align:center;border:1px solid #2a2a3a;
                    height:310px;display:flex;flex-direction:column;justify-content:center;'>
            <div style='font-size:52px;line-height:1;'>{icon}</div>
            <div style='color:{color};margin-top:8px;font-size:18px;font-weight:700;'>{phase}</div>
            <div style='color:#888;font-size:11px;margin-top:6px;line-height:1.35;'>{desc}</div>
            <div style='margin-top:14px;border-top:1px solid #2a2a3a;padding-top:10px;'>
                <span style='color:#aaa;font-size:10px;'>10Y-3M: <b style='color:{color}'>{yield_curve:+.2f}%</b></span>
                {claims_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── SAFE-HAVEN MINI CARDS ────────────────────────────────────────────────────

def render_safe_haven_cards(usd: float | None, gold: float | None, oil: float | None) -> None:
    """Renderuje 3 mini-karty USD/Gold/Oil."""
    from modules.app_helpers import get_help_usd, get_help_gold, get_help_oil

    rA, rB, rC = st.columns(3)
    assets = [
        (rA, "🇺🇸 USD",  usd,  "",  get_help_usd,  "#00ccff"),
        (rB, "🥇 Gold",  gold, "$", get_help_gold, "#f1c40f"),
        (rC, "🛢️ Oil",   oil,  "$", get_help_oil,  "#00ccff"),
    ]
    for col_ref, lbl, val, suf, help_fn, v_color in assets:
        with col_ref:
            if val:
                safe_tip = (
                    help_fn(val)
                    .replace('"', "'")
                    .replace("\n", " ")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                st.markdown(
                    f"""
                    <div title="{safe_tip}"
                         style='background:#0f111a;border:1px solid #2a2a3a;border-radius:8px;
                                padding:8px 4px;text-align:center;margin-top:45px;cursor:default;margin-bottom:8px;'>
                        <div style='font-size:10px;color:#777;'>&#9432; {lbl}</div>
                        <div style='font-size:18px;font-weight:700;color:{v_color};'>{suf}{val:.1f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ─── INTELLIGENCE REPORT BANNER ───────────────────────────────────────────────

def render_intelligence_report(
    report_text: str,
    report_color: str,
    score: float,
    alerts: list[str],
    timestamp: str,
) -> None:
    """
    Renderuje dolny banner Intelligence Report z alertami i score.
    """
    n_red    = sum(1 for a in alerts if "🔴" in a)
    n_yellow = sum(1 for a in alerts if "🟡" in a)
    score_color = "#e74c3c" if n_red > 1 else "#f39c12" if n_yellow > 1 else "#2ecc71"
    alerts_html = "".join(
        [f"<span style='margin-right:18px;font-size:12px;color:#ccc;'>{a}</span>" for a in alerts]
    )
    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,#0a0b0e,#0f111a);
                    padding:16px 20px;border-radius:12px;
                    border-left:5px solid {report_color};border:1px solid #1e1e2e;margin-top:4px;'>
            <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;'>
                <span style='font-size:11px;color:#555;'>⏱ {timestamp} &nbsp;|&nbsp; V9.5 Terminal &nbsp;|&nbsp;
                    🎯 Score: <b style='color:{score_color}'>{score:.0f}/100</b> &nbsp;|&nbsp;
                    🔴 Alerty: <b style='color:#e74c3c'>{n_red}</b> &nbsp;
                    🟡 Ostrzeżenia: <b style='color:#f39c12'>{n_yellow}</b>
                </span>
            </div>
            <p style='margin:0 0 10px 0;color:white;font-size:16px;line-height:1.5;'>
                <b>📋 Raport:</b> {report_text}
            </p>
            <div style='border-top:1px solid #1e1e2e;padding-top:8px;flex-wrap:wrap;display:flex;'>
                {alerts_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── METRICS ROW ──────────────────────────────────────────────────────────────

def render_metrics_row(metrics: dict, cols_per_row: int = 4) -> None:
    """
    Renderuje wiersz kart metryk w siatce.

    Parameters
    ----------
    metrics : dict {"label": (value_str, delta_str, delta_color)}
    """
    items = list(metrics.items())
    for row_start in range(0, len(items), cols_per_row):
        row_items = items[row_start: row_start + cols_per_row]
        cols = st.columns(len(row_items))
        for col, (label, (value_str, delta_str, delta_color)) in zip(cols, row_items):
            with col:
                delta_html = (
                    f"<div style='font-size:11px;color:{delta_color};'>{delta_str}</div>"
                    if delta_str else ""
                )
                st.markdown(
                    f"""
                    <div style='background:#0f111a;border:1px solid #2a2a3a;border-radius:10px;
                                padding:14px 12px;text-align:center;'>
                        <div style='font-size:11px;color:#777;margin-bottom:4px;'>{label}</div>
                        <div style='font-size:22px;font-weight:700;color:white;'>{value_str}</div>
                        {delta_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

"""
charts_builder.py — Centralna biblioteka figur Plotly dla Barbell Dashboard.

Wydziela logikę wizualną z app.py (MVC: widok oddzielony od kontrolera).
Każda funkcja zwraca go.Figure gotową do st.plotly_chart().
"""

import numpy as np
import plotly.graph_objects as go


# ─── PALETA ───────────────────────────────────────────────────────────────────
P_GREEN  = "#00e676"
P_YELLOW = "#ffea00"
P_RED    = "#ff1744"
C_GREEN  = "#2ecc71"
C_YELLOW = "#f39c12"
C_RED    = "#e74c3c"

DARK_BG        = "rgba(0,0,0,0)"
PLOT_BG        = "rgba(15,15,25,0.9)"
FONT_FAMILY    = "Inter, monospace"
GRID_COLOR     = "#1c1c2e"


# ─── 1. GŁÓWNY GAUGE REGIME RADAR ─────────────────────────────────────────────

def build_regime_radar(score: float) -> go.Figure:
    """Duży gauge środkowy Control Center z kolorem zależnym od score."""
    if score <= 30:
        needle_color = P_GREEN
    elif score <= 65:
        needle_color = P_YELLOW
    else:
        needle_color = P_RED

    zone_label = (
        "🟢 HOSSA / RISK-ON" if score <= 30
        else ("🔴 PANIKA / ALARM" if score > 65 else "🟡 NEUTRAL / OSTROŻNIE")
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "font": {"size": 58, "color": needle_color, "family": FONT_FAMILY},
            "suffix": "",
        },
        title={
            "text": (
                "<span style='font-size:15px;color:#aaa;letter-spacing:2px;'>REGIME SCORE</span>"
                f"<br><span style='font-size:11px;color:{needle_color};'>{zone_label}</span>"
            ),
            "font": {"size": 15, "color": "#aaa", "family": "Inter"}
        },
        gauge={
            "axis": {
                "range": [1, 100],
                "tickwidth": 1,
                "tickcolor": "#555",
                "tickvals": [1, 25, 50, 75, 100],
                "ticktext": ["1", "25", "50", "75", "100"],
                "tickfont": {"size": 11, "color": "#888", "family": "Inter"},
            },
            "bar": {"color": needle_color, "thickness": 0.10},
            "bgcolor": "#0a0b0e",
            "borderwidth": 1,
            "bordercolor": "#2a2a3a",
            "steps": [
                {"range": [1,  30],  "color": "rgba(0, 230, 118, 0.20)"},
                {"range": [30, 65],  "color": "rgba(255, 234, 0, 0.13)"},
                {"range": [65, 100], "color": "rgba(255, 23, 68, 0.23)"},
            ],
            "threshold": {
                "line": {"color": needle_color, "width": 5},
                "thickness": 0.92,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=330,
        margin=dict(l=40, r=40, t=90, b=20),
        paper_bgcolor=DARK_BG,
        font={"color": "white", "family": "Inter"},
        annotations=[
            dict(
                text="HOSSA", x=0.12, y=0.08, xref="paper", yref="paper",
                font=dict(size=9, color=P_GREEN, family="Inter"), showarrow=False, opacity=0.7
            ),
            dict(
                text="PANIKA", x=0.88, y=0.08, xref="paper", yref="paper",
                font=dict(size=9, color=P_RED, family="Inter"), showarrow=False, opacity=0.7
            ),
        ],
    )
    return fig


# ─── 2. MINI ADVANCED GAUGE (5-PILLAR) ────────────────────────────────────────

def build_advanced_gauge(
    title: str,
    value: float | None,
    min_val: float,
    max_val: float,
    invert: bool = False,
    suffix: str = "",
    prefix: str = "",
) -> go.Figure:
    """Mini arc-gauge z 3 strefami kolorów — używany w 5-pillar siatce."""
    if value is None:
        return go.Figure()

    z1 = (max_val - min_val) * 0.35 + min_val
    z2 = (max_val - min_val) * 0.65 + min_val

    if not invert:
        steps = [
            {"range": [min_val, z1], "color": C_GREEN},
            {"range": [z1,      z2], "color": C_YELLOW},
            {"range": [z2, max_val], "color": C_RED},
        ]
    else:
        steps = [
            {"range": [min_val, z1], "color": C_RED},
            {"range": [z1,      z2], "color": C_YELLOW},
            {"range": [z2, max_val], "color": C_GREEN},
        ]

    if abs(value) < 10:      fmt = ".2f"
    elif abs(value) < 1000:  fmt = ".1f"
    else:                    fmt = ".0f"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            "prefix": prefix,
            "suffix": suffix,
            "font": {"size": 32, "color": "white", "family": FONT_FAMILY},
            "valueformat": fmt,
        },
        title={"text": "", "font": {"size": 1}},
        gauge={
            "axis": {
                "range": [min_val, max_val],
                "tickwidth": 1,
                "tickcolor": "#555",
                "nticks": 4,
                "tickfont": {"size": 9, "color": "#aaa"},
            },
            "bar": {"color": "rgba(0,0,0,0)"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": steps,
            "threshold": {
                "line": {"color": "white", "width": 5},
                "thickness": 0.85,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=5, b=5),
        paper_bgcolor=DARK_BG,
        font={"color": "white", "family": "Inter"},
    )
    return fig


# ─── 3. VIX TERM STRUCTURE BAR ────────────────────────────────────────────────

def build_vix_term_structure(vix_1m: float | None, vxmt: float | None) -> go.Figure:
    """Poziomy bar porównujący VIX 1M vs VXMT 3M (contango/backwardation)."""
    if vix_1m is None:
        return go.Figure()
    vxmt = vxmt or vix_1m
    ratio = vix_1m / vxmt if vxmt else 1.0
    is_back = ratio > 1.02
    c1 = "#ff1744" if is_back else P_GREEN
    c2 = "#ff5252" if is_back else "#00bcd4"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[vix_1m], y=["VIX 1M"], orientation="h", marker_color=c1,
        text=f"{vix_1m:.1f}", textposition="inside",
        textfont=dict(size=14, color="white"), name="VIX 1M",
    ))
    fig.add_trace(go.Bar(
        x=[vxmt], y=["VXMT 3M"], orientation="h", marker_color=c2,
        text=f"{vxmt:.1f}", textposition="inside",
        textfont=dict(size=14, color="white"), name="VXMT 3M",
    ))
    fig.update_layout(
        height=120,
        margin=dict(l=10, r=10, t=25, b=8),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font={"color": "white", "family": "Inter"},
        barmode="group",
        showlegend=False,
        xaxis=dict(range=[0, max(vix_1m, vxmt) * 1.3], gridcolor=GRID_COLOR, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=11)),
    )
    return fig


# ─── 4. CREDIT SPREAD CHART ───────────────────────────────────────────────────

def build_credit_spread_chart(hy: float | None, cs_baa_aaa: float | None) -> go.Figure:
    """Poziomy bar dla HY Spread i Credit Spread BAA-AAA."""
    if hy is None and cs_baa_aaa is None:
        return go.Figure()
    labels, values, colors = [], [], []
    if hy is not None:
        labels.append("HY Spread (bps)")
        values.append(hy)
        colors.append(C_RED if hy > 600 else P_YELLOW if hy > 400 else P_GREEN)
    if cs_baa_aaa is not None:
        labels.append("Credit Spread %")
        values.append(cs_baa_aaa)
        colors.append(C_RED if cs_baa_aaa > 3.5 else P_YELLOW if cs_baa_aaa > 2.5 else P_GREEN)

    fig = go.Figure()
    for lbl, val, col in zip(labels, values, colors):
        fig.add_trace(go.Bar(
            x=[val], y=[lbl], orientation="h", marker_color=col,
            text=f"{val:.1f}", textposition="inside",
            textfont=dict(size=13, color="white"), name=lbl,
        ))
    fig.update_layout(
        height=120 if len(labels) > 1 else 90,
        margin=dict(l=2, r=2, t=25, b=5),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font={"color": "white", "family": "Inter"},
        showlegend=False,
        xaxis=dict(tickfont=dict(size=8), gridcolor=GRID_COLOR),
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


# ─── 5. WEALTH PATHS FAN CHART ────────────────────────────────────────────────

def build_wealth_paths_chart(
    wealth_paths: np.ndarray,
    n_years: int,
    initial_capital: float = 10_000,
    percentiles: tuple = (5, 25, 50, 75, 95),
) -> go.Figure:
    """
    Fan-chart Monte Carlo ścieżek bogactwa z percentylami i fallback ścieżkami.

    Parameters
    ----------
    wealth_paths : (n_sims, n_days+1)
    n_years      : horyzont (oś X w latach)
    """
    n_sims, n_points = wealth_paths.shape
    x_years = np.linspace(0, n_years, n_points)

    p5   = np.percentile(wealth_paths, 5,  axis=0)
    p25  = np.percentile(wealth_paths, 25, axis=0)
    p50  = np.percentile(wealth_paths, 50, axis=0)
    p75  = np.percentile(wealth_paths, 75, axis=0)
    p95  = np.percentile(wealth_paths, 95, axis=0)

    fig = go.Figure()

    # Shade 5–95
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_years, x_years[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself",
        fillcolor="rgba(0,230,118,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5–95 percentyl",
    ))
    # Shade 25–75
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_years, x_years[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself",
        fillcolor="rgba(0,230,118,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25–75 percentyl",
    ))
    # Median
    fig.add_trace(go.Scatter(
        x=x_years, y=p50,
        line=dict(color=P_GREEN, width=2.5),
        name="Mediana",
    ))
    # p5 / p95 borders
    for arr, lbl, col in [(p5, "P5 (pesymistyczny)", C_RED), (p95, "P95 (optymistyczny)", "#00bcd4")]:
        fig.add_trace(go.Scatter(
            x=x_years, y=arr,
            line=dict(color=col, width=1, dash="dot"),
            name=lbl,
        ))

    # Baseline
    fig.add_hline(y=initial_capital, line_dash="dash", line_color="#555", line_width=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="white", family=FONT_FAMILY),
        xaxis_title="Lata",
        yaxis_title="Wartość portfela (PLN)",
        legend=dict(orientation="h", y=-0.15, x=0),
        height=420,
        margin=dict(l=50, r=20, t=20, b=60),
    )
    return fig


# ─── 6. FACTOR BAR CHART (FF5 / AE) ──────────────────────────────────────────

def build_factor_bar_chart(
    betas: dict,
    r_squared: float,
    alpha_annual_pct: float,
    title: str = "Factor Decomposition",
) -> go.Figure:
    """Beta exposure bars."""
    colors = [C_GREEN if v >= 0 else C_RED for v in betas.values()]
    fig = go.Figure(go.Bar(
        x=list(betas.keys()),
        y=list(betas.values()),
        marker_color=colors,
        text=[f"{v:.3f}" for v in betas.values()],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{title} | α={alpha_annual_pct:.2f}%/yr | R²={r_squared:.2%}",
        xaxis_title="Czynnik",
        yaxis_title="Ekspozycja Beta",
        template="plotly_dark",
        height=400,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color="white"),
        yaxis=dict(zeroline=True, zerolinecolor="gray", zerolinewidth=1),
    )
    return fig


# ─── 7. DRAWDOWN CHART ────────────────────────────────────────────────────────

def build_drawdown_chart(portfolio_series: np.ndarray, dates=None) -> go.Figure:
    """Drawdown timeline z cieniowaniem czerwonym."""
    peaks = np.maximum.accumulate(portfolio_series)
    drawdowns = (portfolio_series - peaks) / np.where(peaks != 0, peaks, 1)

    x = dates if dates is not None else np.arange(len(drawdowns))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=drawdowns * 100,
        fill="tozeroy",
        fillcolor="rgba(255,23,68,0.25)",
        line=dict(color=C_RED, width=1.2),
        name="Drawdown",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PLOT_BG,
        yaxis_title="Drawdown (%)",
        height=250,
        font=dict(color="white", family=FONT_FAMILY),
        margin=dict(l=50, r=20, t=20, b=40),
    )
    return fig


# ─── 8. CORRELATION HEATMAP ───────────────────────────────────────────────────

def build_correlation_heatmap(corr_matrix: np.ndarray, labels: list[str]) -> go.Figure:
    """Korelacja/DCC macierz aktywów."""
    fig = go.Figure(go.Heatmap(
        z=corr_matrix,
        x=labels,
        y=labels,
        colorscale="RdYlGn",
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        height=380,
        font=dict(color="white", family=FONT_FAMILY),
        margin=dict(l=60, r=20, t=20, b=60),
    )
    return fig


# ─── 9. EVT QQ-PLOT ───────────────────────────────────────────────────────────

def build_evt_qq_plot(
    empirical_excesses: np.ndarray,
    xi: float,
    sigma: float,
    title: str = "EVT QQ-Plot (GPD fit)",
) -> go.Figure:
    """QQ-plot empirycznych ogonów vs dopasowanego GPD."""
    from scipy.stats import genpareto

    n = len(empirical_excesses)
    quantiles = (np.arange(1, n + 1) - 0.5) / n
    theoretical = genpareto.ppf(quantiles, c=xi, scale=sigma)
    emp_sorted = np.sort(empirical_excesses)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical, y=emp_sorted,
        mode="markers",
        marker=dict(color=P_GREEN, size=5, opacity=0.7),
        name="Dane empiryczne",
    ))
    # 45-degree line
    lim = max(theoretical.max(), emp_sorted.max())
    fig.add_trace(go.Scatter(
        x=[0, lim], y=[0, lim],
        mode="lines",
        line=dict(color=C_RED, dash="dash", width=1.5),
        name="Idealne dopasowanie",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Kwantyl GPD",
        yaxis_title="Kwantyl empiryczny",
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PLOT_BG,
        height=380,
        font=dict(color="white", family=FONT_FAMILY),
    )
    return fig


# ─── 10. DCC CORRELATION EVOLUTION ───────────────────────────────────────────

def build_dcc_correlation_evolution(
    dates,
    corr_matrix_series: list[np.ndarray],
    asset_pair: tuple[int, int] = (0, 1),
    asset_labels: list[str] | None = None,
) -> go.Figure:
    """Ewolucja DCC korelacji pary aktywów w czasie."""
    i, j = asset_pair
    corr_values = [m[i, j] for m in corr_matrix_series]
    label = (
        f"{asset_labels[i]} / {asset_labels[j]}"
        if asset_labels else f"Asset {i} / Asset {j}"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates if dates is not None else np.arange(len(corr_values)),
        y=corr_values,
        line=dict(color="#00ccff", width=1.8),
        name=label,
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#555", line_width=1)
    fig.update_layout(
        title=f"DCC-GARCH Korelacja: {label}",
        yaxis_title="Korelacja warunkowa R_t",
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PLOT_BG,
        height=300,
        font=dict(color="white", family=FONT_FAMILY),
    )
    return fig


# ─── 11. AUTOENCODER ANOMALY SCORE ────────────────────────────────────────────

def build_anomaly_score_chart(
    dates,
    anomaly_scores: np.ndarray,
    threshold: float | None = None,
    title: str = "Autoencoder Anomaly Score (Reconstruction Error)",
) -> go.Figure:
    """Timeline anomalii z progiem alarmowym."""
    x = dates if dates is not None else np.arange(len(anomaly_scores))
    color_array = [C_RED if s > (threshold or np.inf) else P_GREEN for s in anomaly_scores]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=anomaly_scores,
        fill="tozeroy",
        fillcolor="rgba(0,230,118,0.12)",
        line=dict(color=P_GREEN, width=1.5),
        name="Anomaly Score",
    ))
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=C_RED,
            line_width=2,
            annotation_text=f"Alert threshold ({threshold:.4f})",
            annotation_font_color=C_RED,
        )
    fig.update_layout(
        title=title,
        yaxis_title="Błąd rekonstrukcji MSE",
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PLOT_BG,
        height=300,
        font=dict(color="white", family=FONT_FAMILY),
    )
    return fig


# ─── 12. BLACK-LITTERMAN POSTERIOR vs MARKET IMPLIED ─────────────────────────

def build_bl_returns_comparison(
    asset_names: list[str],
    market_implied: np.ndarray,
    bl_posterior: np.ndarray,
) -> go.Figure:
    """Black-Litterman posterior vs CAPM-implied returns comparison."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=asset_names, y=market_implied * 100,
        name="Market-Implied (CAPM π)",
        marker_color="#00bcd4",
    ))
    fig.add_trace(go.Bar(
        x=asset_names, y=bl_posterior * 100,
        name="BL Posterior (μ_BL)",
        marker_color=P_GREEN,
    ))
    fig.update_layout(
        title="Black-Litterman: AI Views vs Market Equilibrium",
        xaxis_title="Aktywo",
        yaxis_title="Oczekiwany zwrot (%/rok)",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PLOT_BG,
        height=400,
        font=dict(color="white", family=FONT_FAMILY),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig

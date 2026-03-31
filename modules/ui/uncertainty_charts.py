"""
uncertainty_charts.py — Naukowe wykresy niepewności (V.8)
===========================================================
Ref:
  - Hullman, J. (2024) "Uncertainty Visualization"
  - Kay, M. et al. (2024) "Framing Effects in Uncertainty Visualization" CHI 2024
  - Fernandes, M. et al. (2018) "Uncertainty Displays Using Quantile Dotplots
    or CDFs Improve Transit Decision-Making" CHI 2018

Komponenty:
  1. uncertainty_fan_chart()    — fan chart z gradientowymi CI (jak ECDC COVID Fan)
  2. quantile_dot_plot()        — Quantile Dot Plot dla finalnych wartości MC
  3. hops_animation_html()      — HOPs (Hypothetical Outcome Plots) — JS animacja
  4. add_gradient_confidence_bands() — nakładka na istniejący wykres Plotly
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Sequence


# ─────────────────────────────────────────────────────────────────────────────
#  1. UNCERTAINTY FAN CHART
# ─────────────────────────────────────────────────────────────────────────────

def uncertainty_fan_chart(
    paths: np.ndarray,
    index: Optional[Sequence] = None,
    quantiles: tuple = (0.05, 0.25, 0.50, 0.75, 0.95),
    title: str = "Symulacja Monte Carlo — Fan Chart",
    y_label: str = "Wartość",
    x_label: str = "Okres",
    baseline_value: Optional[float] = None,
    show_paths_n: int = 20,
    color_green: bool = True,
) -> go.Figure:
    """
    Fan (wachlarzowy) wykres niepewności z gradientowymi przedziałami ufności.
    Zamiast pokazywać 100 nakładających się ścieżek, pokazuje kwantyle z gradientem.

    Parameters
    ----------
    paths         : (n_simulations, n_periods) — macierz ścieżek MC
    index         : opcjonalne etykiety osi X (daty / numery)
    quantiles     : kwantyle do wyświetlenia (domyślnie 5/25/50/75/95)
    baseline_value: opcjonalna linia startowa / punkt bezpieczeństwa
    show_paths_n  : ile losowych ścieżek tła pokazać (0 = żadnej)
    color_green   : True = ziel. paleta (growth), False = niebieska (neutral)

    Returns
    -------
    go.Figure z fan chartem
    """
    n_sim, n_per = paths.shape
    x = list(index) if index is not None else list(range(n_per))

    # Oblicz kwantyle
    qs = np.quantile(paths, quantiles, axis=0)
    q05, q25, q50, q75, q95 = qs

    # Kolory
    if color_green:
        main_color  = "#00e676"
        band_colors = [
            "rgba(0,230,118,0.06)",   # 90% CI — najbledsze
            "rgba(0,230,118,0.14)",   # 50% CI
        ]
    else:
        main_color  = "#00ccff"
        band_colors = [
            "rgba(0,204,255,0.06)",
            "rgba(0,204,255,0.14)",
        ]

    fig = go.Figure()

    # Losowe ścieżki w tle (szarawe, bardzo przezroczyste)
    if show_paths_n > 0:
        chosen = np.random.choice(n_sim, min(show_paths_n, n_sim), replace=False)
        for idx in chosen:
            fig.add_trace(go.Scatter(
                x=x, y=paths[idx],
                mode="lines",
                line=dict(color="rgba(180,180,200,0.08)", width=0.8),
                showlegend=False,
                hoverinfo="skip",
            ))

    # 90% CI (szeroki pas — q5 do q95)
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(q95) + list(q05[::-1]),
        fill="toself",
        fillcolor=band_colors[0],
        line=dict(color="rgba(0,0,0,0)"),
        name="90% CI",
        showlegend=True,
        hoverinfo="skip",
    ))

    # 50% CI (wąski pas — q25 do q75)
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(q75) + list(q25[::-1]),
        fill="toself",
        fillcolor=band_colors[1],
        line=dict(color="rgba(0,0,0,0)"),
        name="50% CI",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Mediana (linia główna)
    fig.add_trace(go.Scatter(
        x=x, y=q50,
        mode="lines",
        line=dict(color=main_color, width=2.5),
        name="Mediana",
    ))

    # Q5 i Q95 jako cienkie linie
    for q, q_label in [(q05, "Q5"), (q95, "Q95")]:
        fig.add_trace(go.Scatter(
            x=x, y=q,
            mode="lines",
            line=dict(color=main_color, width=0.8, dash="dot"),
            opacity=0.4,
            showlegend=False,
            name=q_label,
        ))

    # Baseline (np. wartość startowa)
    if baseline_value is not None:
        fig.add_hline(
            y=baseline_value,
            line_dash="dash", line_color="#888",
            annotation_text="Start",
            annotation_font_size=9,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#e2e4f0")),
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e4f0", family="Inter"),
        xaxis=dict(title=x_label, gridcolor="#1c1c2e", gridwidth=0.5),
        yaxis=dict(title=y_label, gridcolor="#1c1c2e", gridwidth=0.5),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
        ),
        margin=dict(l=50, r=20, t=55, b=40),
        hovermode="x unified",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  2. QUANTILE DOT PLOT (QDP)
# ─────────────────────────────────────────────────────────────────────────────

def quantile_dot_plot(
    final_values: np.ndarray,
    n_dots: int = 50,
    title: str = "Quantile Dot Plot — Rozkład Końcowych Wartości",
    x_label: str = "Wartość końcowa",
    baseline: Optional[float] = None,
    unit: str = "",
) -> go.Figure:
    """
    Quantile Dot Plot (Kay et al. 2016, Fernandes et al. 2018).
    Każda kropka = 1/(n_dots) populacji. Czytelniejszy niż histogram.
    Ref: "Uncertainty displays using quantile dotplots... improve decision-making"

    Parameters
    ----------
    final_values : (n,) array wartości końcowych (np. kapitał po 250 tradach)
    n_dots       : liczba kropek (zazwyczaj 50 lub 100)
    baseline     : opcjonalna linia bazowa (np. wartość startowa)
    unit         : suffix wartości (np. "PLN", "%")
    """
    vals = np.sort(np.asarray(final_values, dtype=float))
    n    = len(vals)

    # Wyznacz wartości kwantyli dla n_dots równo rozmieszczonych kwantyli
    q_levels  = np.linspace(0.5 / n_dots, 1 - 0.5 / n_dots, n_dots)
    dot_vals  = np.quantile(vals, q_levels)

    # Kolor: zielony jeśli > baseline, czerwony jeśli < baseline
    if baseline is not None:
        colors = [
            "#00e676" if v >= baseline else "#ff1744"
            for v in dot_vals
        ]
    else:
        # Gradient od czerwonego do zielonego
        norm = (dot_vals - dot_vals.min()) / (dot_vals.max() - dot_vals.min() + 1e-9)
        colors = [
            f"rgba({int(255*(1-n_val))},{int(230*n_val)},{int(118*n_val)},0.85)"
            for n_val in norm
        ]

    # Układ: jitter na osi Y dla estetyki dot plot
    np.random.seed(0)
    y_jitter = np.random.uniform(-0.3, 0.3, n_dots)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dot_vals,
        y=y_jitter,
        mode="markers",
        marker=dict(
            color=colors,
            size=11,
            line=dict(color="rgba(255,255,255,0.2)", width=0.5),
            symbol="circle",
        ),
        text=[f"Q{int(q*100)}: {v:.0f}{unit}" for q, v in zip(q_levels, dot_vals)],
        hovertemplate="%{text}<extra></extra>",
        name="Kwantyle",
    ))

    # Mediana — wąskie diamenty
    med     = float(np.median(vals))
    pct_lo  = float(np.mean(vals < (baseline or med)))

    fig.add_vline(
        x=med,
        line_dash="dash", line_color="#ffea00", line_width=1.5,
        annotation_text=f"Mediana: {med:.0f}{unit}",
        annotation_position="top",
        annotation_font_size=9,
        annotation_font_color="#ffea00",
    )

    if baseline is not None:
        fig.add_vline(
            x=baseline,
            line_dash="solid", line_color="#888", line_width=1,
            annotation_text=f"Start: {baseline:.0f}{unit}",
            annotation_position="bottom",
            annotation_font_size=9,
        )
        # Procent powyżej baseline
        pct_above = float(np.mean(vals >= baseline)) * 100
        fig.add_annotation(
            x=0.02, y=1.02,
            xref="paper", yref="paper",
            text=f"↑ {pct_above:.0f}% powyżej startu",
            showarrow=False,
            font=dict(size=10, color="#00e676"),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#e2e4f0")),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e4f0", family="Inter"),
        xaxis=dict(title=x_label, gridcolor="#1c1c2e", gridwidth=0.5),
        yaxis=dict(
            visible=False,
            range=[-1.0, 1.0],
        ),
        margin=dict(l=40, r=20, t=45, b=35),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  3. HOPs — Hypothetical Outcome Plots (JavaScript)
# ─────────────────────────────────────────────────────────────────────────────

def hops_animation_html(
    paths: np.ndarray,
    n_hops: int = 30,
    fps: int = 2,
    height: int = 320,
    baseline_value: Optional[float] = None,
    color: str = "#00e676",
) -> str:
    """
    Generuje HTML+JS dla HOPs (Hypothetical Outcome Plots).
    Animuje sekwencję losowych ścieżek Monte Carlo (1 naraz) zamiast
    nakładania wszystkich — zgodnie z badaniami Kay et al. (2016).

    Ref: Kay et al. "When (ish) is My Bus? User-centered Visualizations
         of Uncertainty in Everyday, Mobile Predictive Systems." CHI 2016.

    Parameters
    ----------
    paths          : (n_simulations, n_periods) — macierz ścieżek
    n_hops         : ile ścieżek animować (kolejno, w pętli)
    fps            : frame per second (prędkość animacji)
    height         : wysokość px
    baseline_value : opcjonalna linia startowa

    Returns
    -------
    str — HTML gotowy do st.components.v1.html() lub st.markdown(unsafe=True)
    """
    n_sim, n_per = paths.shape
    chosen_idx   = np.random.choice(n_sim, min(n_hops, n_sim), replace=False)
    selected     = paths[chosen_idx]

    y_min = float(np.min(selected) * 0.95)
    y_max = float(np.max(selected) * 1.05)

    # Normalizuj ścieżki do SVG viewBox (1000px × height)
    w, h = 1000, height
    x_coords = [int(i / (n_per - 1) * (w - 80) + 40) for i in range(n_per)]

    def norm_y(y_val):
        return int(h - ((y_val - y_min) / (y_max - y_min + 1e-9)) * (h - 60) - 30)

    paths_json = [
        [int(norm_y(v)) for v in path]
        for path in selected
    ]

    baseline_y = int(norm_y(baseline_value)) if baseline_value is not None else None
    baseline_html = (
        f'<line x1="40" y1="{baseline_y}" x2="{w-40}" y2="{baseline_y}" '
        f'stroke="#888" stroke-width="1" stroke-dasharray="4"/>'
    ) if baseline_y else ""

    x_json = x_coords
    interval_ms = int(1000 / fps)

    return f"""
<div style="background:rgba(0,0,0,0);border:1px solid #2a2a3a;border-radius:12px;padding:6px;margin:4px 0;">
<div style="font-size:10px;color:#666;text-align:right;padding:0 8px;">
  HOPs — Hypothetical Outcome Plots (Kay et al. 2016)
</div>
<svg id="hops-svg" viewBox="0 0 {w} {h}" style="width:100%;height:{h}px;">
  <defs>
    <linearGradient id="pathGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="{color}" stop-opacity="0.6"/>
      <stop offset="100%" stop-color="{color}" stop-opacity="1.0"/>
    </linearGradient>
  </defs>
  {baseline_html}
  <polyline id="hops-path" points="" fill="none"
    stroke="url(#pathGrad)" stroke-width="2.5"
    stroke-linecap="round" stroke-linejoin="round"/>
  <text id="hops-counter" x="{w-50}" y="20" fill="#555" font-size="11"
    font-family="Inter,sans-serif" text-anchor="end">1/{n_hops}</text>
</svg>
</div>
<script>
(function() {{
  const paths  = {paths_json};
  const xArr   = {x_json};
  const total  = paths.length;
  let current  = 0;
  let animId   = null;
  let step     = 0;
  const animDot = {max(n_per // 30, 2)};  // punkty per frame

  const svg  = document.getElementById('hops-svg');
  const poly = document.getElementById('hops-path');
  const ctr  = document.getElementById('hops-counter');

  function drawFrame() {{
    const path = paths[current];
    const pts  = [];
    const end  = Math.min(step + animDot, path.length);
    for (let i = 0; i < end; i++) {{
      pts.push(xArr[i] + "," + path[i]);
    }}
    poly.setAttribute('points', pts.join(' '));
    step += animDot;
    if (step >= path.length) {{
      clearInterval(animId);
      setTimeout(() => {{
        current = (current + 1) % total;
        step    = 0;
        poly.setAttribute('points', '');
        ctr.textContent = (current + 1) + '/{n_hops}';
        animId = setInterval(drawFrame, {max(interval_ms // max(n_per // 30, 2), 16)});
      }}, {interval_ms * 3});
    }}
  }}

  animId = setInterval(drawFrame, {max(interval_ms // max(n_per // 30, 2), 16)});
}})();
</script>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  4. ADD GRADIENT CONFIDENCE BANDS (nakładka na istniejący wykres)
# ─────────────────────────────────────────────────────────────────────────────

def add_gradient_confidence_bands(
    fig: go.Figure,
    x: Sequence,
    y_lower_90: np.ndarray,
    y_upper_90: np.ndarray,
    y_lower_50: Optional[np.ndarray] = None,
    y_upper_50: Optional[np.ndarray] = None,
    color: str = "#00e676",
    row: int = 1,
    col: int = 1,
) -> go.Figure:
    """
    Dodaje gradientowe przedziały ufności do istniejącego wykresu Plotly.
    Używaj po add_trace() z linią predykcji.

    Parameters
    ----------
    fig             : istniejący go.Figure
    x               : wartości osi X
    y_lower_90/upper_90 : granice 90% CI
    y_lower_50/upper_50 : granice 50% CI (opcjonalne)
    color           : kolor bazowy (hex)
    row, col        : pozycja subplotu
    """
    x_list = list(x)

    # Parsuj kolor hex na składowe
    c = color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)

    # 90% CI
    fig.add_trace(go.Scatter(
        x=x_list + x_list[::-1],
        y=list(y_upper_90) + list(y_lower_90[::-1]),
        fill="toself",
        fillcolor=f"rgba({r},{g},{b},0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="90% CI",
        showlegend=True,
        hoverinfo="skip",
    ), row=row, col=col)

    # 50% CI
    if y_lower_50 is not None and y_upper_50 is not None:
        fig.add_trace(go.Scatter(
            x=x_list + x_list[::-1],
            y=list(y_upper_50) + list(y_lower_50[::-1]),
            fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="50% CI",
            showlegend=True,
            hoverinfo="skip",
        ), row=row, col=col)

    return fig

"""
7_DCC_Dashboard.py — DCC-GARCH Dynamic Correlations Dashboard

Zawiera:
  - Rolling correlation heatmap (ewolucja w czasie)
  - Alert gdy korelacja > 0.80 (kontagion)
  - Pary asset correlation evolution chart
  - Math explainery DCC-GARCH
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.styling import apply_styling, math_explainer

st.set_page_config(page_title="DCC Correlacje", page_icon="🔗", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown("# 🔗 DCC-GARCH Dynamic Correlations")
st.markdown(
    "<p style='color:#6b7280;'>Dynamic Conditional Correlations — Engle (2002). "
    "Korelacje zmieniają się w czasie — klasyczna static corr matrix mija się z celem "
    "podczas krachów (kontagion).</p>",
    unsafe_allow_html=True,
)
st.divider()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parametry DCC")
    tickers_input = st.text_area(
        "Tickery (po średniku)", value="SPY;TLT;GLD;BTC-USD", height=80
    )
    rolling_window = st.slider("Okno rolling correlation (dni)", 20, 120, 60)
    a_dcc  = st.slider("a_dcc — reakcja na szoki", 0.01, 0.20, 0.04, 0.01)
    b_dcc  = st.slider("b_dcc — persistence korelacji", 0.70, 0.99, 0.92, 0.01)
    n_days_sim = st.number_input("Dni symulacji (fallback)", 500, 2000, 750, 50)
    contagion_threshold = st.slider("Próg kontagion alert", 0.50, 0.95, 0.80, 0.05)
    crash_simul = st.checkbox("Symuluj crash (a_dcc ×3 przez 10 dni)", value=False)

tickers = [t.strip() for t in tickers_input.split(";") if t.strip()]
n_assets = len(tickers)

# ─── DATA LOADING ────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def load_multi_returns(ticker_list: list, n: int) -> pd.DataFrame:
    dfs = []
    for t in ticker_list:
        try:
            import yfinance as yf
            data = yf.download(t, period=f"{n//252+2}y", progress=False)["Adj Close"]
            if len(data) > 30:
                dfs.append(data.pct_change().dropna().rename(t))
        except Exception:
            pass
    if not dfs:
        rng = np.random.default_rng(42)
        idx = pd.date_range("2019-01-01", periods=n, freq="B")
        data = {t: rng.standard_normal(n) * 0.012 for t in ticker_list}
        return pd.DataFrame(data, index=idx)
    df = pd.concat(dfs, axis=1).dropna()
    return df

with st.spinner("Ładowanie danych..."):
    returns_df = load_multi_returns(tickers, n_days_sim)

st.markdown(
    f"<div style='color:#6b7280;font-size:12px;'>"
    f"Dataset: {len(returns_df):,} dni | {len(returns_df.columns)} aktywów: "
    f"<b style='color:#00e676'>{', '.join(returns_df.columns.tolist())}</b></div>",
    unsafe_allow_html=True,
)

# ─── DCC-GARCH FIT ───────────────────────────────────────────────────────────
try:
    from modules.dcc_garch import DCCGARCHModel
    dcc = DCCGARCHModel(a_dcc=a_dcc, b_dcc=b_dcc)
    with st.spinner("DCC-GARCH fitting..."):
        dcc.fit(returns_df)
    R_series = dcc.get_conditional_correlations(returns_df)
    dcc_available = True
except Exception as e:
    st.warning(f"DCC-GARCH fit failed: {e} — używam rolling correlation")
    dcc_available = False

# ─── ROLLING CORRELATION (dla heatmap) ────────────────────────────────────
def rolling_mean_corr(df: pd.DataFrame, window: int) -> pd.Series:
    """Średnia korelacja portfela w rolling window."""
    daily_corrs = []
    for i in range(window, len(df)):
        sub = df.iloc[i-window:i]
        corr = sub.corr().values
        # Średnia off-diagonal
        n = corr.shape[0]
        if n < 2:
            daily_corrs.append(np.nan)
            continue
        mask = ~np.eye(n, dtype=bool)
        daily_corrs.append(corr[mask].mean())
    return pd.Series(daily_corrs, index=df.index[window:])

mean_corr_series = rolling_mean_corr(returns_df, rolling_window)

# ─── ROW 1: CONTAGION ALERT ──────────────────────────────────────────────────
current_corr = float(mean_corr_series.iloc[-1]) if len(mean_corr_series) > 0 else 0.0
is_contagion = current_corr > contagion_threshold

if is_contagion:
    st.markdown(f"""
    <div class='alert-badge-red' style='font-size:16px;padding:10px 24px;border-radius:12px;margin-bottom:12px;'>
        ⚠️ KONTAGION ALERT — Srednia korelacja portfela = {current_corr:.2f}
        (próg = {contagion_threshold:.2f}) — DYWERSYFIKACJA ZAWODZI!
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class='alert-badge-green' style='font-size:14px;padding:8px 20px;border-radius:10px;margin-bottom:12px;'>
        ✓ Korelacja OK — {current_corr:.2f} (próg = {contagion_threshold:.2f})
    </div>""", unsafe_allow_html=True)

# ─── ROW 2: METRICS ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
cols_metrics = [c1, c2, c3, c4]
metric_data = [
    ("Śr. korelacja (bieżąca)", f"{current_corr:.3f}", "#ff1744" if is_contagion else "#00e676"),
    ("Max korelacja (hist.)", f"{float(mean_corr_series.max()):.3f}", "#ffea00"),
    ("Min korelacja (hist.)", f"{float(mean_corr_series.min()):.3f}", "#00e676"),
    ("Liczba aktywów", f"{n_assets}", "#00ccff"),
]
for col, (lbl, val, col_v) in zip(cols_metrics, metric_data):
    with col:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-label'>{lbl}</div>
            <div class='metric-value' style='color:{col_v};'>{val}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── ROW 3: ROLLING CORR TIMELINE ────────────────────────────────────────────
col_timeline, col_heatmap = st.columns([2, 1])

with col_timeline:
    st.markdown(f"#### Ewolucja średniej korelacji portfela (okno {rolling_window}d)")
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(
        x=mean_corr_series.index,
        y=mean_corr_series.values,
        fill="tozeroy",
        fillcolor="rgba(0,204,255,0.10)",
        line=dict(color="#00ccff", width=1.8),
        name="Śr. korelacja",
    ))
    fig_mc.add_hline(
        y=contagion_threshold, line_dash="dash", line_color="#ff1744",
        line_width=1.5,
        annotation_text=f"Próg kontagion ({contagion_threshold:.2f})",
        annotation_font_color="#ff1744",
    )
    fig_mc.add_hline(y=0, line_color="#333", line_width=1)
    fig_mc.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.6)",
        height=340, yaxis_title="Korelacja",
        font=dict(color="white", family="Inter"),
        margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    with st.expander("🧮 Jak działa DCC-GARCH?"):
        st.markdown(math_explainer(
            "DCC-GARCH (Engle 2002)",
            "Q_t = (1-a-b)Q̄ + a·z_{t-1}z'_{t-1} + b·Q_{t-1}; R_t = D(Q_t)Q_tD(Q_t)",
            "Q̄ = długookresowa macierz korelacji. Parametr 'a' (reakcja) + 'b' (persistence). "
            "R_t = macierz korelacji warunkowych. W krachu: z_{t-1}z'_{t-1} rośnie → Q_t rośnie → "
            "korelacje rosną (kontagion). Standardowy model nie widzi tej zmiany.",
            "Engle (2002) — Dynamic Conditional Correlations; Engle & Sheppard (2001)",
        ), unsafe_allow_html=True)

with col_heatmap:
    st.markdown("#### Aktualny snapshot korelacji")
    try:
        if dcc_available and len(R_series) > 0:
            R_current = R_series[-1]
        else:
            R_current = returns_df.tail(rolling_window).corr().values

        col_names = returns_df.columns.tolist()
        fig_heat = go.Figure(go.Heatmap(
            z=np.round(R_current, 2),
            x=col_names,
            y=col_names,
            colorscale="RdYlGn_r",
            zmin=-1, zmax=1,
            text=np.round(R_current, 2),
            texttemplate="%{text}",
            textfont=dict(size=11),
            showscale=True,
        ))
        fig_heat.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            height=300,
            font=dict(color="white", family="Inter", size=10),
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    except Exception as e:
        st.error(f"Heatmap error: {e}")

st.divider()

# ─── ROW 4: PAIR CORRELATION EVOLUTION (DCC) ─────────────────────────────────
if dcc_available and n_assets >= 2 and len(R_series) > 0:
    st.markdown("#### DCC Korelacja par aktywów w czasie")
    pair_options = [(i, j) for i in range(n_assets) for j in range(i+1, n_assets)]
    pair_labels  = [f"{returns_df.columns[i]} / {returns_df.columns[j]}" for i, j in pair_options]
    selected_pairs = st.multiselect("Wybierz pary:", pair_labels, default=pair_labels[:min(2, len(pair_labels))])

    fig_pairs = go.Figure()
    colors = ["#00e676", "#00ccff", "#ffea00", "#ff1744", "#a855f7"]
    for idx, (pair_label) in enumerate(selected_pairs):
        pi_idx = pair_labels.index(pair_label)
        i, j = pair_options[pi_idx]
        corr_vals = [R[i, j] for R in R_series]
        fig_pairs.add_trace(go.Scatter(
            x=returns_df.index[:len(corr_vals)],
            y=corr_vals,
            mode="lines",
            line=dict(color=colors[idx % len(colors)], width=1.8),
            name=pair_label,
        ))
    fig_pairs.add_hline(
        y=contagion_threshold, line_dash="dash", line_color="#ff1744",
        line_width=1, annotation_text="Próg kontagion",
    )
    fig_pairs.add_hline(y=0, line_color="#333", line_width=1)
    fig_pairs.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.6)",
        height=320, yaxis_title="Korelacja warunkowa R_t",
        legend=dict(orientation="h", y=-0.15),
        font=dict(color="white", family="Inter"),
        margin=dict(l=50, r=20, t=20, b=60),
    )
    st.plotly_chart(fig_pairs, use_container_width=True)

# ─── ROW 5: DCC SIMULATION (crash scenario) ──────────────────────────────────
if crash_simul and dcc_available:
    st.divider()
    st.markdown("#### 💥 Symulacja krachu — jak DCC wpływa na portfel?")
    crash_days = list(range(50, 65))  # crash po 50 dniach
    try:
        sim_paths = dcc.simulate_paths(
            n_sims=200, n_days=120,
            crash_regime_days=crash_days,
            crash_dcc_multiplier=3.0, seed=42
        )
        # Średni zwrot portfela equal-weight
        w_eq = np.ones(n_assets) / n_assets
        port_rets = sim_paths @ w_eq  # (n_sims, n_days)
        cum_rets  = np.cumprod(1 + port_rets, axis=1)
        p50 = np.percentile(cum_rets, 50, axis=0)
        p05 = np.percentile(cum_rets, 5,  axis=0)
        p95 = np.percentile(cum_rets, 95, axis=0)
        x = np.arange(120)

        fig_crash = go.Figure()
        fig_crash.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([p95, p05[::-1]]),
            fill="toself", fillcolor="rgba(0,230,118,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="5-95 percentyl",
        ))
        fig_crash.add_trace(go.Scatter(x=x, y=p50, line=dict(color="#00e676", width=2), name="Mediana"))
        fig_crash.add_trace(go.Scatter(x=x, y=p05, line=dict(color="#ff1744", width=1, dash="dot"), name="P5 pesymistyczny"))
        for cd in crash_days[:1]:
            fig_crash.add_vline(x=cd, line_color="#ffea00", line_dash="dash",
                                annotation_text="Crash start", annotation_font_color="#ffea00")
        fig_crash.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,11,20,0.6)",
            height=320, yaxis_title="Skumulowany zwrot portfela",
            xaxis_title="Dzień",
            font=dict(color="white", family="Inter"),
            margin=dict(l=50, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_crash, use_container_width=True)
        st.caption("DCC amplifikuje korelacje podczas krachu → portfel traci bardziej niż zakładałby statyczny model")
    except Exception as e:
        st.error(f"Simualcja crash error: {e}")

st.divider()

# ─── ROW 6: ANIMATED DCC HEATMAP [NEW 2024] ──────────────────────────────────
st.markdown("#### 🎞️ Animowana Ewolucja Korelacji DCC (Play/Pause)")
st.caption("Każda klatka = 1 miesiąc. Obserwuj jak korelacje rosną podczas kryzysów.")

try:
    col_names = returns_df.columns.tolist()

    # Build monthly rolling correlation matrices for animation
    monthly_step = 21  # ~1 month
    frames = []
    frame_dates = []

    start_frame = max(rolling_window, monthly_step)
    for end_i in range(start_frame, len(returns_df), monthly_step):
        window_slice = returns_df.iloc[max(0, end_i - rolling_window): end_i]
        if len(window_slice) < 10:
            continue
        if dcc_available and end_i <= len(R_series):
            corr_mat = R_series[min(end_i - 1, len(R_series) - 1)]
        else:
            corr_mat = window_slice.corr().values

        date_label = str(returns_df.index[end_i - 1].date()) if hasattr(returns_df.index[end_i - 1], 'date') else str(end_i)
        frames.append(go.Frame(
            data=[go.Heatmap(
                z=np.round(corr_mat, 2),
                x=col_names, y=col_names,
                colorscale="RdYlGn_r",
                zmin=-1, zmax=1,
                text=np.round(corr_mat, 2),
                texttemplate="%{text}",
                textfont=dict(size=10),
                showscale=True,
            )],
            name=date_label,
            layout=go.Layout(title_text=f"DCC Korelacje — {date_label}"),
        ))
        frame_dates.append(date_label)

    if frames:
        # Initial frame
        init_data = frames[0].data[0]
        fig_anim = go.Figure(
            data=[init_data],
            frames=frames,
            layout=go.Layout(
                title=f"📊 Animowana Macierz Korelacji DCC — {frame_dates[0] if frame_dates else ''}",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=430,
                font=dict(color="white", family="Inter", size=10),
                margin=dict(l=60, r=20, t=55, b=20),
                updatemenus=[{
                    "buttons": [
                        {"args": [None, {"frame": {"duration": 400, "redraw": True},
                                         "fromcurrent": True}],
                         "label": "▶ Play",
                         "method": "animate"},
                        {"args": [[None], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate"}],
                         "label": "⏸ Pause",
                         "method": "animate"},
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 65},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1, "y": 0,
                    "bgcolor": "#1a1a2e",
                    "font": {"color": "white"},
                }],
                sliders=[{
                    "active": 0,
                    "steps": [{"args": [[f.name], {"frame": {"duration": 300, "redraw": True},
                                                    "mode": "immediate"}],
                                "label": f.name,
                                "method": "animate"}
                               for f in frames],
                    "transition": {"duration": 200},
                    "x": 0.1, "len": 0.9, "y": -0.02,
                    "currentvalue": {
                        "font": {"size": 11, "color": "#00ccff"},
                        "prefix": "Data: ",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "bgcolor": "#1a1a2e",
                    "bordercolor": "#333",
                }],
            )
        )
        st.plotly_chart(fig_anim, use_container_width=True, key="dcc_animated")
    else:
        st.info("Za mało danych do animacji (potrzeba > 2 miesięcy).")
except Exception as e:
    st.warning(f"Animacja DCC niedostępna: {e}")

st.divider()

# ─── ROW 7: CORRELATION NETWORK GRAPH [NEW 2024] ─────────────────────────────
st.markdown("#### 🕸️ Sieć Korelacji Aktywów — Network Graph")
st.caption(
    "Grubość krawędzi = siła korelacji. Kolor: 🔴 wysoka (kontagion) / 🟢 niska / niebieski = ujemna. "
    "Brak krawędzi = korelacja < próg filtra."
)
try:
    edge_threshold = st.slider("Min. |korelacja| dla krawędzi", 0.1, 0.9, 0.30, 0.05,
                                key="nw_thresh")
    if dcc_available and len(R_series) > 0:
        corr_mat_nw = R_series[-1]
    else:
        corr_mat_nw = returns_df.tail(rolling_window).corr().values

    n_a = len(col_names)

    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n_a, endpoint=False)
    node_x = np.cos(angles).tolist()
    node_y = np.sin(angles).tolist()

    # Build edge traces
    edge_traces = []
    for i in range(n_a):
        for j in range(i + 1, n_a):
            c = float(corr_mat_nw[i, j])
            if abs(c) < edge_threshold:
                continue
            # Color by sign and magnitude
            if c > 0:
                # Red for contagion, yellow for moderate, green for low
                r_val = min(255, int(255 * c))
                g_val = min(255, int(255 * (1 - c)))
                b_val = 30
            else:
                # Blue for negative (diversification)
                r_val, g_val = 30, 80
                b_val = min(255, int(255 * abs(c)))

            edge_color = f"rgba({r_val},{g_val},{b_val},{min(0.9, abs(c) * 1.2)})"
            edge_width = max(1.0, abs(c) * 6)

            edge_traces.append(go.Scatter(
                x=[node_x[i], node_x[j], None],
                y=[node_y[i], node_y[j], None],
                mode="lines",
                line=dict(width=edge_width, color=edge_color),
                hoverinfo="none",
                showlegend=False,
            ))

    # Node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=col_names,
        textposition="middle center",
        textfont=dict(color="white", size=12, family="Inter"),
        marker=dict(
            size=40,
            color=[float(np.mean(np.abs(corr_mat_nw[i])) - 1 / n_a)
                   for i in range(n_a)],
            colorscale="RdYlGn_r",
            cmin=0.0, cmax=0.8,
            showscale=True,
            colorbar=dict(title="Śr. |Korelacja|",
                          thickness=12, len=0.6,
                          tickfont=dict(color="white")),
            line=dict(width=2, color="white"),
        ),
        hovertext=[
            f"<b>{col_names[i]}</b><br>Śr. |korelacja|: "
            f"{np.mean(np.abs(corr_mat_nw[i])):.3f}"
            for i in range(n_a)
        ],
        hoverinfo="text",
        name="Aktywa",
    )

    fig_net = go.Figure(data=edge_traces + [node_trace], layout=go.Layout(
        title="Sieć Korelacji — Bieżące DCC",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.7)",
        height=480,
        font=dict(color="white", family="Inter"),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=60, b=20),
    ))
    st.plotly_chart(fig_net, use_container_width=True, key="dcc_network")
except Exception as e:
    st.warning(f"Network graph error: {e}")

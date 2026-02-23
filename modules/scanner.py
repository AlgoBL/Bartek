
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import streamlit as st
from modules.metrics import calculate_sharpe, calculate_sortino, calculate_max_drawdown
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
import plotly.graph_objects as go

from scipy.stats import skew, kurtosis, genpareto
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import plotly.figure_factory as ff

def evt_pot_estimator(returns, threshold_quantile=0.90):
    """
    Extreme Value Theory (EVT) - Peaks Over Threshold (POT) approach.
    Fits a Generalized Pareto Distribution (GPD) to the right tail (gains).
    Returns the Shape parameter (xi). Highly positive xi = extremely fat tail (Black Swans).
    """
    if len(returns) < 50:
        return np.nan
        
    positive_returns = returns[returns > 0]
    if len(positive_returns) < 20:
        return np.nan

    # Determine threshold (u) based on quantile
    u = np.quantile(positive_returns, threshold_quantile)
    
    # Extract exceedances
    exceedances = positive_returns[positive_returns > u] - u
    
    if len(exceedances) < 5:
        return np.nan
        
    try:
        # Fit GPD (scipy returns: shape(c), location(loc), scale(scale))
        # Note: scipy's 'c' is the shape parameter xi in EVT standard notation
        xi, loc, scale = genpareto.fit(exceedances, floc=0)
        return xi
    except:
        return np.nan

def calculate_convecity_metrics(ticker, price_series, benchmark_series=None):
    """
    Oblicza zestaw metryk dla Skanera Wypukłości.
    """
    # Obliczamy zwroty logarytmiczne
    returns = np.log(price_series / price_series.shift(1)).dropna()
    
    if len(returns) < 30:
        return None

    # 1. Podstawowe statystyki
    vol_ann = returns.std() * np.sqrt(252)
    mean_ann = returns.mean() * 252
    
    # 2. Wyższe momenty (odrzucamy Gaussianity)
    skew_val = skew(returns)
    kurt_val = kurtosis(returns) # Excess kurtosis (Fisher)
    
    # 3. Professional Metrics
    sharpe = calculate_sharpe(returns)
    sortino = calculate_sortino(returns)
    max_dd = calculate_max_drawdown(price_series)
    
    # 3. Dodanie EVT (Prawy Ogon - Zyski / Black Swans)
    xi_evt = evt_pot_estimator(returns.values)
    
    # 4. Ryzyko Oporu Wariancji (Variance Drag)
    # R_Geom approx R_Arith - 0.5 * sigma^2
    var_drag = 0.5 * (vol_ann ** 2)
    
    # 5. Kelly (Uproszczony dla 0 stopy wolnej od ryzyka, lub hardcoded)
    risk_free = 0.04
    if vol_ann > 0:
        kelly_full = (mean_ann - risk_free) / (vol_ann ** 2)
        # Factor kurczenia (Shrinkage) - Hardcoded 50% safety
        kelly_safe = kelly_full * 0.5
    else:
        kelly_full = 0
        kelly_safe = 0
        
    return {
        "Ticker": ticker,
        "Annual Return": mean_ann,
        "Volatility": vol_ann,
        "Skewness": skew_val,
        "Kurtosis": kurt_val,
        "EVT Shape (Tail)": xi_evt,
        
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max Drawdown": max_dd,
        
        "Variance Drag": var_drag,
        "Kelly Full": kelly_full,
        "Kelly Safe (50%)": kelly_safe
    }

def score_asset(metrics):
    """
    Ocenia aktywo punktowo pod kątem przydatności do strategii Barbell.
    Nagradzamy: Niskie Alpha Hilla, Wysoki Skew, Zmiennosc (jesli skew > 0).
    """
    if metrics is None:
        return -999
        
    score = 0
    
    # 1. EVT Shape (Tail) - szukamy dodatnich grubych ogonów (xi > 0). Im wyżej tym lepiej (asymetryczne ZYSKI)
    xi = metrics["EVT Shape (Tail)"]
    if not np.isnan(xi):
        if xi > 0.3: # Bardzo grube prawe ogony = świetne wypukłe aktywo
            score += 50
        elif xi > 0.1:
            score += 20
        # Brak grubych ogonów nie ma kary, po prostu brak nagrody.
            
    # 2. Skośność (Musi być dodatnia)
    if metrics["Skewness"] > 0:
        score += 30 * metrics["Skewness"] # Promujemy wysoki skew
    else:
        score -= 50 # Dyskwalifikacja ujemnej skośności (ryzyko lewego ogona)
        
    # 3. Kelly (Musi być dodatni - aktywo musi zarabiać)
    if metrics["Kelly Full"] > 0.1:
        score += 20
    elif metrics["Kelly Full"] <= 0:
        score -= 30
        
    return score

def compute_hierarchical_dendrogram(returns_df: pd.DataFrame) -> "go.Figure | None":
    """
    Hierarchical Risk Parity (HRP) Dendrogram visualization (Lopez de Prado 2016).
    Replaces MST with a proper nested clustering tree.
    Uses distance correlation proxy.
    """
    tickers = returns_df.columns.tolist()
    if len(tickers) < 2:
        return None

    # Correlation distance: d_ij = sqrt(0.5*(1-rho))
    corr = returns_df.corr().fillna(0)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
    
    # Condensed distance matrix required by scipy linkage
    condensed_dist = ssd.squareform(dist.values, checks=False)
    
    # Ward linkage for hierarchical clustering
    Z = sch.linkage(condensed_dist, method='ward')
    
    # Create Dendrogram using Plotly Figure Factory
    fig = ff.create_dendrogram(
        dist.values, 
        labels=tickers, 
        linkagefun=lambda x: sch.linkage(x, method='ward'),
        color_threshold=float(np.percentile(Z[:, 2], 70)) # color top 30% of splits differently
    )
    
    fig.update_layout(
        title="🌳 Dendrogram Hierarchiczny (Zagnieżdżona Struktura Ryzyka)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,25,0.9)",
        height=500,
        xaxis_title="Zgrupowane Aktywa",
        yaxis_title="Dystans (Brak Korelacji)",
        font=dict(family="Inter", color="white")
    )
    fig.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
    
    return fig


def compute_correlation_network(returns_df: pd.DataFrame, metrics_df: pd.DataFrame = None) -> "go.Figure | None":
    """
    Minimum Spanning Tree (MST) correlation network visualization.
    Reference: Mantegna (1999) — Hierarchical Structure in Financial Markets.

    Parameters
    ----------
    returns_df : DataFrame of daily returns, columns = tickers
    metrics_df : optional DataFrame with Sharpe column per ticker (for node color)

    Returns
    -------
    Plotly Figure or None if networkx not installed
    """
    if not HAS_NETWORKX:
        return None

    tickers = returns_df.columns.tolist()
    if len(tickers) < 2:
        return None

    # Build correlation-based distance matrix (Mantegna 1999)
    corr = returns_df.corr()
    dist = np.sqrt(2 * (1 - corr))  # metric distance: d = sqrt(2*(1-rho))

    # Build complete graph
    G = nx.Graph()
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i < j:
                G.add_edge(t1, t2, weight=float(dist.loc[t1, t2]),
                           corr=float(corr.loc[t1, t2]))

    # Minimum Spanning Tree
    mst = nx.minimum_spanning_tree(G, weight="weight")

    # Layout
    pos = nx.spring_layout(mst, seed=42, k=2.0)

    # Node colors: by Sharpe if available, else by degree
    if metrics_df is not None and "Sharpe" in metrics_df.columns:
        sharpe_map = metrics_df.set_index("Ticker")["Sharpe"].to_dict() if "Ticker" in metrics_df.columns else {}
        node_colors = [sharpe_map.get(t, 0.0) for t in mst.nodes()]
        color_label = "Sharpe Ratio"
    else:
        node_colors = [dict(mst.degree())[t] for t in mst.nodes()]
        color_label = "Degree"

    # Build Plotly figure
    edge_x, edge_y, edge_text = [], [], []
    for u, v, data in mst.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_text.append(f"{u}-{v}: corr={data['corr']:.2f}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="rgba(150,150,200,0.5)"),
        hoverinfo="none",
        mode="lines",
        name="Korelacja (MST)"
    )

    node_x = [pos[t][0] for t in mst.nodes()]
    node_y = [pos[t][1] for t in mst.nodes()]
    node_labels = list(mst.nodes())

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        textfont=dict(color="white", size=11),
        marker=dict(
            size=20,
            color=node_colors,
            colorscale="RdYlGn",
            colorbar=dict(title=color_label, thickness=12),
            showscale=True,
            line=dict(color="white", width=1.5)
        ),
        hovertemplate=[
            f"<b>{t}</b><br>{color_label}: {c:.2f}<extra></extra>"
            for t, c in zip(node_labels, node_colors)
        ],
        name="Aktywa"
    )

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title="🕸️ Sieć Korelacji MST (Mantegna 1999)",
        showlegend=False,
        hovermode="closest",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,25,0.9)",
        height=500,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        font=dict(family="Inter", color="white"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
    return fig


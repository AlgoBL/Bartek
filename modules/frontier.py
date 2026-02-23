"""
Efficient Frontier Module — Intelligent Barbell
Computes the Efficient Frontier via Monte Carlo portfolio sampling.
Highlights: Max Sharpe, Min Volatility, Max Omega portfolios.
Reference: Markowitz (1952), Shadwick & Keating (2002).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.metrics import calculate_omega


def compute_efficient_frontier(
    returns_df: pd.DataFrame,
    n_portfolios: int = 3000,
    risk_free_rate: float = 0.04,
    barbell_weights: dict = None,
) -> dict:
    """
    Compute the Efficient Frontier by random sampling portfolio weights.

    Parameters
    ----------
    returns_df : DataFrame of daily returns, columns = asset tickers
    n_portfolios : number of random portfolios to sample
    risk_free_rate : annual risk-free rate
    barbell_weights : dict {ticker: weight} of the current Barbell strategy (optional overlay)

    Returns
    -------
    dict with:
        fig : Plotly figure
        portfolios_df : DataFrame of all random portfolios
        max_sharpe : dict
        min_vol : dict
        max_omega : dict
    """
    tickers = returns_df.columns.tolist()
    n_assets = len(tickers)

    if n_assets < 2:
        return {"error": "Potrzeba co najmniej 2 aktywów do obliczeń granicy efektywnej."}

    # Apply 19% Belka Tax to positive returns for realistic net expectations
    returns_taxed = returns_df.copy()
    returns_taxed.mask(returns_taxed > 0, returns_taxed * 0.81, inplace=True)
    
    mean_returns = returns_taxed.mean() * 252
    cov_matrix = returns_taxed.cov() * 252
    rf_taxed = risk_free_rate * 0.81

    results = {
        "Return": [],
        "Volatility": [],
        "Sharpe": [],
        "Omega": [],
        "Weights": [],
    }

    np.random.seed(42)

    for _ in range(n_portfolios):
        # Random weights (Dirichlet ensures they sum to 1)
        weights = np.random.dirichlet(np.ones(n_assets))

        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(weights @ cov_matrix.values @ weights)
        sharpe = (port_return - rf_taxed) / port_vol if port_vol > 0 else 0

        # Omega from taxed daily returns
        port_daily_returns_taxed = returns_taxed.values @ weights
        omega = calculate_omega(port_daily_returns_taxed, threshold=0.0)
        omega = min(omega, 10.0)  # cap for visualization

        results["Return"].append(port_return)
        results["Volatility"].append(port_vol)
        results["Sharpe"].append(sharpe)
        results["Omega"].append(omega)
        results["Weights"].append(weights)

    df = pd.DataFrame(results)

    # Key portfolios
    idx_max_sharpe = df["Sharpe"].idxmax()
    idx_min_vol = df["Volatility"].idxmin()
    idx_max_omega = df["Omega"].idxmax()

    def get_portfolio_info(idx):
        row = df.iloc[idx]
        return {
            "return": row["Return"],
            "volatility": row["Volatility"],
            "sharpe": row["Sharpe"],
            "omega": row["Omega"],
            "weights": {t: w for t, w in zip(tickers, row["Weights"])},
        }

    max_sharpe_port = get_portfolio_info(idx_max_sharpe)
    min_vol_port = get_portfolio_info(idx_min_vol)
    max_omega_port = get_portfolio_info(idx_max_omega)

    # --- Build Plotly Figure ---
    fig = go.Figure()

    # All portfolios — colored by Sharpe
    fig.add_trace(go.Scatter(
        x=df["Volatility"] * 100,
        y=df["Return"] * 100,
        mode="markers",
        marker=dict(
            color=df["Sharpe"],
            colorscale="Viridis",
            size=4,
            opacity=0.6,
            colorbar=dict(title="Sharpe Ratio"),
            showscale=True,
        ),
        text=[
            f"Sharpe: {s:.2f}<br>Omega: {o:.2f}"
            for s, o in zip(df["Sharpe"], df["Omega"])
        ],
        hovertemplate="%{text}<br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
        name="Portfele losowe",
    ))

    # Star markers for key portfolios
    star_configs = [
        (max_sharpe_port, "⭐ Max Sharpe", "gold", "star"),
        (min_vol_port, "🔵 Min Volatility", "cyan", "diamond"),
        (max_omega_port, "🟢 Max Omega", "lime", "triangle-up"),
    ]

    for port, label, color, symbol in star_configs:
        fig.add_trace(go.Scatter(
            x=[port["volatility"] * 100],
            y=[port["return"] * 100],
            mode="markers+text",
            marker=dict(color=color, size=18, symbol=symbol,
                        line=dict(color="white", width=2)),
            text=[label],
            textposition="top right",
            textfont=dict(color=color, size=11),
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Return: {port['return']*100:.1f}%<br>"
                f"Vol: {port['volatility']*100:.1f}%<br>"
                f"Sharpe: {port['sharpe']:.2f}<br>"
                f"Omega: {port['omega']:.2f}<extra></extra>"
            ),
            name=label,
        ))

    # Barbell portfolio overlay
    if barbell_weights:
        w_arr = np.array([barbell_weights.get(t, 0.0) for t in tickers])
        w_arr = w_arr / w_arr.sum() if w_arr.sum() > 0 else w_arr
        bb_ret = float(np.dot(w_arr, mean_returns))
        bb_vol = float(np.sqrt(w_arr @ cov_matrix.values @ w_arr))
        bb_sharpe = (bb_ret - rf_taxed) / bb_vol if bb_vol > 0 else 0
        bb_daily_taxed = returns_taxed.values @ w_arr
        bb_omega = float(min(calculate_omega(bb_daily_taxed), 10.0))

        fig.add_trace(go.Scatter(
            x=[bb_vol * 100],
            y=[bb_ret * 100],
            mode="markers+text",
            marker=dict(color="red", size=22, symbol="x",
                        line=dict(color="white", width=2)),
            text=["🎯 Twój Barbell"],
            textposition="bottom right",
            textfont=dict(color="red", size=12),
            hovertemplate=(
                f"<b>🎯 Twoja Strategia Barbell</b><br>"
                f"Return: {bb_ret*100:.1f}%<br>"
                f"Vol: {bb_vol*100:.1f}%<br>"
                f"Sharpe: {bb_sharpe:.2f}<br>"
                f"Omega: {bb_omega:.2f}<extra></extra>"
            ),
            name="🎯 Twój Barbell",
        ))

    fig.update_layout(
        title="📐 Granica Efektywna — Markowitz (1952)",
        xaxis_title="Ryzyko (Volatility) [%]",
        yaxis_title="Oczekiwana Stopa Zwrotu [%]",
        template="plotly_dark",
        height=550,
        legend=dict(orientation="v", x=1.12, y=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,25,0.9)",
        font=dict(family="Inter", color="white"),
        hovermode="closest"
    )
    fig.update_xaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")
    fig.update_yaxes(showspikes=True, spikecolor="white", spikethickness=1, spikedash="dot", spikemode="across")

    return {
        "fig": fig,
        "portfolios_df": df,
        "max_sharpe": max_sharpe_port,
        "min_vol": min_vol_port,
        "max_omega": max_omega_port,
        "error": None,
    }

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from modules.styling import apply_styling, module_header, scicard
from modules.ai.data_loader import load_data
from modules.herc_optimizer import compute_herc_weights
from modules.ui.widgets import tickers_area
from config import START_DATE

st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown(module_header(
    title="Optimal Portfolio: HERC (A.2)",
    subtitle="Hierarchical Equal Risk Contribution (Raffinot 2018). Klastrowanie + Alokacja Ryzyka.",
    icon="🧬",
    badge="Machine Learning"
), unsafe_allow_html=True)

st.sidebar.markdown("### 1. Ekosystem Aktywów")
default_assets = "SPY, QQQ, TLT, IEF, GLD, GSG, VNQ, BTC-USD, EEM"
assets_input = tickers_area("Koszyk Inwestycyjny", value=default_assets, height=100, parent=st.sidebar)
tickers = [x.strip().upper() for x in assets_input.split(",") if x.strip()]

n_clusters_opt = st.sidebar.selectbox("Liczba Klastrów (Machine Learning)", ["Auto (Heurystyka)", "2", "3", "4", "5", "6"])
n_clust = None if n_clusters_opt.startswith("Auto") else int(n_clusters_opt)

if st.button("🧬 Zoptymalizuj HERC", type="primary"):
    with st.spinner("Pobieranie danych rynkowych i analiza kowariancji..."):
        df_prices = load_data(tickers, start_date=START_DATE)
        
    if df_prices.empty or len(df_prices.columns) < 2:
         st.error("Brak danych lub za mało aktywów do analizy.")
         st.stop()
         
    # 1. Oblicz Zyski i Kowariancje
    returns = df_prices.pct_change().dropna()
    cov_matrix = returns.cov() * 252 # Annualized
    corr_matrix = returns.corr()
    
    # 2. HERC Optimization
    herc_weights = compute_herc_weights(cov_matrix, n_clust)
    
    # 3. Equal Risk Contribution (Naive / Inverse Variance)
    inv_var = 1.0 / np.diag(cov_matrix)
    naive_erc = pd.Series(inv_var / np.sum(inv_var), index=cov_matrix.columns)
    
    # 4. Equal Weights (1/N)
    equal_weights = pd.Series(1.0 / len(tickers), index=cov_matrix.columns)
    
    # === GUI WIZUALIZACJA ===
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Wagi: HERC vs Tradycyjne")
        df_comp = pd.DataFrame({
            "HERC": herc_weights,
            "ERC Naive": naive_erc,
            "1/N (Równe)": equal_weights
        })
        
        # Plotly Bar Chart
        fig_w = go.Figure()
        colors = ['#00e676', '#a855f7', '#3498db']
        
        for i, col in enumerate(df_comp.columns):
            fig_w.add_trace(go.Bar(
                name=col, x=df_comp.index, y=df_comp[col], marker_color=colors[i]
            ))
            
        fig_w.update_layout(
             barmode='group', template='plotly_dark', height=400,
             yaxis_title="Waga Portfela (%)",
             yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig_w, use_container_width=True)
        
    with col2:
        st.subheader("Odległości (Dendrogram Clusteringu)")
        
        # Recreate distance matrix for plotting
        dist_matrix = np.sqrt(np.clip(0.5*(1.0-corr_matrix), 0.0, 1.0))
        condensed_dist = squareform(dist_matrix, checks=False)
        Z = linkage(condensed_dist, method='ward')
        
        fig, ax = plt.subplots(figsize=(8, 4))
        # Custom mpl style to match dark theme
        fig.patch.set_facecolor('#0d0f1a')
        ax.set_facecolor('#0d0f1a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#2a2a3a')
            
        dendrogram(Z, labels=tickers, ax=ax, color_threshold=0, above_threshold_color='#00e676')
        
        st.pyplot(fig)
        st.caption("Drzewo relacji – aktywa obok siebie są silnie skorelowane (tworzą jeden Reżim Ryzyka).")
        
    st.divider()
    
    def render_herc_chart():
        st.plotly_chart(fig_w, use_container_width=True)
        
    scicard(
        title="Hierarchical Equal Risk Contribution (HERC)",
        icon="🧬",
        level0_html=f"<div style='font-size:18px'>Max Skoncentrowane Aktywo (HERC): <span class='neon-cyan'>{herc_weights.idxmax()} ({herc_weights.max():.1%})</span></div>",
        chart_fn=None, # Already rendered above, we keep logic simple
        explanation_md="Algorytm HERC grupuje aktywa w reżimy wg ich korelacji (np. Akcje, Kruszce, Obligacje), a następnie rozdziela budżet ryzyka **równo na klastry**, a nie na wybrane instrumenty. Chroni to portfel przed efektem 'Double Risking' w przypadku dużej ilości skorelowanych instrumentów np. 10 ETF-ów SP500, QQQ, VOO, DIA.",
        formula_code="w_cluster_i = 1 / Risk_i // w_asset_in_cluster = 1 / Vol_asset",
        reference="Raffinot (2018) 'Hierarchical clustering-based asset allocation'"
    )

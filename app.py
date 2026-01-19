import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.styling import apply_styling
from modules.simulation import simulate_barbell_strategy, calculate_metrics

# 1. Page Configuration
st.set_page_config(
    page_title="Barbell Strategy Simulator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Apply Custom Styling
st.markdown(apply_styling(), unsafe_allow_html=True)

# 3. Sidebar Inputs
st.sidebar.title("🛠️ Konfiguracja Strategii")

st.sidebar.markdown("### 1. Kapitał i Czas")
initial_capital = st.sidebar.number_input("Kapitał Początkowy (PLN)", value=100000, step=10000)
years = st.sidebar.slider("Horyzont Inwestycyjny (Lata)", 1, 30, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("### 2. Część Bezpieczna (Safe Sleeve)")
st.sidebar.info("🔒 Obligacje Skarbowe RP 3-letnie (Stałe 5.51%)")
safe_rate = 0.0551 # Fixed as per requirements

st.sidebar.markdown("---")
st.sidebar.markdown("### 3. Część Ryzykowna (Risky Sleeve)")
risky_mean = st.sidebar.slider("Oczekiwany Zwrot Roczny (Średnia)", -0.20, 0.50, 0.08, 0.01, help="Średnia arytmetyczna zwrotu aktywa ryzykownego (np. BTC, ETF Tech)")
risky_vol = st.sidebar.slider("Zmienność Roczna (Volatility)", 0.10, 1.50, 0.50, 0.05, help="Odchylenie standardowe. 0.20 = Akcje, 0.80+ = Altcoiny/Opcje")
risky_kurtosis = st.sidebar.slider("Grubość Ogonów (Kurtosis Parameter)", 2.1, 30.0, 4.0, 0.1, help="Im niższa wartość, tym grubsze ogony (czyli częstsze ekstremalne zyski/straty). Normalny rozkład to ~30 (nieskończoność w teorii). Krypto ~3-4.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 4. Optymalizacja Kelly'ego")
use_kelly = st.sidebar.checkbox("Użyj Kryterium Kelly'ego", help="Automatycznie oblicz alokację w część ryzykowną")

kelly_fraction = 1.0
kelly_shrinkage = 0.0

if use_kelly:
    kelly_fraction = st.sidebar.slider("Ułamek Kelly'ego (Fraction)", 0.1, 1.0, 0.25, 0.05, help="Zalecane: 0.25 (1/4 Kelly) dla bezpieczeństwa.")
    kelly_shrinkage = st.sidebar.slider("Czynnik Kurczenia (Shrinkage)", 0.0, 0.9, 0.1, 0.05, help="Redukcja alokacji ze względu na niepewność parametrów (Baker-McHale).")
    
    # Kelly Calculation
    # f* = (mu - r) / sigma^2
    # Adjusted = f* * Fraction * (1 - Shrinkage)
    if risky_vol > 0:
        kelly_full = (risky_mean - safe_rate) / (risky_vol ** 2)
    else:
        kelly_full = 0
        
    kelly_optimal = kelly_full * kelly_fraction * (1 - kelly_shrinkage)
    kelly_optimal = max(0.0, min(1.0, kelly_optimal)) # Clamp 0-100%
    
    st.sidebar.markdown(f"""
    **Wyniki Kelly'ego:**
    - Pełny Kelly: `{kelly_full:.2%}`
    - Po korektach: `{kelly_optimal:.2%}`
    """)
    
    alloc_safe = 1.0 - kelly_optimal
    st.sidebar.info(f"🔒 Automatyczna Alokacja Bezpieczna: {alloc_safe:.1%}")

else:
    st.sidebar.markdown("### 5. Alokacja Manualna")
    alloc_safe = st.sidebar.slider("Alokacja w Część Bezpieczną (%)", 0, 100, 85) / 100.0

rebalance_strategy = st.sidebar.selectbox(
    "Strategia Rebalansowania",
    ["None (Buy & Hold)", "Yearly", "Monthly", "Threshold (Shannon's Demon)"]
)

threshold_percent = 0.0
if rebalance_strategy == "Threshold (Shannon's Demon)":
    threshold_percent = st.sidebar.slider("Próg Rebalansowania (%)", 5, 50, 20, 5) / 100.0
    st.sidebar.caption(f"Rebalansuj jeśli waga ryzykownej części zmieni się o +/- {int(threshold_percent*100)}% względem celu.")

# 4. Main Content
st.title("⚖️ Barbell Strategy Optimizer")
st.markdown("""
**Strategia Sztangi (Barbell)**: Połączenie ekstremalnego bezpieczeństwa z ekstremalnym ryzykiem. Unikanie "środka".
*Celem jest antykruchość – korzystanie na zmienności (Demon Shannona) przy zachowaniu kapitału.*
""")

if st.button("🚀 Symuluj Wyniki (Monte Carlo)", type="primary"):
    with st.spinner("Przeprowadzanie symulacji 1000 ścieżek..."):
        # Run Simulation
        wealth_paths = simulate_barbell_strategy(
            n_years=years,
            n_simulations=1000,
            initial_captial=initial_capital,
            safe_rate=safe_rate,
            risky_mean=risky_mean,
            risky_vol=risky_vol,
            risky_kurtosis=risky_kurtosis,
            alloc_safe=alloc_safe,
            rebalance_strategy=rebalance_strategy.split(" ")[0], # Take first word
            threshold_percent=threshold_percent
        )
        
        metrics = calculate_metrics(wealth_paths, years)

    # 5. Display Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Średni Kapitał Końcowy", f"{metrics['mean_final_wealth']:,.0f} PLN", delta_color="normal")
    col2.metric("Oczekiwany CAGR", f"{metrics['mean_cagr']:.2%}")
    col3.metric("Mediana CAGR", f"{metrics['median_cagr']:.2%}", help="Bardziej realistyczny wynik dla typowego inwestora")
    col4.metric("Prawdopodobieństwo Straty", f"{metrics['prob_loss']:.1%}", delta_color="inverse")

    # 6. Visualizations
    
    # Path Chart (Cone)
    days = np.arange(wealth_paths.shape[1])
    percentiles = np.percentile(wealth_paths, [5, 50, 95], axis=0)
    
    fig_paths = go.Figure()
    
    # 95th Percentile (Upper Bound)
    fig_paths.add_trace(go.Scatter(
        x=days, y=percentiles[2],
        mode='lines',
        line=dict(width=0),
        name='95th Percentile',
        showlegend=False
    ))
    
    # 5th Percentile (Lower Bound + Fill)
    fig_paths.add_trace(go.Scatter(
        x=days, y=percentiles[0],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0, 255, 136, 0.2)',
        name='95% Confidence Interval'
    ))
    
    # Median
    fig_paths.add_trace(go.Scatter(
        x=days, y=percentiles[1],
        mode='lines',
        line=dict(color='#00ff88', width=3),
        name='Mediana (Typowy Wynik)'
    ))

    fig_paths.update_layout(
        title="Symulacja Ścieżek Bogactwa (95% Przedział Ufności)",
        xaxis_title="Dni Handlowe",
        yaxis_title="Wartość Portfela (PLN)",
        template="plotly_dark",
        height=500,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_paths, use_container_width=True)

    # Histogram of Returns
    final_values = wealth_paths[:, -1]
    fig_hist = px.histogram(
        x=final_values, 
        nbins=50, 
        title="Rozkład Warości Końcowej Portfela",
        color_discrete_sequence=['#00ccff'],
        template="plotly_dark"
    )
    fig_hist.update_layout(xaxis_title="Wartość Końcowa (PLN)", yaxis_title="Liczba Symulacji")
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # 7. Analysis & Explanations (Expanders)
    col_left, col_right = st.columns(2)
    
    with col_left:
        with st.expander("📉 Analiza Ryzyka (Drawdown)"):
            st.write(f"**Średnie Maksymalne Obsunięcie (Max Drawdown):** {metrics['mean_max_drawdown']:.2%}")
            st.write(f"**Najgorszy scenariusz (Worst Case):** {metrics['worst_case_drawdown']:.2%}")
            st.caption("Dzięki dużej alokacji w Obligacje (Safe Sleeve), obsunięcia są drastycznie zredukowane, nawet przy krachu aktywa ryzykownego.")

    with col_right:
        with st.expander("🧠 Teoria: Demon Shannona i Kelly"):
            st.markdown("""
            **Demon Shannona (Volatility Harvesting)**:
            Jeśli aktywo jest bardzo zmienne (rośnie/spada), regularne rebalansowanie (sprzedawanie wzrostów, kupowanie spadków) pozwala generować dodatni zwrot nawet jeśli samo aktywo netto nie rośnie (średnia geometryczna = 0).
            
            **Kryterium Kelly'ego**:
            Wskazuje optymalną wielkość zakładu. W praktyce (Taleb) stosuje się ułamek Kelly'ego (np. 1/2 lub 1/4), aby uniknąć ryzyka ruiny przy błędnych szacunkach prawdopodobieństwa.
            """)

else:
    st.info("👈 Ustaw parametry w lewym panelu i kliknij 'Symuluj Wyniki', aby zobaczyć projekcję.")

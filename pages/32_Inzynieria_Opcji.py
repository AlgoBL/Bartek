import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.real_options import black_scholes_merton, binomial_tree_real_option, merton_structural_credit_risk

st.set_page_config(page_title="Inżynieria Opcji", page_icon="📈", layout="wide")

st.title("📈 Inżynieria Finansowa i Opcje Realne")
st.markdown("""
Moduł wykorzystujący stochastyczną matematykę finansową do wyceny asymetrycznego ryzyka.
Znajdziesz tu model **Blacka-Scholesa-Mertona**, modelowanie **Opcji Realnych** (w biznesie) oraz strukturalny model **Mertona** wyceny ryzyka bankructwa.
""")

tabs = st.tabs([
    "🎯 Black-Scholes & Greki", 
    "🌳 Opcje Realne (Drzewa)", 
    "💥 Ryzyko Kredytowe (Merton)"
])

# --- Tab 1: Black-Scholes ---
with tabs[0]:
    st.header("1. Model Blacka-Scholesa-Mertona i Greki")
    
    with st.expander("📖 Co to jest i jak to działa?"):
        st.markdown("""
        **Black-Scholes-Merton (1973)** to fundament inżynierii finansowej (Nagroda Nobla 1997).
        Pozwala precyzyjnie wycenić europejskie opcje Call (Kupna) i Put (Sprzedaży).
        
        Kluczową potęgą modelu są **Greki** – miary wrażliwości:
        *   **Delta ($\Delta$)**: O ile zmieni się cena opcji, gdy cena aktywa wzrośnie o $1. (Miara kierunkowa)
        *   **Gamma ($\Gamma$)**: Jak szybko rośnie Delta. (Miernik wypukłości i ryzyka nagłych skoków)
        *   **Vega ($\nu$)**: Jak rośnie wartość opcji, gdy wzrasta strach na rynku (zmienność $\sigma$).
        *   **Theta ($\Theta$)**: Koszt upływu czasu (Time decay). Ile tracisz dziennie na samym trzymaniu opcji.
        """)
        
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parametry Rynku")
        S = st.number_input("Cena aktywa bazowego (S)", value=100.0)
        K = st.number_input("Cena wykonania (Strike K)", value=100.0)
        T = st.slider("Czas do wygaśnięcia (T w latach)", 0.01, 5.0, 1.0)
        sigma = st.slider("Zmienność implikowana (Sigma %)", 1, 150, 20) / 100.0
        r = st.slider("Stopa wolna od ryzyka (r %)", -2.0, 15.0, 5.0) / 100.0
        q = st.slider("Stopa dywidendy (q %)", 0.0, 10.0, 0.0) / 100.0
        
    with col2:
        res_call = black_scholes_merton(S, K, T, r, sigma, q, "call")
        res_put = black_scholes_merton(S, K, T, r, sigma, q, "put")
        
        col_c, col_p = st.columns(2)
        with col_c:
            st.metric("Cena Opcji CALL", f"${res_call['price']:,.2f}")
            st.markdown(f"""
            * **Delta:** {res_call['delta']:.3f}
            * **Gamma:** {res_call['gamma']:.4f}
            * **Vega:** {res_call['vega']:.3f}
            * **Theta (1 dzień):** {res_call['theta']:.3f}
            """)
        with col_p:
            st.metric("Cena Opcji PUT", f"${res_put['price']:,.2f}")
            st.markdown(f"""
            * **Delta:** {res_put['delta']:.3f}
            * **Gamma:** {res_put['gamma']:.4f}
            * **Vega:** {res_put['vega']:.3f}
            * **Theta (1 dzień):** {res_put['theta']:.3f}
            """)
            
        # Wykres profilu wypłaty
        s_range = np.linspace(S * 0.5, S * 1.5, 100)
        prices_call = [black_scholes_merton(s_val, K, T, r, sigma, q, "call")["price"] for s_val in s_range]
        prices_put = [black_scholes_merton(s_val, K, T, r, sigma, q, "put")["price"] for s_val in s_range]
        
        fig_bs = go.Figure()
        fig_bs.add_trace(go.Scatter(x=s_range, y=prices_call, name="CALL Price", line=dict(color="#00CC96")))
        fig_bs.add_trace(go.Scatter(x=s_range, y=prices_put, name="PUT Price", line=dict(color="#EF553B")))
        fig_bs.add_vline(x=S, line_dash="dash", line_color="white", annotation_text="Obecna Cena")
        fig_bs.update_layout(title="Wycena Black-Scholes od Ceny Aktywa", xaxis_title="Cena Aktywa (S)", yaxis_title="Cena Opcji", template="plotly_dark")
        st.plotly_chart(fig_bs, use_container_width=True)

# --- Tab 2: Opcje Realne ---
with tabs[1]:
    st.header("2. Opcje Realne (Wycena Projektów Biznesowych)")
    
    with st.expander("📖 Co to jest i jak to działa?"):
        st.markdown("""
        **Opcje Realne (ROA)** tłumaczą opcje finansowe na decyzje życiowe/biznesowe.
        Standardowe NPV (Net Present Value) zakłada sztywną ścieżkę projektu. Opcje realne wyceniają **wartość elastyczności**.
        
        Przykłady:
        *   **Opcja Opóźnienia (Call):** Masz patent. Możesz zainwestować dzisiaj lub poczekać rok na lepsze warunki rynkowe. 
        *   **Opcja Ekspansji (Call):** Otwierasz jedną restaurację z prawem pierwokupu sąsiednich lokali, jeśli odniesiesz sukces.
        *   **Opcja Porzucenia (Put):** Prawo do zamknięcia kopalni, jeśli ceny złota spadną poniżej kosztów wydobycia.
        """)
        
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("Parametry Projektu")
        type_opt = st.selectbox("Typ Opcji", ["Opcja Ekspansji (CALL)", "Opcja Porzucenia (PUT)"])
        S_real = st.number_input("Wartość Oczekiwana Projektu (V)", value=500000)
        K_real = st.number_input("Koszt Inwestycji / Utrzymania (I)", value=500000)
        T_real = st.slider("Horyzont czasowy decyzji (Lata)", 1, 10, 5)
        sigma_real = st.slider("Niepewność Rynkowa (Zmienność %)", 10, 100, 30) / 100.0
        r_real = st.slider("Stopa dyskontowa (Risk-free %)", 0.0, 15.0, 5.0, key="r_real") / 100.0
        
    with c2:
        t_opt = "call" if "CALL" in type_opt else "put"
        
        # Wycena klasycznego NPV
        npv_classic = S_real - K_real if t_opt == "call" else K_real - S_real
        
        # Wycena opcji z drzewa
        real_option_val = binomial_tree_real_option(S_real, K_real, T_real, r_real, sigma_real, 100, t_opt)
        
        st.subheader("Porównanie Wycen")
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("Klasyczne NPV (Zrób to teraz)", f"${npv_classic:,.2f}")
        with col_r2:
            st.metric("Rozszerzone NPV (Z opcją elastyczności)", f"${real_option_val:,.2f}", delta=f"+${max(0, real_option_val - npv_classic):,.2f} Premium za elastyczność")
            
        if real_option_val > max(0, npv_classic):
            st.success("Wysoka zmienność rynkowa sprawia, że **czekanie z decyzją ma ogromną wartość**. Elastyczność ratuje projekt!")
        else:
            st.info("Opcja elastyczności nie wnosi dodatkowej wartości. Wykonaj projekt natychmiast lub odrzuć go definitywnie.")
            
        st.info("Model używa Amerykańskiego Drzewa Dwumianowego (Cox-Ross-Rubinstein) o 100 krokach, pozwalając na wczesne wykonanie decyzji w dowolnym momencie.")

# --- Tab 3: Model Mertona ---
with tabs[2]:
    st.header("3. Model Mertona (Prawdopodobieństwo Bankructwa)")
    
    with st.expander("📖 Co to jest i jak to działa?"):
        st.markdown("""
        **Strukturalny Model Mertona (1974)** wycenia ryzyko kredytowe firmy.
        Robert Merton genialnie zauważył, że **Kapitał Własny (Akcje)** to nic innego jak opcja CALL na aktywa firmy, gdzie ceną wykonania jest wartość całkowitego zadłużenia!
        
        Jeżeli przyjdzie termin spłaty długu ($T$), a wartość firmy ($V$) jest większa od długu ($D$), akcjonariusze spłacają dług i zatrzymują resztę. 
        Jeśli $V < D$, akcjonariusze nie spłacają długu (Opcja wygasa bez wartości), oddają firmę wierzycielom (Bankructwo).
        
        Na podstawie tego model potrafi matematycznie wyliczyć **Probability of Default (PD)** – szansę, że firma upadnie.
        """)
        
    m1, m2 = st.columns([1, 2])
    
    with m1:
        st.subheader("Bilans Firmy")
        v_merton = st.number_input("Rynkowa Wartość Aktywów (V w mln $)", value=100.0)
        d_merton = st.number_input("Całkowite Zadłużenie (D w mln $)", value=80.0)
        sigma_v = st.slider("Zmienność Aktywów (Vol %)", 5, 100, 25) / 100.0
        t_merton = st.slider("Średni czas zapadalności długu (Lata)", 1, 10, 1)
        r_merton = st.slider("Stopa wolna od ryzyka (Merton %)", 0.0, 10.0, 5.0, key="r_mert") / 100.0
        
    with m2:
        mert = merton_structural_credit_risk(v_merton, d_merton, t_merton, r_merton, sigma_v)
        
        st.subheader("Wyniki Modelu Kredytowego")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Odległość do Bankructwa (Distance to Default)", f"{mert['distance_to_default']:.2f} σ")
            pd_color = "normal" if mert['probability_of_default'] < 0.05 else "inverse"
            st.metric("Prawdopodobieństwo Bankructwa (PD)", f"{mert['probability_of_default']*100:.2f}%", delta_color=pd_color)
        with col_m2:
            st.metric("Kapitalizacja Akcji (Opcja Call na V)", f"${mert['equity_value']:.2f} mln")
            spread_str = f"{mert['credit_spread_bps']:.0f} bps" if mert['credit_spread_bps'] != float('inf') else "DEFAULT"
            st.metric("Wymagany Spread Kredytowy (Ryzyko)", spread_str)
            
        # Gauge chart for PD
        fig_pd = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = mert['probability_of_default'] * 100,
            title = {'text': "Prawdopodobieństwo Bankructwa (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 5], 'color': "green"},
                    {'range': [5, 20], 'color': "orange"},
                    {'range': [20, 100], 'color': "red"}],
            }
        ))
        fig_pd.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_pd, use_container_width=True)

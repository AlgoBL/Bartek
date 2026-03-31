"""23_Walk_Forward.py — Probability of Backtest Overfitting & CPCV"""
import streamlit as st
import pandas as pd
from modules.styling import apply_styling
from modules.ai.data_loader import load_data
from modules.walk_forward import generate_strategy_matrix, cpcv_pbo, plot_cpcv_results, adversarial_validation_auc
from modules.i18n import t

st.set_page_config(page_title="PBO Scorecard", page_icon="🔬", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown("# 🔬 Walk-Forward & PBO Scorecard")
st.markdown("*Combinatorial Purged Cross-Validation i Probability of Backtest Overfitting wg Bailey et al. (2014).*")
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Konfiguracja Testu")
    risky_asset = st.text_input("Ryzykowne aktywo (np. SPY)", "SPY")
    safe_asset = st.text_input("Bezpieczne aktywo (np. TLT)", "TLT")
    period = st.selectbox("Okres danych", ["10y", "15y", "20y", "max"], index=1)
    n_chunks = st.slider("Liczba bloków CPCV (N)", 4, 10, 6, 2, help="Im więcej bloków, tym więcej kombinacji. Typowo używa się N=6 (15 kombinacji IS/OOS) lub N=8 (70 kombinacji).")
    n_strats = st.slider("Liczba wariantów strategii (M)", 10, 100, 20, 10, help="Rozmiar przestrzeni parametrów (im większa, tym ryzyko Overfittingu większe).")

if st.button("🚀 Przeprowadź Test CPCV (Probability of Backtest Overfitting)", type="primary"):
    with st.spinner("Pobieranie danych rynkowych i symulacja wariantów (M wariantów na wektorze cen)..."):
        try:
            df = load_data([risky_asset, safe_asset], period=period)
        except Exception as e:
            st.error(f"Błąd api danych: {e}")
            st.stop()
            
        if df.empty or risky_asset not in df.columns or safe_asset not in df.columns:
            st.error("Błąd pobierania danych. Upewnij się, że wpisane tickery istnieją na Yahoo Finance.")
            st.stop()
            
        r_risky = df[risky_asset].pct_change().dropna()
        r_safe  = df[safe_asset].pct_change().dropna()
        
        # Wyrównanie indeksów dla aktywów o różnych datach notowań
        idx = r_risky.index.intersection(r_safe.index)
        r_risky = r_risky.loc[idx]
        r_safe = r_safe.loc[idx]
        
        if len(idx) < 252 * 5:
            st.warning("⚠️ Ostrzeżenie: Okres danych mniejszy niż 5 lat. Wyniki CPCV mogą być statystycznie niewiarygodne.")
            
        strat_matrix = generate_strategy_matrix(r_risky, r_safe, n_strategies=n_strats)
        
    with st.spinner("Wyliczanie ścieżek Combinatorial Purged Cross-Validation i ocena przetrwania In-Sample do Out-Of-Sample..."):
        pbo, rankits = cpcv_pbo(strat_matrix, n_chunks=n_chunks)
        fig = plot_cpcv_results(rankits, pbo)
        
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("### 📊 Wyniki Testu Przeuczenia (PBO)")
            pbo_val = pbo * 100
            
            # Kolorowanie
            if pbo_val < 10:
                color = "#00e676"  # Zielony (dobry)
                status = "✅ SOLIDNA (Brak data snooping)"
            elif pbo_val < 30:
                color = "#ffea00"  # Żółty (ostrzeżenie)
                status = "⚠️ UMIARKOWANE RYZYKO (Sprawdź założenia)"
            else:
                color = "#ff1744"  # Czerwony (źle)
                status = "❌ PRZEUCZONA (Zbyt dopasowana do historii)"
                
            st.markdown(f"**PBO (Probability of Backtest Overfitting):** <span style='font-size:32px;color:{color};font-weight:bold;'>{pbo_val:.1f}%</span>", unsafe_allow_html=True)
            st.markdown(f"**Ocena:** <span style='color:{color};font-weight:bold;'>{status}</span>", unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("""
            **Matematyka PBO (Bailey & Lopez de Prado 2014):**
            PBO definiuje prawdopodobieństwo, z jakim strategia zoptymalizowana In-Sample (IS), w próbce Out-Of-Sample (OOS) okaże się gorsza od *losowo wybranej innej (mediany)*.
            
            **Zasada działania testu:**
            - **CPCV** (Combinatorial Purged CV) wycina historię na **N** bloków.
            - Tworzy wszystkie unikalne podziały na zbiory uczące (In-Sample) złożone z **N/2** bloków. (Dla N=6 test wykonuje się 15 razy).
            - Na każdym wariancie szuka optymalnej alokacji ze zbioru M strategii, i testuje ją na wyciętych blokach testowych (OOS).
            - Odkłada rangę wyniku OOS (w rozkładzie wszystkich wyników). Histogram obok pokazuje zlogarytmowane dystrybucje tych rang (Logit). PBO to pole powierzchni pod medianą wyliczonych logitów rang!
            """)

        # --- ADVERSARIAL VALIDATION SECTION ---
        st.markdown("---")
        st.markdown("### 🕵️ Adversarial Validation (Wykrywanie Leakage)")
        
        # Prepare 1D data for the classifier (using returns from risky asset)
        train_len = len(r_risky) // 2
        train_rets = r_risky.values[:train_len].reshape(-1, 1)
        test_rets = r_risky.values[train_len:].reshape(-1, 1)
        
        auc_val = adversarial_validation_auc(train_rets, test_rets)
        
        ac1, ac2 = st.columns([1.2, 1])
        with ac1:
            import numpy as np
            import plotly.graph_objects as go
            
            # Simple gauge for AUC
            fig_auc = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = auc_val,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "AUC: Podobieństwo zbiorów Train vs Test", 'font': {'size': 18}},
                gauge = {
                    'axis': {'range': [0.4, 1.0], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#00ff88" if auc_val < 0.6 else "#ffea00" if auc_val < 0.75 else "#ff1744"},
                    'steps': [
                        {'range': [0.4, 0.6], 'color': 'rgba(0, 255, 136, 0.1)'},
                        {'range': [0.6, 0.75], 'color': 'rgba(255, 234, 0, 0.1)'},
                        {'range': [0.75, 1.0], 'color': 'rgba(255, 23, 68, 0.1)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.70
                    }
                }
            ))
            fig_auc.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Inter"}, height=300)
            st.plotly_chart(fig_auc, use_container_width=True)
            
        with ac2:
            st.markdown("#### Interpretacja AUC Klasyfikatora")
            if auc_val < 0.6:
                st.success("✅ **Niskie AUC:** Klasyfikator nie potrafi rozróżnić Train od Test. Dane są statystycznie spójne (brak istotnego dryfu).")
            elif auc_val < 0.75:
                 st.warning("⚠️ **Podwyższone AUC:** Wystąpił dryf danych lub zmiana reżimu. Twoja strategia może działać inaczej w przyszłości niż w przeszłości.")
            else:
                 st.error("🚨 **Wysokie AUC:** Dane testowe radykalnie różnią się od treningowych. Prawdopodobny wyciek danych (Data Leakage) lub ekstremalny Regime Shift.")
            
            st.info("""
            **Adversarial Validation** to metoda, w której trenujemy model AI (np. Regresję Logistyczną), by odgadł, czy dany wiersz pochodzi z przeszłości czy z przyszłości.
            Jeśli model zgaduje to bezbłędnie (AUC -> 1.0), oznacza to, że przeszłość NIE jest reprezentatywna dla przyszłości.
            """)

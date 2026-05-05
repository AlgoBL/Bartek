import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from modules.contract_theory import nash_bargaining, rubinstein_bargaining, principal_agent_contract
from modules.styling import apply_styling, module_header
from modules.i18n import t

# 2. Apply Custom Styling
st.markdown(apply_styling(), unsafe_allow_html=True)

# 3. Main Navigation Header
st.markdown(module_header(
    title="Negocjacje i Kontrakty",
    subtitle="Asymetria informacji, problem Pryncypała-Agenta oraz modele przetargowe Nasha i Rubinsteina.",
    icon="🤝",
    badge="Ekonomia Kontraktów"
), unsafe_allow_html=True)

tabs = st.tabs([
    "🤝 Model Przetargowy Nasha", 
    "⏳ Model Rubinsteina (Czas to Pieniądz)", 
    "🕵️ Problem Pryncypała-Agenta"
])

# --- Tab 1: Nash Bargaining ---
with tabs[0]:
    st.header("1. Model Przetargowy Nasha (Nash Bargaining)")
    
    with st.expander("📖 Co to jest i jak to działa?"):
        st.markdown("""
        **Zasada działania:**
        Model odpowiada na pytanie: "Jak sprawiedliwie podzielić tort, gdy obie strony mają opcję alternatywną (BATNA)?"
        **BATNA** (Best Alternative to a Negotiated Agreement) to punkt groźby. Jeśli nie dogadasz się z partnerem, to z czym zostajesz?
        
        Złota zasada Nasha mówi, że w pierwszej kolejności każdy otrzymuje swoją BATNA, a reszta (tzw. Nadwyżka z kooperacji) jest dzielona **dokładnie na pół**. Oznacza to, że im silniejsza Twoja alternatywa, tym więcej zyskasz z samego faktu zasiadania do stołu.
        """)
        
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Parametry Negocjacji")
        total_val = st.number_input("Wartość kontraktu / Tort do podziału ($)", value=100000)
        batna_a = st.number_input("Twoja BATNA ($)", value=20000)
        batna_b = st.number_input("BATNA Partnera ($)", value=40000)
        
    with col2:
        res = nash_bargaining(batna_a, batna_b, total_val)
        
        if res["status"] == "Brak porozumienia":
            st.error(f"Negocjacje zerwane! Suma Waszych żądań (BATNA) wynosi {batna_a + batna_b}, a tort ma tylko {total_val}. Nie ma nadwyżki do podziału.")
        else:
            st.success(f"Porozumienie osiągnięte! Nadwyżka do podziału wynosi **${res['surplus']:,.2f}**.")
            
            # Waterfall chart for visualization
            fig_nash = go.Figure(go.Waterfall(
                name="20", orientation="v",
                measure=["relative", "relative", "relative", "relative", "total"],
                x=["Twoja BATNA", "BATNA Partnera", "Twoja premia", "Premia Partnera", "Wartość Całkowita"],
                textposition="outside",
                text=[f"${batna_a/1000}k", f"${batna_b/1000}k", f"${(res['surplus']/2)/1000}k", f"${(res['surplus']/2)/1000}k", f"${total_val/1000}k"],
                y=[batna_a, batna_b, res['surplus']/2, res['surplus']/2, 0],
                connector={"line":{"color":"rgb(63, 63, 63)"}}
            ))
            
            fig_nash.update_layout(
                title="Wizualizacja Podziału Wartości (Nash Bargaining)",
                template="plotly_dark",
                showlegend=False
            )
            st.plotly_chart(fig_nash, use_container_width=True)
            
            st.markdown(f"**Twój końcowy zysk:** ${res['share_a']:,.2f} (BATNA + Połowa nadwyżki)")
            st.markdown(f"**Zysk partnera:** ${res['share_b']:,.2f} (BATNA + Połowa nadwyżki)")

# --- Tab 2: Rubinstein ---
with tabs[1]:
    st.header("2. Model Rubinsteina (Naprzemienne Oferty)")
    
    with st.expander("📖 Co to jest i jak to działa?"):
        st.markdown("""
        **Zasada działania:**
        W rzeczywistości negocjacje to proces w czasie. A czas kosztuje.
        Współczynnik dyskontowy (Cierpliwość $\delta$) od 0 do 1 określa, jak bardzo tracisz na odkładaniu porozumienia w czasie (1 = w ogóle nie tracisz, 0 = tracisz wszystko natychmiast).
        
        **Kluczowe wnioski:**
        1. **Zaletą pierwszego ruchu:** Gracz, który składa pierwszą ofertę, zawsze ma drobną przewagę (tzw. First Mover Advantage).
        2. **Cierpliwość popłaca:** Ten, kto ma wyższy współczynnik dyskonta (jest bardziej cierpliwy / ma więcej gotówki na przeczekanie), zgarnia większą część tortu!
        """)
        
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("Cierpliwość Stron")
        val_r = st.number_input("Wartość przedmiotu do podziału ($)", value=100000, key="val_r")
        delta_a = st.slider("Twoja cierpliwość (Gracz A - składa ofertę)", 0.01, 1.0, 0.9)
        delta_b = st.slider("Cierpliwość Przeciwnika (Gracz B)", 0.01, 1.0, 0.8)
        
    with c2:
        res_r = rubinstein_bargaining(val_r, delta_a, delta_b)
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Twój Zysk (Gracz A)', 'Zysk Przeciwnika (Gracz B)'], 
            values=[res_r["share_a"], res_r["share_b"]],
            hole=.4,
            marker=dict(colors=['#00CC96', '#EF553B'])
        )])
        
        fig_pie.update_layout(title="Wynik Negocjacji w Modelu Rubinsteina", template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.success(f"Dzięki swojej cierpliwości ({delta_a}) oraz przewadze pierwszego ruchu, zgarniasz **{res_r['share_a']/val_r*100:.1f}%** tortu (${res_r['share_a']:,.2f}).")

# --- Tab 3: Principal-Agent ---
with tabs[2]:
    st.header("3. Problem Pryncypała-Agenta (Moral Hazard)")
    
    with st.expander("📖 Co to jest i jak to działa?"):
        st.markdown("""
        **Zasada działania:**
        Wcielasz się w rolę Pracodawcy (Pryncypała). Zatrudniasz Pracownika (Agenta), którego wysiłku **nie możesz w pełni kontrolować**.
        Zamiast płacić z góry, konstruujesz **Kontrakt Motywacyjny** złożony z dwóch pensji: $W_H$ (Premia za sukces) i $W_L$ (Podstawa, czasem karna za porażkę).
        
        Aby Agentowi opłacało się starać, musisz spełnić warunek zachęty (Incentive Compatibility).
        Ale uwaga: dawanie potężnych premii kosztuje! Algorytm sprawdza, czy w ogóle opłaca Ci się motywować pracownika, czy może lepiej zapłacić mu minimum socjalne i zaakceptować mniejszą szansę na sukces biznesu.
        """)
        
    pc1, pc2 = st.columns([1, 2])
    
    with pc1:
        st.subheader("Środowisko Biznesowe")
        r_succ = st.number_input("Twój Zysk w przypadku Sukcesu ($)", value=200000)
        r_fail = st.number_input("Twój Zysk w przypadku Porażki ($)", value=50000)
        
        st.subheader("Profil Pracownika (Agenta)")
        p_eff = st.slider("Szansa sukcesu, gdy Agent się STARA (%)", 1, 100, 80) / 100.0
        p_no_eff = st.slider("Szansa sukcesu, gdy Agent się Leni (%)", 1, 100, 40) / 100.0
        c_eff = st.number_input("Koszt wysiłku Agenta ($)", value=10000)
        u_res = st.number_input("Opcja rezerwowa Agenta / Zasiłek ($)", value=30000)
        
    with pc2:
        res_pa = principal_agent_contract(r_succ, r_fail, c_eff, p_eff, p_no_eff, u_res)
        
        if "error" in res_pa:
            st.error(res_pa["error"])
        else:
            st.subheader("Rekomendacja Architektury Kontraktu")
            
            if res_pa["profit_with_effort"] >= res_pa["profit_no_effort"]:
                st.success(f"**{res_pa['recommendation']}** - Różnica prawdopodobieństwa jest warta zachodu.")
            else:
                st.warning(f"**{res_pa['recommendation']}** - Koszt premii zjada zyski! Lepiej zaakceptować lenistwo Agenta.")
                
            col_k1, col_k2 = st.columns(2)
            with col_k1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">PŁACA ZA SUKCES (W_H)</div>
                    <div class="metric-value" style="color:var(--green)">${res_pa['w_H']:,.0f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">PŁACA ZA PORAŻKĘ (W_L)</div>
                    <div class="metric-value" style="color:var(--red)">${res_pa['w_L']:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_k2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">ZYSK (WYSIŁEK)</div>
                    <div class="metric-value" style="color:var(--cyan)">${res_pa['profit_with_effort']:,.0f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">ZYSK (LENISTWO)</div>
                    <div class="metric-value" style="color:var(--text-dim)">${res_pa['profit_no_effort']:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
                
            fig_bar = go.Figure(data=[
                go.Bar(name='Twój Zysk Netto', x=['Z Kontraktem (Wysiłek)', 'Bez Kontraktu (Lenistwo)'], y=[res_pa['profit_with_effort'], res_pa['profit_no_effort']], marker_color='#636EFA')
            ])
            fig_bar.update_layout(title="Porównanie Strategii HR", template="plotly_dark", yaxis_title="Oczekiwany Zysk ($)")
            st.plotly_chart(fig_bar, use_container_width=True)

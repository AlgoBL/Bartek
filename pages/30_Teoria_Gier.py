import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import odeint
from modules.styling import apply_styling, module_header
from modules.i18n import t
from modules.game_theory_engine import (
    find_pure_nash, find_all_nash,
    find_correlated_equilibrium,
    gale_shapley, check_stability,
    solve_bne_two_types,
    stackelberg_cournot,
    lq_mean_field_game,
)

# 2. Apply Custom Styling
st.markdown(apply_styling(), unsafe_allow_html=True)

# 3. Main Navigation Header
st.markdown(module_header(
    title="Teoria Gier",
    subtitle="Optymalizacja strategii rynkowych: gry macierzowe, ewolucyjne oraz teoria aukcji.",
    icon="♟️",
    badge="Strategia Interakcyjna"
), unsafe_allow_html=True)

tabs = st.tabs([
    "⚔️ Gry Macierzowe (Dylematy)",
    "🎲 Strategie Mieszane",
    "🌳 Gry Sekwencyjne",
    "🔨 Teoria Aukcji",
    "🦅 Dynamika Ewolucyjna",
    "⚙️ Mechanism Design",
    "🔄 Gry Powtarzalne",
    "🤝 Matching Theory",
    "🧩 Solver NxM (Ogólny Nash)",
    "📡 Równowaga Korelatywna",
    "🔮 Bayesowska RN (BNE)",
    "👑 Równowaga Stackelberga",
    "🌊 Mean-Field Games",
])

# --- Tab 1: Gry Macierzowe ---
# (Previous content unchanged up to line 418)
with tabs[0]:
    st.header("1. Gry Macierzowe i Równowaga Nasha")
    st.markdown("Analiza 2-osobowych gier strategicznych. Gracz Wierszowy (Ty) wybiera wiersz, Gracz Kolumnowy (Przeciwnik) wybiera kolumnę.")
    
    with st.expander("📖 Jak z tego korzystać i co to oznacza?"):
        st.markdown("""
        **Co tu się dzieje?**
        Modelujesz sytuację dwóch firm/inwestorów. 
        **Równowaga Nasha** to sytuacja, w której *żaden z graczy nie ma interesu, by w pojedynkę zmienić swoją decyzję*, jeśli drugi jej nie zmienia. To "magnetyczny" punkt przyciągający rynek.
        
        **Wbudowane szablony:**
        - **Dylemat Więźnia (Kartel):** Obu firmom opłaca się trzymać wysokie ceny (Współpraca), ale pokusa zdrady (obniżenia ceny) jest tak silna, że obie firmy w końcu zdradzają, lądując w najgorszym scenariuszu.
        - **Tchórz (Wojna Cenowa):** Zderzenie czołowe. Nikt nie chce ustąpić, bo ustąpienie oznacza stratę, ale zderzenie to bankructwo. Równowaga Nasha nakazuje ustąpić, jeśli drugi nie ustępuje.
        - **Polowanie na Jelenia (Współpraca Ryzyka):** Obie firmy mogą zyskać dużo (Jeleń), ale wymaga to 100% pewności, że drugi nie stchórzy. Ryzykiem jest mały zysk bez ryzyka (Zając).
        """)
        
    game_preset = st.selectbox("Wybierz scenariusz gry:", [
        "Dylemat Więźnia (Kartel)", 
        "Tchórz (Chicken / Wojna Cenowa)", 
        "Polowanie na Jelenia (Stag Hunt)",
        "Gra o sumie zerowej (Matching Pennies)"
    ])
    
    # Define payoffs
    if game_preset == "Dylemat Więźnia (Kartel)":
        r_labels = ["Współpraca (Wysokie Ceny)", "Zdrada (Niskie Ceny)"]
        c_labels = ["Współpraca (Wysokie Ceny)", "Zdrada (Niskie Ceny)"]
        payoffs_r = [[3, 0], [5, 1]]
        payoffs_c = [[3, 5], [0, 1]]
    elif game_preset == "Tchórz (Chicken / Wojna Cenowa)":
        r_labels = ["Ustąp (Wycofanie)", "Jedź prosto (Agresja)"]
        c_labels = ["Ustąp (Wycofanie)", "Jedź prosto (Agresja)"]
        payoffs_r = [[0, -1], [1, -10]]
        payoffs_c = [[0, 1], [-1, -10]]
    elif game_preset == "Polowanie na Jelenia (Stag Hunt)":
        r_labels = ["Poluj na Jelenia (Ryzyko)", "Poluj na Zająca (Bezpiecznie)"]
        c_labels = ["Poluj na Jelenia (Ryzyko)", "Poluj na Zająca (Bezpiecznie)"]
        payoffs_r = [[5, 0], [2, 2]]
        payoffs_c = [[5, 2], [0, 2]]
    else:
        r_labels = ["Strategia A", "Strategia B"]
        c_labels = ["Strategia A", "Strategia B"]
        payoffs_r = [[1, -1], [-1, 1]]
        payoffs_c = [[-1, 1], [1, -1]]
        
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Macierz Wypłat (Możesz modyfikować)")
        
        pr00 = st.number_input(f"Ty: {r_labels[0]} vs {c_labels[0]}", value=payoffs_r[0][0])
        pc00 = st.number_input(f"On: {r_labels[0]} vs {c_labels[0]}", value=payoffs_c[0][0])
        st.divider()
        pr01 = st.number_input(f"Ty: {r_labels[0]} vs {c_labels[1]}", value=payoffs_r[0][1])
        pc01 = st.number_input(f"On: {r_labels[0]} vs {c_labels[1]}", value=payoffs_c[0][1])
        st.divider()
        pr10 = st.number_input(f"Ty: {r_labels[1]} vs {c_labels[0]}", value=payoffs_r[1][0])
        pc10 = st.number_input(f"On: {r_labels[1]} vs {c_labels[0]}", value=payoffs_c[1][0])
        st.divider()
        pr11 = st.number_input(f"Ty: {r_labels[1]} vs {c_labels[1]}", value=payoffs_r[1][1])
        pc11 = st.number_input(f"On: {r_labels[1]} vs {c_labels[1]}", value=payoffs_c[1][1])
        
        # update matrix
        P_R = np.array([[pr00, pr01], [pr10, pr11]])
        P_C = np.array([[pc00, pc01], [pc10, pc11]])
        
    with col2:
        # Find Nash Equilibria
        nash_eqs = []
        best_responses_r = np.zeros((2,2))
        best_responses_c = np.zeros((2,2))
        
        for c_idx in range(2):
            best_r_val = np.max(P_R[:, c_idx])
            for r_idx in range(2):
                if P_R[r_idx, c_idx] == best_r_val:
                    best_responses_r[r_idx, c_idx] = 1
                    
        for r_idx in range(2):
            best_c_val = np.max(P_C[r_idx, :])
            for c_idx in range(2):
                if P_C[r_idx, c_idx] == best_c_val:
                    best_responses_c[r_idx, c_idx] = 1
                    
        for r_idx in range(2):
            for c_idx in range(2):
                if best_responses_r[r_idx, c_idx] == 1 and best_responses_c[r_idx, c_idx] == 1:
                    nash_eqs.append((r_idx, c_idx))
                    
        st.subheader("Wynik: Znalezione Równowagi Nasha")
        
        if not nash_eqs:
            st.warning("Brak równowagi Nasha w strategiach czystych (sprawdź zakładkę Strategie Mieszane).")
        else:
            for (r, c) in nash_eqs:
                st.success(f"**Równowaga Nasha:** {r_labels[r]} / {c_labels[c]} (Wypłaty: Ty {P_R[r,c]}, Przeciwnik {P_C[r,c]})")
        
        # Build Table visualization using Plotly Heatmap
        text_matrix = [[f"Ty: {P_R[i,j]}<br>On: {P_C[i,j]}" for j in range(2)] for i in range(2)]
        
        # Visual color for Nash
        color_matrix = [[0, 0], [0, 0]]
        for (r, c) in nash_eqs:
            color_matrix[r][c] = 1
            
        fig_mat = go.Figure(data=go.Heatmap(
            z=color_matrix,
            x=c_labels,
            y=r_labels,
            text=text_matrix,
            texttemplate="%{text}",
            colorscale=[[0, "#1f2937"], [1, "#10b981"]],
            showscale=False
        ))
        fig_mat.update_layout(
            title="Wypłaty (Zielony kwadrat to stabilna Równowaga Nasha)",
            xaxis_title="Strategia Przeciwnika",
            yaxis_title="Twoja Strategia",
            template="plotly_dark",
            height=400
        )
        # Fix y-axis order to read top-to-bottom
        fig_mat.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_mat, use_container_width=True)

# --- Tab 2: Mixed Strategies ---
with tabs[1]:
    st.header("2. Strategie Mieszane (Probabilistyczne)")
    
    with st.expander("📖 Jak z tego korzystać i co to oznacza?"):
        st.markdown("""
        **Co tu się dzieje?**
        Czasami najlepszą strategią jest bycie nieprzewidywalnym. Na przykład w rzutach karnych, jeśli zawsze strzelasz w lewo, bramkarz zawsze rzuci się w lewo. Musisz "mieszać" strategie (np. 60% w lewo, 40% w prawo).
        Wykres poniżej pozwala wyliczyć takie idealne proporcje.
        
        **Jak używać?**
        1. Suwaki po lewej stronie określają zyski w danej sytuacji (np. jeśli grasz strategię A, a wróg odpowie B).
        2. Wykres pokazuje "Zysk Oczekiwany" (Utility) w zależności od tego, z jakim prawdopodobieństwem grasz Strategię A.
        3. Miejsce, gdzie kolorowe linie się przecinają, to moment, w którym przeciwnik staje się całkowicie obojętny (nie jest w stanie Cię ograć). To Twoja **Mieszana Równowaga Nasha** – matematycznie bezpieczny punkt.
        """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Twoje Wypłaty")
        pA_A = st.slider("Zysk gdy Ty: A, Wróg: A", -10, 10, 1)
        pA_B = st.slider("Zysk gdy Ty: A, Wróg: B", -10, 10, -1)
        pB_A = st.slider("Zysk gdy Ty: B, Wróg: A", -10, 10, -1)
        pB_B = st.slider("Zysk gdy Ty: B, Wróg: B", -10, 10, 2)
        
    with col2:
        # P = prob that I play A. 1-P = prob I play B.
        p_range = np.linspace(0, 1, 100)
        
        # Expected utility if Enemy plays A constantly
        u_if_enemy_A = p_range * pA_A + (1 - p_range) * pB_A
        # Expected utility if Enemy plays B constantly
        u_if_enemy_B = p_range * pA_B + (1 - p_range) * pB_B
        
        # Intersection
        # p*A_A + (1-p)*B_A = p*A_B + (1-p)*B_B
        # p*(A_A - B_A) + B_A = p*(A_B - B_B) + B_B
        # p*(A_A - B_A - A_B + B_B) = B_B - B_A
        denom = (pA_A - pB_A - pA_B + pB_B)
        p_star = (pB_B - pB_A) / denom if denom != 0 else None
        
        fig_mix = go.Figure()
        fig_mix.add_trace(go.Scatter(x=p_range, y=u_if_enemy_A, mode='lines', name='Twój Zysk, gdy Wróg gra A', line=dict(color='cyan')))
        fig_mix.add_trace(go.Scatter(x=p_range, y=u_if_enemy_B, mode='lines', name='Twój Zysk, gdy Wróg gra B', line=dict(color='orange')))
        
        if p_star is not None and 0 <= p_star <= 1:
            u_star = p_star * pA_A + (1 - p_star) * pB_A
            fig_mix.add_trace(go.Scatter(x=[p_star], y=[u_star], mode='markers', name='Punkt Równowagi (Nash)', marker=dict(color='magenta', size=12, symbol='star')))
            st.success(f"**Optymalna strategia:** Graj **Strategię A** przez **{p_star*100:.1f}%** czasu, a **B** przez **{(1-p_star)*100:.1f}%** czasu. Zysk gwarantowany: **{u_star:.2f}**")
        else:
            st.warning("Brak równowagi mieszanej wewnętrznej (opłaca się grać czystą strategię 100%).")
            
        fig_mix.update_layout(
            title="Oczekiwana Użyteczność dla Strategii Mieszanych",
            xaxis_title="Prawdopodobieństwo grania Strategii A",
            yaxis_title="Twój Zysk Oczekiwany",
            template="plotly_dark",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_mix, use_container_width=True)

# --- Tab 3: Sequential Games ---
with tabs[2]:
    st.header("3. Gry Sekwencyjne i Drzewa Decyzyjne")
    
    with st.expander("📖 Jak z tego korzystać i co to oznacza?"):
        st.markdown("""
        **Co tu się dzieje?**
        Na rynkach ruchy rzadko są równoczesne (jak w Dylemacie Więźnia). Zazwyczaj Lider wykonuje ruch (np. Apple obniża cenę), a Obserwator reaguje. To gra sekwencyjna.
        Rozwiązujemy ją za pomocą **Indukcji Wstecznej (Backward Induction)** – zaczynamy analizę "od końca", patrząc jak zareaguje przeciwnik, i na tej podstawie wybieramy nasz dzisiejszy ruch.
        
        **Jak używać?**
        - Wcielasz się w Firmę 1 (Lidera). Masz do wyboru strategię Agresywną lub Ugodową.
        - Zależnie od Twojego ruchu, Firma 2 ma swoje wypłaty. Suwakami ustawiasz końcowe nagrody (Zysk Firmy 1, Zysk Firmy 2).
        - Zobaczysz, jak algorytm inteligentnie podświetli na złoto opłacalną gałąź. Zauważysz np., że czasami agresja się nie opłaca, jeśli wywoła odwet, który kosztuje więcej niż zysk.
        """)
        
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Firma 1 gra Agresywnie")
        p1_agg_agg = st.text_input("Zyski (Firma1, Firma2) jeśli F2 odpowie Agresją", "0, 0")
        p1_agg_pass = st.text_input("Zyski (Firma1, Firma2) jeśli F2 odpowie Pasywnie", "10, -5")
        
        st.subheader("Firma 1 gra Ugodowo")
        p1_pass_agg = st.text_input("Zyski (Firma1, Firma2) jeśli F2 odpowie Agresją", "-5, 10")
        p1_pass_pass = st.text_input("Zyski (Firma1, Firma2) jeśli F2 odpowie Pasywnie", "5, 5")
        
    with c2:
        # Parse inputs
        try:
            v_aa = [float(x.strip()) for x in p1_agg_agg.split(',')]
            v_ap = [float(x.strip()) for x in p1_agg_pass.split(',')]
            v_pa = [float(x.strip()) for x in p1_pass_agg.split(',')]
            v_pp = [float(x.strip()) for x in p1_pass_pass.split(',')]
        except:
            v_aa, v_ap, v_pa, v_pp = [0,0], [10,-5], [-5,10], [5,5]
            
        # Backward Induction
        # Node 2 (top): F2 chooses max of v_aa[1] and v_ap[1]
        f2_choice_top = "Agresja" if v_aa[1] > v_ap[1] else "Pasywnie"
        f1_payoff_top = v_aa[0] if f2_choice_top == "Agresja" else v_ap[0]
        
        # Node 3 (bottom): F2 chooses max of v_pa[1] and v_pp[1]
        f2_choice_bot = "Agresja" if v_pa[1] > v_pp[1] else "Pasywnie"
        f1_payoff_bot = v_pa[0] if f2_choice_bot == "Agresja" else v_pp[0]
        
        # Node 1 (root): F1 chooses max of f1_payoff_top and f1_payoff_bot
        f1_choice = "Agresja" if f1_payoff_top > f1_payoff_bot else "Ugodowo"
        
        st.success(f"**Rozwiązanie metodą indukcji wstecznej:** Firma 1 powinna wybrać strategię **{f1_choice}**, ponieważ przewiduje, że Firma 2 odpowie **{'Agresją' if f1_choice=='Agresja' and f2_choice_top=='Agresja' or f1_choice=='Ugodowo' and f2_choice_bot=='Agresja' else 'Pasywnie'}**.")

        # Hardcode Tree drawing
        X = [0, 1, 1, 2, 2, 2, 2]
        Y = [0, 1, -1, 1.5, 0.5, -0.5, -1.5]
        labels = [
            "Firma 1", 
            "Firma 2", 
            "Firma 2",
            f"{v_aa}", f"{v_ap}", f"{v_pa}", f"{v_pp}"
        ]
        
        # Edges
        edges = [(0,1,"Agresja"), (0,2,"Ugodowo"), (1,3,"Agresja"), (1,4,"Pasywnie"), (2,5,"Agresja"), (2,6,"Pasywnie")]
        
        fig_tree = go.Figure()
        
        for edge in edges:
            x0, y0 = X[edge[0]], Y[edge[0]]
            x1, y1 = X[edge[1]], Y[edge[1]]
            
            # Determine line color (gold if it's the chosen path)
            line_color = "gray"
            line_width = 1
            if edge[0] == 0 and edge[2] == f1_choice:
                line_color = "gold"
                line_width = 4
            if edge[0] == 1 and edge[2] == f2_choice_top and f1_choice == "Agresja":
                line_color = "gold"
                line_width = 4
            if edge[0] == 2 and edge[2] == f2_choice_bot and f1_choice == "Ugodowo":
                line_color = "gold"
                line_width = 4
                
            fig_tree.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines', line=dict(color=line_color, width=line_width), showlegend=False))
            fig_tree.add_trace(go.Scatter(x=[(x0+x1)/2], y=[(y0+y1)/2 + 0.15], mode='text', text=[edge[2]], textfont=dict(color="white", size=10), showlegend=False))

        fig_tree.add_trace(go.Scatter(x=X, y=Y, mode='markers+text', text=labels, textposition="middle right", marker=dict(size=15, color='#4A90E2'), showlegend=False))
        fig_tree.update_layout(title="Drzewo Decyzyjne (Złota linia to optymalna ścieżka SPE)", template="plotly_dark", xaxis=dict(showgrid=False, zeroline=False, visible=False), yaxis=dict(showgrid=False, zeroline=False, visible=False), height=400)
        st.plotly_chart(fig_tree, use_container_width=True)

# --- Tab 4: Auction Theory ---
with tabs[3]:
    st.header("4. Teoria Aukcji i Licytacje (Gry Bayesowskie)")
    
    with st.expander("📖 Jak z tego korzystać i co to oznacza?"):
        st.markdown("""
        **Co tu się dzieje?**
        Licytujesz aktywo (np. spółkę, nieruchomość, obligację). Wiesz, ile jest dla Ciebie warte, ale nie wiesz, jak oceniają to inni. 
        - **Aukcja Pierwszej Ceny (First-Price):** Kto wygrywa, płaci to, co zalicytował. Tutaj występuje zjawisko *Bid Shading* – musisz zalicytować *mniej* niż to jest dla Ciebie warte, by cokolwiek zyskać. Jeśli zalicytujesz 100%, twój "zysk" wynosi 0!
        - **Aukcja Drugiej Ceny (Vickreya):** Wygrywasz, ale płacisz tylko tyle, ile wynosiła druga najwyższa oferta. Szlachetna matematyka udowadnia, że tutaj JEDYNĄ słuszną strategią jest zaproponowanie dokładnie tego, ile wyceniasz aktywo.
        
        **Jak używać?**
        - Zmieniaj **Prawdziwą Twoją Wartość**.
        - Wykres pokazuje symulację: Jeśli zaczniesz licytować różne procenty swojej wartości w Aukcji Pierwszej Ceny, gdzie leży szczyt Twojego opłacalnego Zysku? Oś Y to "Średni Zysk".
        """)
        
    col1, col2 = st.columns([1, 2])
    with col1:
        my_value = st.slider("Twoja wycena aktywa ($)", 100, 1000, 500)
        n_bidders = st.slider("Liczba konkurentów", 1, 20, 5)
        st.info("Algorytm symuluje 10,000 wirtualnych aukcji dla każdej Twojej decyzji, by znaleźć wartość oczekiwaną zysku (Expected Profit).")
        
    with col2:
        # Simulate First Price Auction expected profit
        bids = np.linspace(0.1, 1.0, 50) * my_value
        expected_profits = []
        
        np.random.seed(42)
        for bid in bids:
            # Competitors bid uniformly between 0 and 1000
            comp_bids = np.random.uniform(0, 1000, (10000, n_bidders))
            max_comp_bid = np.max(comp_bids, axis=1)
            
            # I win if my bid is greater than max competitor bid
            wins = bid > max_comp_bid
            
            # My profit if I win is (my_value - my_bid), 0 otherwise
            profit = np.mean(wins * (my_value - bid))
            expected_profits.append(profit)
            
        optimal_bid_idx = np.argmax(expected_profits)
        optimal_bid = bids[optimal_bid_idx]
        
        fig_auc = go.Figure()
        fig_auc.add_trace(go.Scatter(x=bids, y=expected_profits, mode='lines', name='Oczekiwany Zysk', line=dict(color='#00CC96', width=3)))
        fig_auc.add_trace(go.Scatter(x=[optimal_bid], y=[expected_profits[optimal_bid_idx]], mode='markers', name='Optymalna Oferta (Bid Shading)', marker=dict(color='magenta', size=12, symbol='star')))
        
        fig_auc.add_vline(x=my_value, line_dash="dash", line_color="red", annotation_text="Prawdziwa Wycena")
        
        fig_auc.update_layout(title="Aukcja Pierwszej Ceny: Zysk oczekiwany od wysokości oferty", xaxis_title="Złożona Oferta ($)", yaxis_title="Średni Oczekiwany Zysk ($)", template="plotly_dark")
        st.plotly_chart(fig_auc, use_container_width=True)
        
        theoretical_optimal = my_value * (n_bidders / (n_bidders + 1))
        st.success(f"**Twierdzenie analityczne:** W idealnej sytuacji powinieneś licytować **{n_bidders}/{n_bidders+1}** swojej wyceny. Symulacja wykazała optimum przy **${optimal_bid:.0f}** (Prawda = ${my_value}). W Aukcji Vickreya po prostu podaj ${my_value}!")

# --- Tab 5: Evolutionary Game Theory ---
with tabs[4]:
    st.header("5. Gry Ewolucyjne (Dynamika Rynku)")
    
    with st.expander("📖 Jak z tego korzystać i co to oznacza?"):
        st.markdown("""
        **Co tu się dzieje?**
        Teoria Gier Ewolucyjnych (EGT) tłumaczy, jak strategie zachowują się w dużych, ślepych populacjach, które uczą się przez naśladownictwo (np. spanikowani inwestorzy kopiujący zyskowne ruchy innych).
        Model "Jastrząb vs Gołąb" (Hawk-Dove) świetnie opisuje byki (agresywnych) i niedźwiedzie (ostrożnych). 
        
        **Zasady:**
        - Jastrząb walczy z Jastrzębiem: tracą krew (Koszt walki C).
        - Gołąb z Gołębiem: dzielą się zyskiem (Nagroda V).
        - Jastrząb z Gołębiem: Jastrząb zabiera wszystko, Gołąb ucieka.
        
        **Jak używać?**
        Zmieniaj Nagrodę (V) i Koszt (C). 
        - Zauważ, że jeśli Koszt jest mniejszy niż Nagroda (C < V), Jastrzębie całkowicie eliminują Gołębie (dominacja agresji rynkowej). 
        - Jeśli jednak Koszt > Nagroda (np. wielki Drawdown, bolesne rynki niedźwiedzia), ustala się matematycznie ścisła równowaga – na rynku przeżywa specyficzny ułamek agresywnych i pasywnych graczy. Wykres przedstawia Portret Fazowy tego, jak rynek sam dąży do równowagi poprzez oscylacje z czasem!
        """)
        
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("Parametry Środowiska")
        v = st.slider("Nagroda za wygraną rynkową (V)", 1.0, 10.0, 4.0)
        c = st.slider("Koszt walki / Drawdown (C)", 1.0, 10.0, 6.0)
        init_hawks = st.slider("Początkowy % Jastrzębi (Hossy)", 0.01, 0.99, 0.10)
        
    with c2:
        # Replicator dynamics
        def replicator(x, t, v, c):
            p_h = x
            p_d = 1 - x
            
            u_h = p_h * (v - c)/2 + p_d * v
            u_d = p_h * 0 + p_d * v/2
            
            u_avg = p_h * u_h + p_d * u_d
            
            dxdt = p_h * (u_h - u_avg)
            return dxdt
            
        t = np.linspace(0, 20, 200)
        x_out = odeint(replicator, init_hawks, t, args=(v, c))
        
        fig_evo = go.Figure()
        fig_evo.add_trace(go.Scatter(x=t, y=x_out.ravel()*100, mode='lines', name='% Jastrzębi', line=dict(color='#ff1744', width=3)))
        fig_evo.add_trace(go.Scatter(x=t, y=(1 - x_out.ravel())*100, mode='lines', name='% Gołębi', line=dict(color='#3498db', dash='dash')))
        
        fig_evo.update_layout(title="Dynamika Replikatorowa populacji rynkowej", xaxis_title="Czas", yaxis_title="Procent populacji (%)", template="plotly_dark", yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_evo, use_container_width=True)
        
        # Eq
        if v >= c:
            st.success("Nagroda przewyższa koszt. Ewolucyjna Równowaga Stabilna (ESS) to **100% Jastrzębi**.")
        else:
            eq = v / c
            st.success(f"Koszt przewyższa nagrodę. Społeczeństwo wymusza koegzystencję! ESS to **{eq*100:.1f}% Jastrzębi** i **{(1-eq)*100:.1f}% Gołębi**.")

# --- Tab 6: Mechanism Design ---
with tabs[5]:
    st.header("6. Mechanism Design — Projektowanie Rynków")
    st.markdown("Mechanism Design to 'odwrócona teoria gier'. Zaczynamy od pożądanego wyniku (np. sprawiedliwa alokacja zasobów) i projektujemy zasady gry tak, by gracze, działając w swoim interesie, go osiągnęli.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Aukcja Kombinatoryczna (Package Bidding)")
        st.markdown("""
        Wyobraź sobie sprzedaż licencji na częstotliwości (Spectrum Auction). 
        Wartość licencji A+B jest często większa niż suma A i B oddzielnie. 
        Projektujemy mechanizm, który pozwala licytować pakiety (bundles).
        """)
        v_a = st.number_input("Twoja wycena licencji A", 100, 500, 200)
        v_b = st.number_input("Twoja wycena licencji B", 100, 500, 250)
        v_ab = st.number_input("Twoja wycena pakietu A+B (Synergia)", 200, 1500, 600)
        
        bid_a = st.slider("Twoja oferta za A", 0, v_a, int(v_a*0.8))
        bid_b = st.slider("Twoja oferta za B", 0, v_b, int(v_b*0.8))
        bid_ab = st.slider("Twoja oferta za pakiet A+B", 0, v_ab, int(v_ab*0.8))

    with col2:
        st.subheader("Symulacja VCG (Vickrey-Clarke-Groves)")
        st.markdown("Mechanizm VCG to potężne narzędzie: każdy płaci koszt, jaki nakłada na innych uczestników. To zmusza do podawania prawdziwych wycen.")
        
        # Mock competitors
        comp_bids = {
            "Comp1": {"A": 180, "B": 210, "AB": 450},
            "Comp2": {"A": 220, "B": 190, "AB": 480}
        }
        
        # Simple allocation logic: max total bid
        scenarios = [
            {"alloc": "You get AB", "val": bid_ab},
            {"alloc": "Comp1 gets AB", "val": comp_bids["Comp1"]["AB"]},
            {"alloc": "Comp2 gets AB", "val": comp_bids["Comp2"]["AB"]},
            {"alloc": "You A, Comp1 B", "val": bid_a + comp_bids["Comp1"]["B"]},
            {"alloc": "You B, Comp2 A", "val": bid_b + comp_bids["Comp2"]["A"]},
        ]
        best_scen = max(scenarios, key=lambda x: x["val"])
        
        st.info(f"**Wynik alokacji:** {best_scen['alloc']} (Suma ofert: {best_scen['val']})")
        
        # Visualization of Bid Shading vs VCG
        fig_vcg = go.Figure()
        fig_vcg.add_trace(go.Bar(x=["Twoja Wycena", "Twoja Oferta"], y=[v_ab, bid_ab], marker_color=["#3498db", "#ffea00"]))
        fig_vcg.update_layout(title="Wycena vs Oferta w aukcji kombinatorycznej", template="plotly_dark", height=300)
        st.plotly_chart(fig_vcg, use_container_width=True)
        
        st.markdown("""
        **Zastosowanie rynkowe:**
        - Tworzenie ETF-ów (alokacja jednostek).
        - Aukcje Google Ads (GSP - Generalized Second Price).
        - Rynki energii (dopasowanie podaży i popytu w sieciach).
        """)

# --- Tab 7: Repeated Games ---
with tabs[6]:
    st.header("7. Gry Powtarzalne i Folk Theorem")
    st.markdown("Większość interakcji rynkowych nie jest jednorazowa. Reputacja i oczekiwanie przyszłych zysków pozwalają na współpracę, która w grze jednorazowej byłaby niemożliwa.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Strategie w IPD")
        st.markdown("Iterated Prisoner's Dilemma (IPD):")
        delta = st.slider("Współczynnik dyskontowania (δ) — prawdopodobieństwo następnej rundy", 0.0, 0.99, 0.9)
        
        strat = st.selectbox("Twoja Strategia:", ["Tit-for-Tat", "Always Defect", "Grim Trigger", "Always Cooperate"])
        
        st.markdown("""
        - **Grim Trigger:** Współpracuję, dopóki Ty współpracujesz. Jeśli raz zdradzisz, będę Cię karać wiecznie.
        - **Folk Theorem:** Jeśli δ jest wystarczająco wysokie (jesteś cierpliwy), każdy wynik lepszy niż Równowaga Nasha jest osiągalny jako stabilna kooperacja.
        """)

    with col2:
        # Simulation of Grim Trigger stability
        # Payoffs from Tab 1 (Prisoner's Dilemma)
        R, T, S, P = 3, 5, 0, 1 # Reward, Temptation, Sucker, Punishment
        
        # Value of cooperation (perpetual R): V_c = R / (1-delta)
        v_coop = R / (1 - delta) if delta < 1 else 100
        # Value of defecting once and then being punished (P): V_d = T + delta*P / (1-delta)
        v_defect = T + (delta * P) / (1 - delta) if delta < 1 else 100
        
        fig_rep = go.Figure()
        fig_rep.add_trace(go.Bar(x=["Ciągła Współpraca", "Jednorazowa Zdrada + Kara"], y=[v_coop, v_defect], marker_color=["#00e676", "#ff1744"]))
        fig_rep.update_layout(title=f"Wartość długoterminowa strategii (δ = {delta})", template="plotly_dark", height=350)
        st.plotly_chart(fig_rep, use_container_width=True)
        
        if v_coop > v_defect:
            st.success(f"Dla δ = {delta} kooperacja jest matematycznie OPŁACALNA (Punkt Grim Trigger).")
        else:
            st.error(f"Dla δ = {delta} pokusa zdrady jest zbyt silna. Rynek upadnie do wzajemnej zdrady.")

# --- Tab 8: Matching Theory ---
with tabs[7]:
    st.header("8. Matching Theory — Algorytm Gale-Shapley (Interaktywny)")
    with st.expander("📖 Jak działa Deferred Acceptance?"):
        st.markdown("""
        **Algorytm Gale-Shapley (1962, Nobel 2012 — Roth & Shapley):**
        1. Proponenci (Inwestorzy) składają oferty wg listy preferencji.
        2. Akceptanci (Startupy) trzymają *tymczasowo* najlepszą ofertę i odrzucają gorsze.
        3. Odrzuceni proponenci składają kolejne oferty. Powtarzamy aż nikt nowy nie proponuje.
        
        **Wynik:** Stabilne dopasowanie (brak blokujących par) optymalne dla proponentów.
        **Nobel 2012:** Alvin Roth i Lloyd Shapley.
        """)
    st.subheader("Edytuj preferencje (wpisz ranking oddzielony przecinkami)")
    participants = ["I1", "I2", "I3"]
    targets = ["S1", "S2", "S3"]
    prop_prefs = {}
    rec_prefs = {}
    c_p, c_r = st.columns(2)
    with c_p:
        st.markdown("**Preferencje Inwestorów →**")
        defaults_p = {"I1": "S1, S2, S3", "I2": "S2, S1, S3", "I3": "S1, S3, S2"}
        for p in participants:
            val = st.text_input(f"{p} preferuje:", value=defaults_p[p], key=f"pp_{p}")
            prop_prefs[p] = [x.strip() for x in val.split(",") if x.strip() in targets]
    with c_r:
        st.markdown("**Preferencje Startupów →**")
        defaults_r = {"S1": "I2, I1, I3", "S2": "I1, I3, I2", "S3": "I3, I2, I1"}
        for r in targets:
            val = st.text_input(f"{r} preferuje:", value=defaults_r[r], key=f"rp_{r}")
            rec_prefs[r] = [x.strip() for x in val.split(",") if x.strip() in participants]
    if st.button("▶ Uruchom Algorytm Gale-Shapley", type="primary"):
        try:
            matching = gale_shapley(prop_prefs, rec_prefs)
            blocking = check_stability(matching, prop_prefs, rec_prefs)
            st.success("✅ Stabilne dopasowanie znalezione!" if not blocking else f"⚠️ Niestabilne! Blokujące pary: {blocking}")
            match_df = pd.DataFrame(list(matching.items()), columns=["Inwestor", "Startup"])
            st.dataframe(match_df, use_container_width=True, hide_index=True)
            # Wizualizacja
            fig_match = go.Figure()
            py = {p: (len(participants) - i) for i, p in enumerate(participants)}
            ry = {r: (len(targets) - i) for i, r in enumerate(targets)}
            fig_match.add_trace(go.Scatter(
                x=[1]*len(participants), y=list(py.values()),
                mode="markers+text", text=list(py.keys()),
                textposition="middle left", name="Inwestorzy",
                marker=dict(size=24, color="#3498db")))
            fig_match.add_trace(go.Scatter(
                x=[2]*len(targets), y=list(ry.values()),
                mode="markers+text", text=list(ry.keys()),
                textposition="middle right", name="Startupy",
                marker=dict(size=24, color="#00e676")))
            for inv, sta in matching.items():
                fig_match.add_trace(go.Scatter(
                    x=[1, 2], y=[py[inv], ry[sta]],
                    mode="lines", line=dict(color="#f1c40f", width=3),
                    showlegend=False))
            fig_match.update_layout(
                title="Dopasowanie Inwestorzy ↔ Startupy",
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                template="plotly_dark", height=350)
            st.plotly_chart(fig_match, use_container_width=True)
        except Exception as e:
            st.error(f"Błąd algorytmu: {e}")
    else:
        st.info("Skonfiguruj preferencje i kliknij ▶ Uruchom.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 9: Solver NxM
# ─────────────────────────────────────────────────────────────────────────────
with tabs[8]:
    st.header("9. Solver NxM — Wszystkie Równowagi Nasha")
    with st.expander("📖 Jak to działa?"):
        st.markdown("""
        Klasyczny 2×2 to za mało. Ten solver używa **Support Enumeration Method** —
        sprawdza każdą możliwą kombinację nośników strategii i rozwiązuje układ równań
        indifferentności. Zwraca **wszystkie** czyste i mieszane równowagi.
        """)
    size_n = st.slider("Liczba strategii Gracza A (wiersze)", 2, 4, 3)
    size_m = st.slider("Liczba strategii Gracza B (kolumny)", 2, 4, 3)
    st.markdown("**Macierz wypłat Gracza A**")
    default_r = np.random.randint(-2, 6, (size_n, size_m)).tolist()
    default_c = np.random.randint(-2, 6, (size_n, size_m)).tolist()
    col_r, col_c = st.columns(2)
    with col_r:
        st.caption("Wypłaty Gracza A")
        input_r = []
        for i in range(size_n):
            row_vals = st.text_input(f"Wiersz {i+1} (Gracz A)",
                value=", ".join(str(v) for v in default_r[i]), key=f"nr_{i}")
            try:
                input_r.append([float(x) for x in row_vals.split(",")])
            except:
                input_r.append(default_r[i])
    with col_c:
        st.caption("Wypłaty Gracza B")
        input_c = []
        for i in range(size_n):
            row_vals = st.text_input(f"Wiersz {i+1} (Gracz B)",
                value=", ".join(str(v) for v in default_c[i]), key=f"nc_{i}")
            try:
                input_c.append([float(x) for x in row_vals.split(",")])
            except:
                input_c.append(default_c[i])
    try:
        PR = np.array(input_r)
        PC = np.array(input_c)
        if PR.shape == (size_n, size_m) and PC.shape == (size_n, size_m):
            all_ne = find_all_nash(PR, PC)
            if not all_ne:
                st.warning("Brak równowag Nasha (sprawdź macierz).")
            else:
                st.success(f"Znaleziono **{len(all_ne)}** równowag(i) Nasha.")
                for idx, ne in enumerate(all_ne):
                    label = "Czysta" if ne['type'] == 'pure' else "Mieszana"
                    with st.expander(f"RN #{idx+1} ({label}) — Zysk A: {ne['payoff_r']:.2f}, Zysk B: {ne['payoff_c']:.2f}"):
                        df_ne = pd.DataFrame({
                            "Strategia": [f"S{i+1}" for i in range(size_n)],
                            "σ Gracz A": [f"{v:.3f}" for v in ne['sigma_r']],
                        })
                        df_ne2 = pd.DataFrame({
                            "Strategia": [f"S{j+1}" for j in range(size_m)],
                            "σ Gracz B": [f"{v:.3f}" for v in ne['sigma_c']],
                        })
                        c1, c2 = st.columns(2)
                        c1.dataframe(df_ne, hide_index=True)
                        c2.dataframe(df_ne2, hide_index=True)
            # Heatmap czystych
            pure_ne = find_pure_nash(PR, PC)
            color_m = np.zeros((size_n, size_m))
            for (ri, ci) in pure_ne:
                color_m[ri, ci] = 1
            text_m = [[f"A:{PR[i,j]:.0f} B:{PC[i,j]:.0f}" for j in range(size_m)] for i in range(size_n)]
            fig_hm = go.Figure(go.Heatmap(
                z=color_m,
                text=text_m, texttemplate="%{text}",
                colorscale=[[0,"#1f2937"],[1,"#10b981"]], showscale=False,
                x=[f"B-S{j+1}" for j in range(size_m)],
                y=[f"A-S{i+1}" for i in range(size_n)],
            ))
            fig_hm.update_layout(title="Heatmapa czystych RN (zielony = NE)",
                template="plotly_dark", height=300)
            fig_hm.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_hm, use_container_width=True)
    except Exception as e:
        st.error(f"Błąd: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 10: Correlated Equilibrium
# ─────────────────────────────────────────────────────────────────────────────
with tabs[9]:
    st.header("10. Równowaga Korelatywna (Aumann 1987)")
    with st.expander("📖 Czym różni się od NE?"):
        st.markdown("""
        **Równowaga Korelatywna** to rozszerzenie NE. Wprowadza **zewnętrznego koordynatora**
        (np. sygnał Fed, rekomendacja analityczna), który wysyła prywatne sugestie graczom.
        Gracze dobrowolnie je słuchają, bo im to odpowiada (warunek IC).
        
        🔑 **Kluczowe fakty:**
        - Każda RN jest szczególnym przypadkiem CE (zbiór CE ⊇ zbiór NE).
        - CE może dawać **wyższy dobrobyt społeczny** niż NE.
        - Rozwiązywalne przez programowanie liniowe (LP).
        - **Nobel 2005 (Aumann)** za teorię powtarzalnych gier i CE.
        
        **Zastosowania rynkowe:** Sygnały makro Fed → koordynacja portfeli.
        """)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Macierz 2×2")
        ce_preset = st.selectbox("Preset:", ["Dylemat Więźnia", "Tchórz", "Polowanie na Jelenia", "Własny"])
        presets = {
            "Dylemat Więźnia": (np.array([[3,0],[5,1]]), np.array([[3,5],[0,1]])),
            "Tchórz": (np.array([[0,-1],[1,-10]]), np.array([[0,1],[-1,-10]])),
            "Polowanie na Jelenia": (np.array([[5,0],[2,2]]), np.array([[5,2],[0,2]])),
            "Własny": (np.array([[4,0],[6,2]]), np.array([[4,6],[0,2]])),
        }
        ce_pr, ce_pc = presets[ce_preset]
        obj = st.radio("Cel optymalizacji:", ["welfare","row","col"],
            format_func=lambda x: {"welfare":"Dobrobyt społeczny","row":"Max Gracz A","col":"Max Gracz B"}[x])
    with c2:
        ce_res = find_correlated_equilibrium(ce_pr, ce_pc, objective=obj)
        if ce_res["success"]:
            dist = ce_res["distribution"]
            st.success(f"✅ CE znalezione! Dobrobyt: **{ce_res['social_welfare']:.3f}** | A: {ce_res['payoff_r']:.3f} | B: {ce_res['payoff_c']:.3f}")
            fig_ce = go.Figure(go.Heatmap(
                z=dist,
                text=[[f"{dist[i,j]:.3f}" for j in range(2)] for i in range(2)],
                texttemplate="%{text}",
                colorscale="Teal", showscale=True,
                x=["B: Współpraca","B: Zdrada"],
                y=["A: Współpraca","A: Zdrada"],
            ))
            fig_ce.update_layout(title="Rozkład sygnałów CE p(i,j)",
                template="plotly_dark", height=350)
            fig_ce.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_ce, use_container_width=True)
            # Porównanie CE vs NE
            ne_list = find_all_nash(ce_pr, ce_pc)
            ne_welfares = [ne['payoff_r']+ne['payoff_c'] for ne in ne_list]
            best_ne_w = max(ne_welfares) if ne_welfares else 0
            delta_w = ce_res['social_welfare'] - best_ne_w
            st.metric("Zysk z koordynacji (CE − najlepsza NE)",
                f"{delta_w:+.3f}", delta_color="normal")
        else:
            st.error("LP nie znalazł rozwiązania.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 11: Bayesian NE
# ─────────────────────────────────────────────────────────────────────────────
with tabs[10]:
    st.header("11. Bayesowska Równowaga Nasha (BNE)")
    with st.expander("📖 Czym jest BNE?"):
        st.markdown("""
        **BNE** (Harsanyi 1967–68, Nobel 1994) rozszerza NE o **prywatną informację**.
        Każdy gracz ma *typ* (np. koszt produkcji, wycenę aktywa), który zna tylko on.
        Strategia jest teraz funkcją warunkową: *co robię, jeśli jestem typem X?*
        
        **Przykład rynkowy:** Gracz H = insider (zna wyniki kwartalne),
        Gracz L = retail investor. Jakie strategie są równowagą Bayesowską?
        """)
    bc1, bc2 = st.columns(2)
    with bc1:
        prob_H = st.slider("P(Gracz jest typem H — informed)", 0.05, 0.95, 0.4)
        st.markdown("**Wypłaty Typ H vs Typ H:**")
        bHH = np.array([[st.number_input("HH(0,0)",value=4.0,key="hh00"),
                         st.number_input("HH(0,1)",value=0.0,key="hh01")],
                        [st.number_input("HH(1,0)",value=6.0,key="hh10"),
                         st.number_input("HH(1,1)",value=2.0,key="hh11")]])
    with bc2:
        st.markdown("**Wypłaty Typ H vs Typ L / Typ L vs reszta:**")
        bHL = np.array([[3.0,0.0],[5.0,1.0]])
        bLH = np.array([[2.0,0.0],[4.0,1.0]])
        bLL = np.array([[1.0,0.0],[3.0,1.0]])
        st.info("Uproszczenie: HL/LH/LL używają presetów Dylematu Więźnia z obniżonymi stawkami.")
    bne_res = solve_bne_two_types(bHH, bHL, bLH, bLL, prob_H)
    eqs = bne_res["equilibria"]
    st.subheader(f"Znalezione BNE: {len(eqs)}")
    if not eqs:
        st.warning("Brak BNE w zadanej siatce.")
    else:
        rows = []
        for eq in eqs:
            rows.append({
                "Typ H gra S1 z p=": eq["p_H"],
                "Typ L gra S1 z p=": eq["p_L"],
                "E[u] Typ H": eq["eu_H"],
                "E[u] Typ L": eq["eu_L"],
                "Rodzaj": eq["type"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        # Wykres przestrzeni BNE
        pure_bne = [e for e in eqs if e["type"]=="pure"]
        mixed_bne = [e for e in eqs if e["type"]=="mixed"]
        fig_bne = go.Figure()
        if mixed_bne:
            fig_bne.add_trace(go.Scatter(
                x=[e["p_H"] for e in mixed_bne],
                y=[e["p_L"] for e in mixed_bne],
                mode="markers", name="Mieszane BNE",
                marker=dict(color="cyan", size=10, symbol="circle")))
        if pure_bne:
            fig_bne.add_trace(go.Scatter(
                x=[e["p_H"] for e in pure_bne],
                y=[e["p_L"] for e in pure_bne],
                mode="markers", name="Czyste BNE",
                marker=dict(color="magenta", size=14, symbol="star")))
        fig_bne.update_layout(
            title="Przestrzeń BNE (os X = p_H, os Y = p_L)",
            xaxis_title="P(Typ H gra S1)", yaxis_title="P(Typ L gra S1)",
            template="plotly_dark", height=380,
            xaxis=dict(range=[-0.05,1.05]), yaxis=dict(range=[-0.05,1.05]))
        st.plotly_chart(fig_bne, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 12: Stackelberg
# ─────────────────────────────────────────────────────────────────────────────
with tabs[11]:
    st.header("12. Równowaga Stackelberga — Lider i Naśladowca")
    with st.expander("📖 Stackelberg vs Cournot vs Nash"):
        st.markdown("""
        **Stackelberg (1934):** Lider ogłasza ilość produkcji *jako pierwszy*, 
        wiedząc że Follower zoptymalizuje odpowiedź. Kluczowe: **commitment** — 
        wiarygodne zobowiązanie do pierwszego ruchu daje *First Mover Advantage*.
        
        **Zastosowania finansowe:**
        - Fed (Lider) → banki komercyjne (Follower)
        - Dominant market maker → pozostałe biura kwotujące
        - Emitent obligacji → inwestorzy na rynku wtórnym
        """)
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        a_d = st.slider("Parametr popytu (a)", 50.0, 200.0, 100.0)
        b_d = st.slider("Nachylenie krzywej popytu (b)", 0.1, 5.0, 1.0)
        c_l = st.slider("Koszt marginalny Lidera", 0.0, 30.0, 10.0)
        c_f = st.slider("Koszt marginalny Followera", 0.0, 30.0, 15.0)
    with sc2:
        sk = stackelberg_cournot(a_d, b_d, c_l, c_f)
        # Wykres — krzywe reakcji
        q1_range = np.linspace(0, (a_d - c_l) / b_d, 200)
        br_f = np.maximum(0, (a_d - c_f - b_d * q1_range) / (2 * b_d))
        br_l_follower = np.maximum(0, (a_d - c_l - b_d * q1_range) / (2 * b_d))
        fig_sk = go.Figure()
        fig_sk.add_trace(go.Scatter(x=q1_range, y=br_f, mode='lines',
            name='Funkcja reakcji Followera (BR₂)', line=dict(color='cyan')))
        fig_sk.add_trace(go.Scatter(x=q1_range, y=br_l_follower, mode='lines',
            name='Funkcja reakcji Lidera (BR₁)', line=dict(color='orange', dash='dash')))
        fig_sk.add_trace(go.Scatter(
            x=[sk['q_leader']], y=[sk['q_follower']], mode='markers',
            name='Stackelberg NE', marker=dict(color='magenta', size=14, symbol='star')))
        cournot_q = sk['nash_q_symmetric']
        fig_sk.add_trace(go.Scatter(
            x=[cournot_q], y=[cournot_q], mode='markers',
            name='Cournot NE (symultaniczny)', marker=dict(color='gold', size=12, symbol='diamond')))
        fig_sk.update_layout(
            title="Krzywe Reakcji i Równowagi",
            xaxis_title="Ilość Lidera (q₁)", yaxis_title="Ilość Followera (q₂)",
            template="plotly_dark", height=400)
        st.plotly_chart(fig_sk, use_container_width=True)
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Ilość Lidera", f"{sk['q_leader']:.2f}")
        col_m2.metric("Ilość Followera", f"{sk['q_follower']:.2f}")
        col_m3.metric("Zysk Lidera", f"{sk['profit_leader']:.2f}")
        col_m4.metric("First Mover Advantage", f"{sk['first_mover_advantage']:+.2f}")
        st.success(f"Cena rynkowa: **{sk['price']:.2f}** | Łączna produkcja: **{sk['total_output']:.2f}**")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 13: Mean-Field Games
# ─────────────────────────────────────────────────────────────────────────────
with tabs[12]:
    st.header("13. Mean-Field Games — Populacje Agentów")
    with st.expander("📖 Czym są MFG?"):
        st.markdown("""
        **Mean-Field Games** (Lasry & Lions 2006; Huang, Malhamé, Caines 2006) 
        modelują gry z **nieskończenie wieloma** racjonalnymi agentami. Zamiast śledzić
        każdego z osobna, śledzimy tylko **rozkład** stanu populacji (mean field).
        
        🔥 **Gorący temat 2024-2025:**
        - DeFi: Automated Market Makers (Bergault et al. 2024)
        - HFT: miliony mikrozleceń jako pole średnie (NeurIPS 2024)
        - Bitcoin mining: równowaga przy centralizacji nagród
        
        **Model LQ (Linear-Quadratic):** Każdy inwestor minimalizuje koszt 
        odchylenia od średniego portfela rynkowego (tracking error) plus koszt transakcji.
        """)
    mf1, mf2 = st.columns([1, 2])
    with mf1:
        mf_alpha = st.slider("α — siła mean field (presja do konformizmu)", 0.01, 2.0, 0.5)
        mf_beta = st.slider("β — koszt transakcji", 0.1, 3.0, 1.0)
        mf_sigma = st.slider("σ — zmienność rynku", 0.0, 1.0, 0.3)
        mf_T = st.slider("Horyzont T (lata)", 1.0, 20.0, 10.0)
        mf_x0 = st.slider("x₀ — startowy poziom portfela", 0.5, 2.0, 1.3)
        st.info("Agent startuje z x₀ > 1.0 (przeważony). MFG wyznacza optymalną ścieżkę powrotu do środka.")
    with mf2:
        mfg = lq_mean_field_game(T=mf_T, n_steps=300, alpha=mf_alpha,
                                  beta=mf_beta, sigma=mf_sigma, x0=mf_x0)
        t = mfg["t"]
        fig_mf = go.Figure()
        fig_mf.add_trace(go.Scatter(
            x=t, y=mfg["mean_field"], mode='lines',
            name='Mean Field m(t) — śr. portfel rynku',
            line=dict(color='#00e676', width=3)))
        fig_mf.add_trace(go.Scatter(
            x=t, y=mfg["agent_path"], mode='lines',
            name='Trajektoria agenta x(t)',
            line=dict(color='#ff6d00', width=2, dash='dot')))
        fig_mf.update_layout(
            title="LQ Mean-Field Game: Optymalny Tracking Portfela",
            xaxis_title="Czas", yaxis_title="Poziom portfela",
            template="plotly_dark", height=360,
            legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig_mf, use_container_width=True)
        fig_ctrl = go.Figure(go.Scatter(
            x=t[:-1], y=mfg["optimal_control"], mode='lines',
            name='u*(t) — optymalne sterowanie',
            line=dict(color='#e040fb', width=2)))
        fig_ctrl.add_hline(y=0, line_dash='dash', line_color='gray')
        fig_ctrl.update_layout(
            title="Optymalny Sygnał Transakcji u*(t)",
            xaxis_title="Czas", yaxis_title="Intensywność transakcji",
            template="plotly_dark", height=250)
        st.plotly_chart(fig_ctrl, use_container_width=True)
        avg_tracking_error = float(np.mean((mfg["agent_path"] - mfg["mean_field"])**2))
        st.metric("Średni Błąd Śledzenia (MSE)", f"{avg_tracking_error:.4f}",
                  help="Jak daleko agent był od mean field przez cały horyzont")

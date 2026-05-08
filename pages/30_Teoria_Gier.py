import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.integrate import odeint
from modules.styling import apply_styling, module_header
from modules.i18n import t

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
    "🤝 Matching Theory"
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
    st.header("8. Matching Theory — Algorytm Gale-Shapley")
    st.markdown("Jak połączyć dwie grupy (np. Inwestorów i Startupy) tak, by nikt nie chciał 'uciec' do kogoś innego? To problem **Stabilnego Dopasowania (Stable Marriage Problem)**.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Preferencje")
        st.markdown("""
        Mamy 3 Inwestorów (I1, I2, I3) i 3 Startupy (S1, S2, S3).
        Każdy ma listę rankingową partnerów.
        """)
        # Mock Data for simplicity
        st.write("Inwestor I1 preferuje: S1 > S2 > S3")
        st.write("Startup S1 preferuje: I2 > I1 > I3")
        
        if st.button("Uruchom Algorytm Gale-Shapley"):
            st.info("Algorytm: Inwestorzy 'oświadczają się' startupom. Startupy trzymają najlepszą ofertę 'na później' (deferred acceptance).")
            st.success("Dopasowanie Stabilne: (I1-S2), (I2-S1), (I3-S3)")
            st.caption("To dopasowanie jest optymalne dla strony proponującej (Inwestorów).")

    with col2:
        st.subheader("Zastosowanie w Finansach")
        st.markdown("""
        - **Dark Pools:** Dopasowanie dużych zleceń kupna/sprzedaży bez wpływu na rynek publiczny.
        - **HFT Matching Engines:** Kolejkowanie i parowanie transakcji.
        - **Private Equity:** Dobieranie limitowanych partnerów (LP) do funduszy (GP).
        """)
        
        # Diagram of matching flow
        fig_match = go.Figure()
        # Nodes
        fig_match.add_trace(go.Scatter(x=[1, 1, 1], y=[3, 2, 1], mode="markers+text", text=["I1", "I2", "I3"], textposition="middle left", name="Inwestorzy", marker=dict(size=20, color="#3498db")))
        fig_match.add_trace(go.Scatter(x=[2, 2, 2], y=[3, 2, 1], mode="markers+text", text=["S1", "S2", "S3"], textposition="middle right", name="Startupy", marker=dict(size=20, color="#00e676")))
        # Edges (Matching example)
        fig_match.add_trace(go.Scatter(x=[1, 2, None, 1, 2, None, 1, 2], y=[3, 2, None, 2, 3, None, 1, 1], mode="lines", line=dict(color="#aaa", width=2), showlegend=False))
        
        fig_match.update_layout(title="Dopasowanie Inwestorzy ↔ Startupy", xaxis=dict(visible=False), yaxis=dict(visible=False), template="plotly_dark", height=400)
        st.plotly_chart(fig_match, use_container_width=True)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats
from sklearn.decomposition import TruncatedSVD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

st.set_page_config(page_title="Kalkulator Bayesa & Statystyka", page_icon="🧮", layout="wide")

st.title("🧮 Zaawansowany Kalkulator Bayesa")
st.markdown("""
Ten moduł to potężne narzędzie do analizy prawdopodobieństwa i wnioskowania statystycznego. 
Łączy w sobie klasyczne **Twierdzenie Bayesa** z fundamentalnymi prawami statystyki i uczenia maszynowego.
Został zaprojektowany, aby dostarczyć wysokiej jakości wizualizacje pomagające zrozumieć trudne koncepty matematyczne.
""")

tabs = st.tabs([
    "🎲 Dyskretny Bayes", 
    "📈 Bayes Ciągły", 
    "📊 Prawa Graniczne", 
    "🌐 Niezależność (Wielowymiarowy)", 
    "🧠 Zaawansowane Modele ML"
])

# --- Tab 1: Dyskretny Bayes ---
with tabs[0]:
    st.header("1. Dyskretny Kalkulator Bayesa")
    st.markdown("""
    Wykorzystuje **Twierdzenie Bayesa** i **Twierdzenie o Prawdopodobieństwie Całkowitym** dla wielu hipotez i dowodów sekwencyjnych.
    """)
    st.latex(r"P(H_i | E) = \frac{P(E | H_i) P(H_i)}{\sum_j P(E | H_j) P(H_j)}")
    
    with st.expander("📖 Jak z tego korzystać i co to oznacza?"):
        st.markdown("""
        **Co tu się dzieje?**
        Ten moduł pozwala obliczyć, jak nowe dowody (obserwacje) zmieniają nasze zaufanie do określonych hipotez.
        - **Liczba hipotez (H):** Wybierz, ile różnych możliwości rozważasz (np. 2: Pacjent jest chory lub zdrowy).
        - **Prior P(H):** Twoje wyjściowe przekonanie, że dana hipoteza jest prawdziwa, ZANIM zobaczysz jakiekolwiek dowody (np. występowanie choroby w populacji wynosi 1%).
        - **Likelihood P(E | H):** Prawdopodobieństwo zaobserwowania danego dowodu, przy założeniu, że hipoteza jest prawdziwa (np. jeśli pacjent jest chory, test wyjdzie pozytywnie w 99% przypadków).
        - **Posterior P(H|E):** Zaktualizowane prawdopodobieństwo hipotez po uwzględnieniu dowodu (wynik obliczeń na pomarańczowo na wykresie).
        
        **Jak używać?** 
        Zmieniaj suwaki "Prior" i "Likelihood" dla różnych hipotez. Obserwuj na wykresie, jak mocny dowód (wysoki Likelihood dla jednej hipotezy, a niski dla innych) drastycznie przesuwa prawdopodobieństwo Posterior, nawet jeśli Prior był początkowo bardzo mały.
        """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Konfiguracja Modelu")
        num_hypotheses = st.number_input("Liczba konkurencyjnych hipotez (H)", min_value=2, max_value=10, value=2)
        
        st.markdown("**Prawdopodobieństwa A Priori $P(H)$**")
        priors = []
        for i in range(num_hypotheses):
            p = st.slider(f"Prior P(H{i+1})", 0.0, 1.0, 1.0/num_hypotheses, key=f"prior_{i}")
            priors.append(p)
            
        # Normalize priors if needed
        total_prior = sum(priors)
        if total_prior > 0 and total_prior != 1.0:
            st.info(f"Normalizuję priory (suma wynosiła {total_prior:.2f}).")
            priors = [p / total_prior for p in priors]
            
        st.markdown("**Obserwacja (Dowód E): Wiarygodność (Likelihood) $P(E | H)$**")
        likelihoods = []
        for i in range(num_hypotheses):
            l = st.slider(f"Likelihood P(E | H{i+1})", 0.0, 1.0, 0.5, key=f"likelihood_{i}")
            likelihoods.append(l)
            
    with col2:
        # Calculate posterior using Total Probability
        evidence_prob = sum(p * l for p, l in zip(priors, likelihoods))
        if evidence_prob > 0:
            posteriors = [(p * l) / evidence_prob for p, l in zip(priors, likelihoods)]
        else:
            posteriors = priors # fall back
            
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Hipoteza {i+1}" for i in range(num_hypotheses)],
            y=priors,
            name='A Priori P(H)',
            marker_color='rgba(55, 128, 191, 0.6)',
            marker_line_color='rgba(55, 128, 191, 1.0)',
            marker_line_width=2
        ))
        fig.add_trace(go.Bar(
            x=[f"Hipoteza {i+1}" for i in range(num_hypotheses)],
            y=posteriors,
            name='A Posteriori P(H|E)',
            marker_color='rgba(255, 153, 51, 0.8)',
            marker_line_color='rgba(255, 153, 51, 1.0)',
            marker_line_width=2
        ))
        fig.update_layout(
            barmode='group', 
            title="Aktualizacja Prawdopodobieństw (Wpływ dowodu)",
            template="plotly_dark", 
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title='Prawdopodobieństwo', range=[0, 1]),
            legend=dict(x=0.01, y=0.99)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Twierdzenie o Prawdopodobieństwie Całkowitym:** Prawdopodobieństwo zaobserwowania dowodu E wynosi (mianownik): **{evidence_prob:.4f}**")

# --- Tab 2: Continuous Bayes ---
with tabs[1]:
    st.header("2. Ciągły Kalkulator Bayesa (Rozkłady Sprzężone)")
    st.markdown("""
    W statystyce bayesowskiej często pracujemy z całymi rozkładami. Wykorzystuje się tu **regułę Bayesa dla rozkładów**:
    """)
    st.latex(r"\text{Posterior} \propto \text{Likelihood} \times \text{Prior}")
    
    st.subheader("Przykład: Rozkład Beta-Dwumianowy (Beta-Binomial)")
    st.markdown("Często używany do modelowania wskaźników konwersji (np. współczynnik klikalności CTR) lub rzutów monetą.")
    
    with st.expander("📖 Jak z tego korzystać i co to oznacza?"):
        st.markdown("""
        **Co tu się dzieje?**
        W statystyce Bayesowskiej często nie szukamy jednej konkretnej wartości parametru (np. dokładnego "0.5" dla rzutu monetą), ale całego *rozkładu* możliwych wartości (krzywa pokazująca, w które wartości wierzymy najbardziej).
        - **Alpha i Beta (Prior):** Zamiast podawać "wstępne prawdopodobieństwo", podajesz swoje wcześniejsze "sukcesy i porażki". Jeśli ustawisz Alpha=2 i Beta=2, mówisz modelowi: "wierzę, że na początku mieliśmy po 2 sukcesy i porażki, więc szansa to ok. 50%, ale jestem dość niepewny" (szeroka krzywa szara).
        - **Sukcesy i Porażki (Obserwacje):** To nowe, twarde dane z eksperymentu. (Krzywa przerywana pomarańczowa).
        - **Posterior (Jasnoniebieska strefa):** Wynik połączenia Prioru z Danymi. 
        
        **Jak używać?**
        1. Ustaw silny Prior (np. Alpha=50, Beta=50). Krzywa szara będzie bardzo wąska wokół 0.5.
        2. Ustaw małą liczbę obserwacji, np. 5 sukcesów, 0 porażek. Zauważ, że Posterior (jasnoniebieski) ledwo się przesunie, bo silny Prior dominuje nad słabymi danymi.
        3. Ustaw ogromną liczbę obserwacji, np. 100 sukcesów, 0 porażek. Teraz dane "przekrzyczą" Prior i krzywa przesunie się bardzo blisko wartości 1.0. Tak działa uczenie maszynowe z użyciem Bayesa!
        """)
        
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("**Prior: Rozkład Beta**")
        st.caption("Nasze wcześniejsze przekonanie o parametrze sukcesu. Alpha=sukcesy, Beta=porażki.")
        alpha_prior = st.slider("Alpha a priori (Wzmocnienie Sukcesu)", 0.1, 50.0, 2.0, 0.1)
        beta_prior = st.slider("Beta a priori (Wzmocnienie Porażki)", 0.1, 50.0, 2.0, 0.1)
        
        st.markdown("**Dane (Obserwacje): Rozkład Dwumianowy**")
        successes = st.slider("Zaobserwowane sukcesy", 0, 100, 10)
        failures = st.slider("Zaobserwowane porażki", 0, 100, 5)
        
    with col2:
        # Analitczne rozwiązanie rozkładu sprzężonego
        alpha_post = alpha_prior + successes
        beta_post = beta_prior + failures
        
        x = np.linspace(0, 1, 500)
        y_prior = stats.beta.pdf(x, alpha_prior, beta_prior)
        y_post = stats.beta.pdf(x, alpha_post, beta_post)
        
        # Likelihood (scaled to match chart heights for visual comparison)
        n_trials = successes + failures
        if n_trials > 0:
            y_like = x**successes * (1-x)**failures
            # scaling
            y_like = y_like / np.max(y_like) * np.max(y_post) 
        else:
            y_like = np.zeros_like(x)
            
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x, y=y_prior, mode='lines', name='Prior (Beta)', line=dict(color='gray', width=3, dash='dash')))
        fig2.add_trace(go.Scatter(x=x, y=y_like, mode='lines', name='Likelihood (Dopasowane)', line=dict(color='orange', width=2, dash='dot')))
        fig2.add_trace(go.Scatter(x=x, y=y_post, mode='lines', name='Posterior (Beta)', fill='tozeroy', line=dict(color='cyan', width=3)))
        
        fig2.update_layout(
            title="Ewolucja Przekonania (Beta-Binomial Conjugate Prior)", 
            template="plotly_dark",
            xaxis_title="Prawdopodobieństwo sukcesu (θ)",
            yaxis_title="Gęstość Prawdopodobieństwa (PDF)"
        )
        st.plotly_chart(fig2, use_container_width=True)

# --- Tab 3: Limit Theorems ---
with tabs[2]:
    st.header("3. Prawa Graniczne i Zbieżność")
    st.markdown("Eksperymentalne potwierdzenie **Centralnego Twierdzenia Granicznego (CTG)** i **Prawa Wielkich Liczb (PWL)**.")
    
    with st.expander("📖 Jak z tego korzystać i co to oznacza?"):
        st.markdown("""
        **Co tu się dzieje?**
        Te dwa prawa to absolutne fundamenty pozwalające nam w ogóle uprawiać statystykę. Mówią one o tym, że chaos (losowość) na dużą skalę staje się przewidywalny.
        
        **Jak używać Centralnego Twierdzenia Granicznego (Po lewej):**
        - Zmieniaj *Rozmiar pojedynczej próby (n)*. Obserwuj, co się dzieje, gdy zmieniasz z n=1 na n=30.
        - Zobaczysz, że rozkład dla n=1 (surowe zmienne jednostajne) jest bardzo płaski. 
        - Gdy zwiększasz n, bierzesz średnią z wielu pomiarów. Średnie te (pomimo że pochodzą z płaskiego rozkładu) zaczynają formować idealny "dzwon Gaussa" (rozkład normalny). To dlatego rozkład normalny jest wszędzie w naturze!
        
        **Jak używać Prawa Wielkich Liczb (Po prawej):**
        - Zmieniaj *Prawdziwe prawdopodobieństwo* oraz *Liczbę prób*.
        - Wykres pokazuje "średnią kumulatywną" z rzutów (np. monetą, gdzie szansa na sukces P=0.5).
        - Zauważ, że na początku (np. przy pierwszych 10 rzutach) zielona linia dziko skacze – to przypadek i wariancja. Ale im więcej masz prób (N > 200), zielona linia nieubłaganie przykleja się do czerwonej przerywanej linii (prawdziwej wartości). To gwarancja, że z odpowiednio dużą ilością danych, estymator zawsze znajdzie prawdę.
        """)
    
    col_clt, col_lln = st.columns(2)
    with col_clt:
        st.subheader("Centralne Twierdzenie Graniczne (CTG)")
        st.markdown("Suma lub średnia dużej liczby zmiennych z dowolnego rozkładu zmierza do **rozkładu normalnego**.")
        sample_size = st.slider("Rozmiar pojedynczej próby (n)", 1, 500, 30)
        num_experiments = st.slider("Liczba eksperymentów", 100, 5000, 1000)
        
        # Generowanie z rozkładu jednostajnego
        samples = np.random.uniform(0, 1, (num_experiments, sample_size))
        sample_means = np.mean(samples, axis=1)
        
        fig_clt = px.histogram(
            sample_means, nbins=50, 
            title=f"Rozkład średnich z prób (Zbiega do krzywej dzwonowej)", 
            template="plotly_dark", 
            color_discrete_sequence=['#ab63fa']
        )
        fig_clt.update_layout(showlegend=False, xaxis_title="Wartość średniej z próby", yaxis_title="Częstotliwość")
        st.plotly_chart(fig_clt, use_container_width=True)
        
    with col_lln:
        st.subheader("Prawo Wielkich Liczb (PWL)")
        st.markdown("Średnia z próby zbiega do prawdziwej wartości oczekiwanej przy rosnącej liczbie obserwacji.")
        n_flips = st.slider("Liczba prób (N)", 10, 2000, 500)
        prob_success = st.slider("Prawdziwe prawdopodobieństwo", 0.1, 0.9, 0.5)
        
        flips = np.random.binomial(1, prob_success, n_flips)
        running_mean = np.cumsum(flips) / np.arange(1, n_flips + 1)
        
        fig_lln = go.Figure()
        fig_lln.add_trace(go.Scatter(x=np.arange(1, n_flips+1), y=running_mean, mode='lines', name='Średnia empiryczna', line=dict(color='#00CC96')))
        fig_lln.add_trace(go.Scatter(x=[1, n_flips], y=[prob_success, prob_success], mode='lines', name='Wartość oczekiwana (Prawda)', line=dict(color='red', dash='dash')))
        fig_lln.update_layout(
            title="Zbieżność średniej z próby", 
            template="plotly_dark", 
            yaxis_range=[0, 1],
            xaxis_title="Liczba obserwacji (N)",
            yaxis_title="Średnia kumulatywna"
        )
        st.plotly_chart(fig_lln, use_container_width=True)

# --- Tab 4: Multivariate / Naive Bayes ---
with tabs[3]:
    st.header("4. Bayes Wielowymiarowy i Reguła Łańcuchowa")
    st.markdown("""
    **Twierdzenie o mnożeniu prawdopodobieństw** dla zdarzeń niezależnych stanowi fundament algorytmów takich jak **Naiwny Klasyfikator Bayesa**.
    Zakłada on, że cechy są warunkowo niezależne od klasy.
    """)
    st.latex(r"P(C | X_1, X_2) \propto P(C) \cdot P(X_1 | C) \cdot P(X_2 | C)")
    
    st.markdown("Zbadaj wpływ poszczególnych cech na klasyfikację dokumentu (np. Spam/Ham). Klasa 1 to Spam.")
    
    with st.expander("📖 Jak z tego korzystać i co to oznacza?"):
        st.markdown("""
        **Co tu się dzieje?**
        Wielowymiarowy Bayes pozwala nam łączyć wiele *niezależnych* wskazówek, aby podjąć jedną decyzję. Naiwny Klasyfikator Bayesa wykorzystuje to do klasyfikacji tekstu (np. filtry spam). Nazywa się "naiwny", bo zakłada, że wystąpienie jednego słowa nie zależy od wystąpienia drugiego (co w języku ludzkim nie jest prawdą, ale w praktyce taki algorytm działa świetnie).
        
        **Parametry (Suwaki):**
        - **Prior Klasy P(C):** Jaki procent maili w ogóle to spam? (Ustaw nisko, np. na 0.2, jeśli większość maili masz czystych).
        - **Słowo X1 i X2:** Jak często dane słowo występuje w zwykłych mailach (Ham, C=0) a jak często w Spamie (C=1). Ustalasz to z osobna.
        
        **Jak używać?**
        1. Ustaw bardzo wysokie "P('Darmowe' | C=1)" (np. 0.9) - oznaczając, że słowo "Darmowe" w 90% przypadków pojawia się w typowym spamie.
        2. Ustaw niskie "P('Darmowe' | C=0)" (np. 0.05) - oznaczając, że takie słowo rzadko występuje w zwykłym mailu.
        3. Obserwuj dolny wykres wynikowy. Pokazuje on sytuację: Co się stanie, jeśli właśnie przyszedł DO CIEBIE mail, w którym SĄ OBA te słowa (X1=1, X2=1)? Zobaczysz, jak niebieski (Prior) słupek zwykłego maila maleje na rzecz czerwonego (Posterior Spamu), całkowicie przechylając szalę na korzyść Spamu.
        """)
        
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Prior Klasy P(C)**")
        prior_c1 = st.slider("P(C=1) (Spam)", 0.0, 1.0, 0.5)
        prior_c0 = 1 - prior_c1
        
    with col2:
        st.markdown("**Słowo X1 ('Kredyt'): P(X1=1 | C)**")
        p_x1_c0 = st.slider("P('Kredyt' | C=0)", 0.0, 1.0, 0.1)
        p_x1_c1 = st.slider("P('Kredyt' | C=1)", 0.0, 1.0, 0.8)
        
    with col3:
        st.markdown("**Słowo X2 ('Darmowe'): P(X2=1 | C)**")
        p_x2_c0 = st.slider("P('Darmowe' | C=0)", 0.0, 1.0, 0.2)
        p_x2_c1 = st.slider("P('Darmowe' | C=1)", 0.0, 1.0, 0.9)
        
    st.divider()
    st.subheader("Wynik dla maila zawierającego słowa 'Kredyt' i 'Darmowe' (X1=1, X2=1)")
    
    evidence_c0 = prior_c0 * p_x1_c0 * p_x2_c0
    evidence_c1 = prior_c1 * p_x1_c1 * p_x2_c1
    total_evidence = evidence_c0 + evidence_c1
    
    post_c0 = evidence_c0 / total_evidence if total_evidence > 0 else 0
    post_c1 = evidence_c1 / total_evidence if total_evidence > 0 else 0
    
    fig_nb = go.Figure(data=[
        go.Bar(name='Prior P(C)', x=['Ham (C=0)', 'Spam (C=1)'], y=[prior_c0, prior_c1], marker_color='#4A90E2'),
        go.Bar(name='Posterior po ujrzeniu maila', x=['Ham (C=0)', 'Spam (C=1)'], y=[post_c0, post_c1], marker_color='#E02020')
    ])
    fig_nb.update_layout(
        barmode='group', 
        title="Działanie Naiwnego Klasyfikatora Bayesa", 
        template="plotly_dark",
        yaxis_title="Prawdopodobieństwo",
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig_nb, use_container_width=True)

# --- Tab 5: Advanced Models ---
with tabs[4]:
    st.header("5. Zaawansowane Modele i Twierdzenia (Fundamenty ML)")
    
    adv_tabs = st.tabs(["💡 SVD (Rozkład Spektralny)", "🎯 Representer Theorem (GP)", "📈 Warunki KKT", "🔄 De Finetti"])
    
    with adv_tabs[0]:
        st.subheader("Twierdzenie o rozkładzie spektralnym (SVD)")
        st.markdown("""
        Wykorzystywane w kompresji obrazu, systemach rekomendacyjnych i redukcji wymiarów (PCA).
        SVD dekomponuje macierz na ortogonalne kierunki największej wariancji danych.
        """)
        
        with st.expander("📖 Jak z tego korzystać i co to oznacza?"):
            st.markdown("""
            **Co tu się dzieje?**
            Masz zbiór danych osadzony w 3 wymiarach (wyobraź to sobie jako wiek, zarobki i wydatki ludzi). Cechy te w prawdziwym świecie są często ze sobą skorelowane (tworząc chmurę przypominającą elipsoidę). Zamiast analizować ciężki zbiór 3-parametrowy, możemy "skompresować" dane do najpłaskiego rzutu 2D, tracąc absolutne minimum informacji.
            
            **Jak używać?**
            1. Zmień liczbę punktów, by zobaczyć jak zachowują się gęstsze "chmury".
            2. Poeksperymentuj obracając lewy wykres myszką w środowisku 3D. Możesz znaleźć taki kąt nachylenia "cygara", z którego wygląda płasko.
            3. Spójrz na Prawy Wykres (2D). To jest właśnie zmatematyzowany "cień" - rzut skomplikowanej chmury 3D na 2 wyliczone matematycznie główne osie (tzw. "Główne Składowe") za pomocą operacji macierzowej SVD. To najefektywniejszy sposób na to, by rysować trudne problemy w 2 wymiarach!
            """)
            
        n_points = st.slider("Liczba punktów w przestrzeni 3D", 50, 1000, 300)
        
        # Generowanie skorelowanych danych 3D (elipsoida)
        np.random.seed(42)
        cov = [[1, 0.9, 0.6], [0.9, 1, 0.4], [0.6, 0.4, 1]]
        data_3d = np.random.multivariate_normal([0,0,0], cov, n_points)
        
        # Zastosowanie SVD (PCA)
        svd = TruncatedSVD(n_components=2)
        data_2d = svd.fit_transform(data_3d)
        
        col_3d, col_2d = st.columns(2)
        with col_3d:
            fig_3d = px.scatter_3d(x=data_3d[:,0], y=data_3d[:,1], z=data_3d[:,2], title="Oryginalna Elipsoida 3D", template="plotly_dark")
            fig_3d.update_traces(marker=dict(size=3, color='#17BECF'))
            fig_3d.update_layout(scene=dict(xaxis_title='X1', yaxis_title='X2', zaxis_title='X3'), margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig_3d, use_container_width=True)
            
        with col_2d:
            fig_2d = px.scatter(x=data_2d[:,0], y=data_2d[:,1], title="Kompresja do Płaszczyzny 2D (PCA/SVD)", template="plotly_dark")
            fig_2d.update_traces(marker=dict(color='#E377C2', size=5))
            fig_2d.update_layout(xaxis_title='Główna Składowa 1', yaxis_title='Główna Składowa 2', margin=dict(l=0, r=0, b=0, t=40))
            st.plotly_chart(fig_2d, use_container_width=True)

    with adv_tabs[1]:
        st.subheader("Twierdzenie o Przedstawieniu (Representer Theorem) - Proces Gaussa")
        st.markdown("""
        Mówi o tym, że optymalne rozwiązanie dla uczenia z użyciem metod jądrowych (Kernel Methods) leży w powłoce wyznaczonej przez dane treningowe. 
        **Proces Gaussa** to idealny bayesowski model pokazujący to zjawisko: im dalej od danych treningowych, tym "wstęga ufności" jest szersza (większa niepewność).
        """)
        
        with st.expander("📖 Jak z tego korzystać i co to oznacza?"):
            st.markdown("""
            **Co tu się dzieje?**
            Tradycyjna statystyka (jak regresja liniowa) często wymusza "sztywne" linie do dopasowania do punktów. Proces Gaussa (GP) oparty o metody jądrowe w ogóle nie narzuca sztywnego kształtu – on operuje na pojęciu "bliskości". Zamiast tego wyznacza bayesowską niepewność na całą przestrzeń funkcji. Wynik GP jest pewny swojej predykcji tam, gdzie leżą zaobserwowane dane, ale szczerze "przyznaje się do niewiedzy" w miejscach, gdzie nic nie widział.
            
            **Jak używać parametrów na lewym panelu?**
            1. Zmieniaj **Liczbę zaobserwowanych punktów**. Zobaczysz na wykresie, że wokół każdego *nowego czerwonego krzyżyka*, zielona wstęga niepewności (Przedział Ufności) gwałtownie "szczupleje" i zbliża się do punktu. W miejscach bez czerwonych krzyżyków (zwłaszcza ekstrapolacja po bokach), wstęga mocno się rozszerza.
            2. **Poziom szumu (Wariancja):** Ustaw tę wartość wyżej. W ten sposób mówisz modelowi: "czerwone punkty mogą być trochę niedokładne". Zauważysz, że GP "przestanie ufać" czerwonym krzyżykom całkowicie i wstęga rozszerzy się nawet w samych punktach (dane są traktowane z dystansem).
            3. **Szerokość Jądra:** Określa ona "pamięć" i to, jak płynna lub "gładka" jest nasza poszukiwana funkcja. Bardzo mała szerokość rzędu `0.1` sprawi, że model zacznie dziko "wibrować" między punktami zamiast je gładko łączyć.
            """)
        
        col_gp1, col_gp2 = st.columns([1, 3])
        with col_gp1:
            noise_level = st.slider("Poziom szumu (Wariancja)", 0.01, 1.0, 0.1)
            num_train_pts = st.slider("Liczba zaobserwowanych punktów", 2, 30, 6)
            kernel_length_scale = st.slider("Szerokość Jądra (Kernel RBF)", 0.1, 5.0, 1.0)
            
        with col_gp2:
            kernel = C(1.0, "fixed") * RBF(kernel_length_scale, "fixed")
            gp = GaussianProcessRegressor(kernel=kernel, alpha=noise_level**2, n_restarts_optimizer=0)
            
            # Generowanie "prawdziwej" funkcji i próbek
            np.random.seed(1)
            X_train = np.random.uniform(-5, 5, num_train_pts).reshape(-1, 1)
            y_train = np.sin(X_train).ravel() + np.random.normal(0, noise_level, num_train_pts)
            
            gp.fit(X_train, y_train)
            
            X_test = np.linspace(-6, 6, 200).reshape(-1, 1)
            y_pred, sigma = gp.predict(X_test, return_std=True)
            y_true = np.sin(X_test).ravel()
            
            fig_gp = go.Figure()
            fig_gp.add_trace(go.Scatter(x=X_test.ravel(), y=y_true, mode='lines', name='Prawdziwa funkcja (ukryta)', line=dict(color='gray', dash='dash')))
            fig_gp.add_trace(go.Scatter(x=X_test.ravel(), y=y_pred, mode='lines', name='Średnia predykcja modelu', line=dict(color='#00CC96', width=3)))
            fig_gp.add_trace(go.Scatter(x=X_test.ravel(), y=y_pred + 1.96 * sigma, mode='lines', line=dict(width=0), showlegend=False))
            fig_gp.add_trace(go.Scatter(x=X_test.ravel(), y=y_pred - 1.96 * sigma, mode='lines', fill='tonexty', fillcolor='rgba(0,204,150,0.2)', line=dict(width=0), name='95% Przedział Ufności'))
            fig_gp.add_trace(go.Scatter(x=X_train.ravel(), y=y_train, mode='markers', name='Zaobserwowane Dowody (Dane)', marker=dict(color='red', size=8, symbol='cross')))
            
            fig_gp.update_layout(title="Wnioskowanie z użyciem Procesu Gaussa", template="plotly_dark")
            st.plotly_chart(fig_gp, use_container_width=True)

    with adv_tabs[2]:
        st.subheader("Warunki Karusha-Kuhna-Tuckera (KKT)")
        st.markdown("""
        Powszechnie wykorzystywane w uczeniu maszynowym do znajdowania rozwiązań optymalizacyjnych z ograniczeniami. 
        Przykład: **Regresja z regularyzacją** (L1/L2) lub maszyny wektorów nośnych (SVM) działają w oparciu o KKT.
        """)
        st.info("💡 **Intuicja:** Optymalny punkt znajduje się w miejscu, gdzie kierunek największego spadku (gradient funkcji celu) jest równoważony przez kierunek wyznaczany przez więzy (nie pozwala pójść dalej).")
        
        with st.expander("📖 Jak z tego korzystać i co to oznacza?"):
            st.markdown("""
            **Co tu się dzieje?**
            Wyobraź sobie, że chcesz zejść na samo dno ogromnego krateru wulkanu. Im niżej zejdziesz (kolor fioletowy), tym mniejszą wartość osiągnie Twoja "funkcja optymalizacji". Ten wulkan to funkcja $f(x,y)=x^2+y^2$. Dno leży dokładnie w $x=0, y=0$. Niestety, w poprzek wulkanu ktoś postawił długi szklany mur (to właśnie ograniczenie, czerwona linia: $x+y=c$). Nie możesz go przekroczyć. Twoim zadaniem jest znalezienie matematycznie i algorytmicznie punktu tak głęboko w kraterze, jak to tylko możliwe, ale uderzając o ten mur, bez przechodzenia na drugą stronę. 
            
            Warunki KKT to zaawansowany wzór, który to rozwiązuje na masową skalę, będąc silnikiem potężnych modeli AI - Maszyn Wektorów Nośnych (SVM) stosowanych m.in. w medycynie.
            
            **Jak używać wykresu?**
            1. Środek ciemnych fioletowych kręgów to "złoty środek" bez więzów. To dno "krateru".
            2. Poruszaj suwakiem **Położenie więzów**. Przesuwasz w ten sposób ten twardy "szklany mur" (czerwoną linię ograniczenia).
            3. Żółta Gwiazdka to "Optimum wyliczone z równań KKT". Jest to najbardziej fioletowy odcień (dół lejka), w jakim możesz się prawnie znaleźć, zważając na czerwoną linię.
            4. Zwróć uwagę, że punkt KKT jest *zawsze styczny* – gwiazdka dotyka muru dokładnie tam, gdzie kształt czerwonej linii ociera się o okrąg izolinii krateru. 
            """)
        
        # Wizualizacja izoliniowa dla f(x,y)=x^2+y^2 i ograniczenia x+y>=c
        x_range = np.linspace(-3, 3, 100)
        y_range = np.linspace(-3, 3, 100)
        X_kkt, Y_kkt = np.meshgrid(x_range, y_range)
        Z_kkt = X_kkt**2 + Y_kkt**2 
        
        constraint_val = st.slider("Położenie więzów prostoliniowych (c)", -3.0, 3.0, 2.0)
        
        fig_kkt = go.Figure(data=go.Contour(z=Z_kkt, x=x_range, y=y_range, colorscale="Viridis", contours=dict(showlines=False)))
        
        # Linia ograniczenia x + y = c  => y = c - x
        y_c = constraint_val - x_range
        # Tylko wartości mieszczące się na wykresie
        mask = (y_c >= -3) & (y_c <= 3)
        fig_kkt.add_trace(go.Scatter(x=x_range[mask], y=y_c[mask], mode='lines', line=dict(color='red', width=4), name='Ograniczenie: x+y=c'))
        
        # Punkt optimum (x=c/2, y=c/2)
        opt_x, opt_y = constraint_val/2, constraint_val/2
        if -3 <= opt_x <= 3 and -3 <= opt_y <= 3:
            fig_kkt.add_trace(go.Scatter(x=[opt_x], y=[opt_y], mode='markers', name='Optimum KKT', marker=dict(color='yellow', size=12, symbol='star')))
            
        fig_kkt.update_layout(title="Optymalizacja KKT: Minimalizacja x²+y² z więzami", template="plotly_dark", height=600, width=800)
        st.plotly_chart(fig_kkt, use_container_width=False)

    with adv_tabs[3]:
        st.subheader("Twierdzenie graniczne de Finettiego")
        st.markdown("""
        **"Nieskończona wymienna sekwencja zmiennych losowych 0–1 jest mieszanką rozkładów Bernoulliego."**
        """)
        st.success("""
        To najgłębsze uzasadnienie teoretyczne dla budowania modeli bayesowskich. 
        Zamiast twardo zakładać, że rzuty monetą są całkowicie niezależne z nieznanym "prawdziwym" parametrem $p$, 
        Twierdzenie de Finettiego mówi: jeśli wierzysz, że **kolejność obserwacji nie ma znaczenia (są wymienne)**, 
        to matematycznie zachowują się one tak, jakby istniał ukryty rozkład prawdopodobieństwa (nasz Prior), 
        a obserwacje były warunkowo niezależne.
        """)
        st.markdown("""
        **Co to zmienia w praktyce?**
        - Uprawnia to stosowanie Priory na nieznanych parametrach (Traktujemy parametry jako **zmienne losowe**).
        - Wyjaśnia, dlaczego "uczymy się z doświadczenia": każda kolejna zaobserwowana próba aktualizuje nasze pojęcie o całej mieszance (Aktualizacja Posteriori).
        """)
        
        with st.expander("📖 Co to oznacza dla Ciebie jako inwestora/analityka finansowego?"):
            st.markdown("""
            **Jak wykorzystać tę wiedzę w praktyce inwestycyjnej?**
            Kiedy analizujesz giełdę (np. sprawdzając zachowania w Symulatorze tej aplikacji), zakładasz, że przeszłe stopy zwrotu pozwolą ocenić te przyszłe. Dlaczego to ma matematyczny sens?
            
            Podejście klasyczne (frekwentystyczne) twierdzi: "Rynek ma jakieś jedno, sztywne prawdopodobieństwo wzrostu p=53% a my próbujemy je odgadnąć". To błąd myślowy.
            
            Podejście wg Tw. de Finettiego i statystyki bayesowskiej stwierdza: 
            "Giełda nie jest rzutem jedną i tą samą monetą, ale mamy powody uważać, że dni rynkowe w obrębie danego *Reżimu Gospodarczego* wykazują wspólne cechy – są względem siebie *wymienialne*". Jeśli są wymienialne, istnieje dla nich pewien wewnętrzny, ukryty rozkład generujący. Każdy kolejny dzień rynkowy, który widzimy i przetwarzamy (nasza nowa informacja / Dowód), matematycznie pozwala nam coraz dokładniej "przemodelować" nasze wcześniejsze domniemania co do ukrytego mechanizmu rządzącego rynkiem (nasz Posterior). 
            
            Właśnie tak pod maską działają moduły SI i filtry Kalmana ukryte w zakładce Doradcy – traktują rynek jako ciągły strumień dowodów, aktualizując prawdopodobieństwa kolejnych ruchów po każdym nowym dzwonku giełdowym. 
            """)

"""
54_Factor_Zoo.py — Factor Zoo Multiple Testing Analysis
Strona Streamlit dla korekty na multiple testing w analizie faktorów.
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Factor Zoo — Multiple Testing",
    page_icon="🦁",
    layout="wide",
)

st.markdown("""
<style>
    .main { background: #0a0b14; }
    .block-container { padding: 1.5rem 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #12131f, #1a1b2e);
        border: 1px solid #2a2b3d;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 6px 0;
    }
    .significant { border-left: 4px solid #00e676; }
    .not-sig     { border-left: 4px solid #e74c3c; }
    h1 { color: #e2e4f0; font-family: 'Inter', sans-serif; font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🦁 Factor Zoo — Multiple Testing Analysis")
st.markdown("""
> *"Jeśli testujesz 100 faktorów przy α=5%, oczekujesz ~5 fałszywie istotnych wyników
> nawet jeśli żaden faktor nie jest naprawdę istotny."*
> — Harvey, Liu & Zhu (2016), Journal of Finance

Strona koryguje t-statystyki faktorów na problem **Data Snooping Bias**.
""")
st.divider()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Ustawienia")
    correction_method = st.selectbox(
        "Metoda korekcji",
        ["BHY", "bonferroni", "holm", "storey", "harvey_liu"],
        help="""
        BHY: Benjamini-Hochberg-Yekutieli (zalecana dla skorelowanych testów)
        Bonferroni: Konserwatywna, FWER
        Holm: Step-down Bonferroni
        Storey: Adaptacyjna FDR
        Harvey-Liu: Specyficzna dla Factor Zoo (t*≈3.0)
        """
    )
    alpha = st.slider("Poziom istotności α", 0.01, 0.10, 0.05, 0.01)
    st.divider()
    st.markdown("### 📊 Demo: Własne t-statystyki")
    use_demo = st.checkbox("Użyj danych demonstracyjnych", value=True)

    if not use_demo:
        st.markdown("**Wpisz t-statystyki faktorów (jeden per linia):**")
        t_input = st.text_area(
            "t-statystyki",
            value="3.5\n2.8\n1.9\n4.2\n1.7\n2.1\n0.8\n3.1\n1.5\n2.4",
            height=200,
        )
        factor_names_input = st.text_area(
            "Nazwy faktorów",
            value="Momentum\nSize\nValue\nProfitability\nInvestment\nQuality\nLiquidity\nLow_Vol\nBAB\nROE",
            height=200,
        )

    st.divider()
    st.markdown("### 📖 Harvey, Liu & Zhu (2016)")
    with st.expander("Pokaż kluczowe wnioski"):
        st.markdown("""
        **Kluczowe wnioski HLZ (2016):**

        - W literaturze finansowej przetestowano **300+ faktorów**
        - Przy standardowym α=5%: ~15 fałszywych odkryć
        - **Zalecany minimalny t*=3.0** dla nowych faktorów (2016)
        - Dla faktorów publikowanych przed 2005: t*=2.0
        - Po 2005: t*=3.0 (data mining eksplozja)

        *"The recommended t-statistic cutoff for new factors
        is 3.0, substantially higher than the conventional 2.0"*
        — Harvey, Liu & Zhu (JF 2016)
        """)

# ─── Dane demonstracyjne ─────────────────────────────────────────────────────
DEMO_FACTORS = [
    # (name, t_stat, year_discovered, source)
    ("Market Beta",        1.82, 1964, "Sharpe/Lintner"),
    ("Size (SMB)",         2.15, 1981, "Banz"),
    ("Value (HML)",        3.12, 1992, "Fama-French"),
    ("Momentum (WML)",     4.01, 1993, "Jegadeesh-Titman"),
    ("Profitability (RMW)",2.89, 2015, "Fama-French 5F"),
    ("Investment (CMA)",   2.41, 2015, "Fama-French 5F"),
    ("Low Volatility",     2.78, 2006, "Baker-Bradley"),
    ("Quality",            2.23, 2014, "Novy-Marx"),
    ("BAB (Betting Ag.)",  3.45, 2014, "Frazzini-Pedersen"),
    ("Accruals",           1.95, 1996, "Sloan"),
    ("Dividend Yield",     1.41, 1978, "Rosenberg"),
    ("Short Interest",     3.21, 2004, "Dechow et al."),
    ("Idiosyncratic Vol",  1.67, 2006, "Ang et al."),
    ("Asset Growth",       2.34, 2008, "Cooper et al."),
    ("ROE",                2.56, 1995, "Haugen-Baker"),
    ("Cash Flow Yield",    1.88, 1988, "Chan-Hamao-Lakonishok"),
    ("Gross Profit/Assets",2.95, 2013, "Novy-Marx"),
    ("R&D/Assets",         1.72, 1999, "Chan et al."),
    ("Earnings Surprise",  2.12, 1968, "Ball-Brown"),
    ("Net Issuance",       2.43, 2004, "Pontiff-Woodgate"),
    ("Turn-of-Month",      1.53, 1988, "Ariel"),
    ("January Effect",     1.76, 1976, "Rozeff-Kinney"),
    ("Post-Earnings Drift",2.08, 1968, "Ball-Brown"),
    ("Liquidity",          1.94, 2002, "Pastor-Stambaugh"),
    ("Tail Risk Beta",     1.38, 2014, "Kelly-Jiang"),
]

if use_demo:
    t_stats = np.array([f[1] for f in DEMO_FACTORS])
    factor_names = [f[0] for f in DEMO_FACTORS]
    factor_meta = pd.DataFrame(DEMO_FACTORS, columns=["Factor", "t_stat", "Year", "Source"])
else:
    try:
        t_stats = np.array([float(x.strip()) for x in t_input.split("\n") if x.strip()])
        names_list = [x.strip() for x in factor_names_input.split("\n") if x.strip()]
        factor_names = names_list[:len(t_stats)]
        factor_meta = None
    except Exception:
        st.error("Błąd parsowania t-statystyk. Sprawdź format (jeden float per linia).")
        st.stop()

# ─── Analiza ─────────────────────────────────────────────────────────────────
with st.spinner("Koryguję na multiple testing..."):
    from modules.factor_significance import correct_for_multiple_testing, plot_factor_significance

    result = correct_for_multiple_testing(
        t_stats,
        n_obs=252 * 30,  # ~30 lat danych (typowe w literaturze)
        method=correction_method,
        alpha=alpha,
        return_all=True,
    )

# ─── Metryki główne ───────────────────────────────────────────────────────────
st.markdown("## 📊 Wyniki Analizy")

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("Faktorów łącznie", result["n_factors"])
with m2:
    n_sig = result["n_significant"]
    n_raw = int((result["p_values_raw"] <= alpha).sum())
    st.metric("Istotnych po korekcji", n_sig,
              delta=f"{n_sig - n_raw} vs bez korekcji",
              delta_color="inverse")
with m3:
    st.metric("Fałszywych odkryć (estym.)", f"{result['n_false_positives_expected']:.1f}")
with m4:
    st.metric("Szacowane FDR", f"{result['false_discovery_rate']:.1%}")
with m5:
    st.metric("Próg |t|", f"{result['threshold_tstat']:.2f}",
              help="Minimalna t-statystyka dla istotności po korekcji")

# ─── Główny wykres ────────────────────────────────────────────────────────────
st.divider()
factor_df = pd.DataFrame({
    "factor":     factor_names[:len(t_stats)],
    "t_stat":     t_stats,
    "significant":result["significant"],
    "p_raw":      result["p_values_raw"],
    "p_adj":      result["adjusted_pvalues"],
})

fig = plot_factor_significance(
    factor_df,
    threshold_t=result["threshold_tstat"],
    title=f"Factor Zoo — {correction_method} Correction (α={alpha})",
)
st.plotly_chart(fig, use_container_width=True)

# ─── Tabela wyników ───────────────────────────────────────────────────────────
st.markdown("## 📋 Szczegółowe Wyniki")

display_df = factor_df.copy()
display_df["Istotny po korekcji"] = display_df["significant"].map({True: "✅ Tak", False: "❌ Nie"})
display_df["p_raw"] = display_df["p_raw"].round(4)
display_df["p_adj"] = display_df["p_adj"].round(4)
display_df = display_df.rename(columns={
    "factor": "Faktor",
    "t_stat": "|t| statystyka",
    "p_raw": "p-value (surowe)",
    "p_adj": f"p-value ({correction_method})",
})

# Kolorowanie
def color_row(row):
    if row.get("✅ Tak" if "Istotny po korekcji" in row else False, False):
        return ["background-color: rgba(0,230,118,0.08)"] * len(row)
    return [""] * len(row)

st.dataframe(
    display_df[["Faktor", "|t| statystyka", "p-value (surowe)", f"p-value ({correction_method})", "Istotny po korekcji"]],
    use_container_width=True,
    hide_index=True,
)

# ─── Porównanie metod ─────────────────────────────────────────────────────────
if "all_methods" in result:
    st.divider()
    st.markdown("## 🔄 Porównanie Metod Korekcji")
    cmp_data = {
        "Metoda": ["Bez korekcji", "BHY (zalecana)", "Bonferroni", "Holm"],
        "Istotnych faktorów": [
            int((result["p_values_raw"] <= alpha).sum()),
            result["all_methods"].get("BHY", {}).get("n_significant", "N/D"),
            result["all_methods"].get("bonferroni", {}).get("n_significant", "N/D"),
            result["all_methods"].get("holm", {}).get("n_significant", "N/D"),
        ],
        "Komentarz": [
            "❌ Inflacja false positives",
            "✅ Kontroluje FDR (zalecana dla finansów)",
            "⚠️ Zbyt konserwatywna (za mało mocy)",
            "⚠️ Kompromis FWER/moc",
        ]
    }
    st.dataframe(pd.DataFrame(cmp_data), use_container_width=True, hide_index=True)

# ─── Edukacja ─────────────────────────────────────────────────────────────────
st.divider()
with st.expander("📚 Metodologia — Dlaczego BHY?"):
    st.markdown("""
    ### Hierarchia metod korekcji

    | Metoda | Kontroluje | Moc | Zalecana gdy |
    |--------|-----------|-----|-------------|
    | **BHY** | FDR (przy zależnych testach) | Wysoka | Faktory finansowe (skorelowane) |
    | **Storey q-value** | Adaptacyjne FDR | Najwyższa | Wiele testów, mało sygnału |
    | **Holm** | FWER (step-down) | Umiarkowana | Kilka hipotez kluczowych |
    | **Bonferroni** | FWER (kontrola) | Niska | Bardzo konserwatywna granica |
    | **Harvey-Liu** | Factor Zoo bias | Średnia | Nowe faktory finansowe |

    ### Kluczowa różnica: FWER vs FDR
    - **FWER** (Family-Wise Error Rate): P(choć jeden false positive) ≤ α
    - **FDR** (False Discovery Rate): E[false positives / all positives] ≤ α

    → FDR jest mniej konserwatywny, daje więcej mocy, odpowiedni dla eksploracyjnej analizy faktorów.

    *Ref: Benjamini & Yekutieli (2001), Harvey, Liu & Zhu (2016)*
    """)

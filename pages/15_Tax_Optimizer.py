"""15_Tax_Optimizer.py — Optymalizator Podatkowy (Belka, IKE/IKZE)"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.styling import apply_styling
from modules.tax_optimizer_pl import (
    create_position, tax_loss_harvesting, ike_ikze_optimizer,
    annual_belka_estimate, IKE_LIMIT_PLN, IKZE_LIMIT_PLN, TAX_BELKA,
    ppk_simulator, asset_location_optimizer
)
from modules.i18n import t

st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown("# 💰 Tax Optimizer PL")
st.markdown("*Podatek Belka, Tax Loss Harvesting, IKE/IKZE — bezpieczny zysk dla polskiego inwestora*")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 IKE / IKZE Kalkulator", "✂️ Tax Loss Harvesting", "📊 Roczna Belka (PIT-38)", "🏢 PPK vs Prywatnie", "🗂️ Asset Location"])

with tab1:
    st.markdown("### 💚 Kalkulator Oszczędności IKE / IKZE")
    col1, col2 = st.columns(2)
    with col1:
        ike_funded = st.number_input("Wpłacono na IKE w tym roku (PLN)", 0, int(IKE_LIMIT_PLN), 0, 1000)
        ikze_funded = st.number_input("Wpłacono na IKZE w tym roku (PLN)", 0, int(IKZE_LIMIT_PLN), 0, 500)
        years_ret = st.slider("Lat do emerytury", 5, 40, 20)
        cagr_ike = st.slider("Oczekiwany CAGR (%)", 3, 15, 8) / 100
    with col2:
        pit_rate = st.selectbox("Twoja stawka PIT", [0.17, 0.32], format_func=lambda x: f"{x:.0%}", index=1)
        is_dg = st.checkbox("Prowadzę działalność gospodarczą (wyższy limit IKZE)")

    ike_res = ike_ikze_optimizer(ike_funded, ikze_funded, pit_rate, cagr_ike, years_ret, is_dg)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IKE — do wpłaty", f"{ike_res['ike_remaining']:,.0f} PLN")
    c2.metric("IKZE — do wpłaty", f"{ike_res['ikze_remaining']:,.0f} PLN")
    c3.metric("Odliczenie IKZE (teraz)", f"{ike_res['ikze_deduction_current_year']:,.0f} PLN", delta="zwrot PIT")
    c4.metric(f"Przewaga IKE vs rachunek ({years_ret}L)", f"{ike_res['ike_vs_regular_advantage']:,.0f} PLN")

    for rec in ike_res.get("recommendations", []):
        if "💚" in rec or "💙" in rec:
            st.success(rec)
        else:
            st.info(rec)

    # Visualize IKE vs regular account growth
    years_range = list(range(0, years_ret + 1))
    ike_growth = [IKE_LIMIT_PLN * (1 + cagr_ike) ** y for y in years_range]
    regular_growth_pretax = [IKE_LIMIT_PLN * (1 + cagr_ike) ** y for y in years_range]
    regular_growth_aftertax = [
        IKE_LIMIT_PLN + (IKE_LIMIT_PLN * (1 + cagr_ike) ** y - IKE_LIMIT_PLN) * (1 - TAX_BELKA)
        for y in years_range
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_range, y=ike_growth, name="IKE (bez Belki)", line=dict(color="#00e676")))
    fig.add_trace(go.Scatter(x=years_range, y=regular_growth_aftertax, name="Rachunek maklerski (po Belce)", line=dict(color="#ff1744", dash="dash")))
    fig.update_layout(
        template="plotly_dark", height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Lata", yaxis_title="Wartość (PLN)",
        title=f"IKE vs Rachunek Maklerski — {years_ret} lat przy CAGR={cagr_ike:.0%}",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("### ✂️ Tax Loss Harvesting — Identyfikacja Strat do Realizacji")
    st.markdown("Wprowadź pozycje z portfela:")

    n_pos = st.number_input("Liczba pozycji", 1, 20, 5, 1)
    positions = []
    with st.expander("📝 Wprowadź pozycje portfela", expanded=True):
        for i in range(int(n_pos)):
            cc = st.columns([2, 1, 1, 1, 1])
            ticker = ticker_input(f"Ticker {i+1}", key=f"tlh_t_{i}", value=["SPY", "QQQ", "AAPL", "TSLA", "BNDX"][i % 5], parent=cc[0])
            qty = cc[1].number_input("Ilość", 1.0, key=f"tlh_q_{i}", value=10.0)
            avg_cost = cc[2].number_input("Śr. koszt (USD)", 1.0, key=f"tlh_c_{i}", value=[500.0, 400.0, 180.0, 250.0, 60.0][i % 5])
            curr_price = cc[3].number_input("Cena bieżąca", 1.0, key=f"tlh_p_{i}", value=[490.0, 410.0, 170.0, 260.0, 55.0][i % 5])
            fx = cc[4].number_input("FX (PLN/USD)", 1.0, key=f"tlh_fx_{i}", value=4.12)
            pos = create_position(ticker, qty, avg_cost, curr_price, fx_rate=fx)
            positions.append(pos)

    realised_gains = st.number_input("Zrealizowane zyski YTD (PLN)", 0.0, step=1000.0, value=10000.0)
    if positions:
        tlh = tax_loss_harvesting(positions, realised_gains)
        c1, c2, c3 = st.columns(3)
        c1.metric("Strata do TLH", f"{tlh.get('total_loss_available', 0):,.0f} PLN")
        c2.metric("Belka do odzysku", f"{tlh.get('tax_saved_gross', 0):,.0f} PLN")
        c3.metric("Netto po TC", f"{tlh.get('tax_saved_net', 0):,.0f} PLN", delta="zysk z TLH")

        for rec in tlh.get("recommendations", []):
            if "🔴" in rec:
                st.success(rec)
            elif "⚠️" in rec:
                st.warning(rec)
            else:
                st.info(rec)

        candidates = tlh.get("candidates", [])
        if candidates:
            df_cand = pd.DataFrame(candidates)
            st.dataframe(df_cand[["ticker", "loss_pln", "loss_pct", "tax_benefit_pln", "recommendation"]].rename(
                columns={"ticker": "Ticker", "loss_pln": "Strata (PLN)", "loss_pct": "Strata %",
                         "tax_benefit_pln": "Korzyść podatkowa (PLN)", "recommendation": "Akcja"}
            ), use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### 📊 Roczny Szacunek Podatku Belka (PIT-38)")
    col1, col2 = st.columns(2)
    with col1:
        from modules.ui.widgets import tickers_area
        planned_sells_val = tickers_area(
            "Planowane sprzedaże (ticker, ilość, cena sprzedaży — po przecinku, każda w linii)",
            "SPY, 5, 495\nAAPL, 20, 185",
            height=120,
        )
        dywidend = st.number_input("Dywidendy brutto (PLN)", 0.0, step=500.0, value=2000.0)
        withholding = st.number_input("Podatek u źródła zapłacony (PLN)", 0.0, step=100.0, value=300.0)
    with col2:
        st.info("""
        **Jak działa PIT-38:**
        - Zyski kapitałowe: 19% Belka
        - Straty: kompensata z zyskami w tym samym roku
        - Dywidendy: 19%, minus podatek u źródła
        - IKE/IKZE: zero Belki przy wypłacie po 60 r.ż. (IKE) / 10% ryczałt (IKZE)
        """)

    planned_sells = []
    if positions:
        for line in planned_sells_val.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    planned_sells.append({"ticker": parts[0], "quantity": float(parts[1]), "sell_price": float(parts[2])})
                except Exception:
                    pass

    belka = annual_belka_estimate(positions if positions else [], planned_sells, dywidend, withholding)

    c1, c2, c3 = st.columns(3)
    c1.metric("Zyski kapitałowe", f"{belka.get('capital_gains_pln', 0):,.0f} PLN")
    c2.metric("Straty", f"{belka.get('capital_losses_pln', 0):,.0f} PLN")
    c3.metric("Podstawa opodatkowania", f"{belka.get('net_taxable_pln', 0):,.0f} PLN")

    c1, c2, c3 = st.columns(3)
    c1.metric("Belka od zynsku kap.", f"{belka.get('belka_due_pln', 0):,.0f} PLN")
    c2.metric("Podatek od dywidend", f"{belka.get('dividends_tax_pln', 0):,.0f} PLN")
    c3.metric("💰 ŁĄCZNY PODATEK", f"{belka.get('total_tax_due_pln', 0):,.0f} PLN",
              delta=f"efektywna stawka: {belka.get('effective_rate', 0):.1%}")

    st.markdown(f"*Efektywna stawka podatkowa: {belka.get('effective_rate', 0):.1%}*")

with tab4:
    st.markdown("### 🏢 PPK (Pracownicze Plany Kapitałowe) vs Inwestowanie Prywatne")
    st.markdown("Sprawdź, ile zyskujesz dzięki dopłatom od pracodawcy i państwa.")
    
    col1, col2 = st.columns(2)
    with col1:
        salary = st.number_input("Miesięczne wynagrodzenie brutto (PLN)", 3000, 50000, 10000, 500)
    with col2:
        y_sim = st.slider("Horyzont symulacji (Lata)", 5, 40, 20)
        
    ppk_res = ppk_simulator(salary, years=y_sim)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Wpłaty własne", f"{ppk_res['total_employee_paid']:,.0f} PLN")
    c2.metric("Wpłaty pracodawcy/państwa", f"{ppk_res['total_employer_paid']+ppk_res['total_state_paid']:,.0f} PLN")
    c3.metric("Ostateczny kapitał", f"{ppk_res['final_balance']:,.0f} PLN")
    
    st.success(f"Darmowy kapitał przewagi nad inwestowaniem samodzielnym: **{ppk_res['ppk_advantage']:,.0f} PLN**")

with tab5:
    st.markdown("### 🗂️ Asset Location (Optymalizacja rozmieszczenia aktywów)")
    st.markdown("IKE i IKZE to limitowane darmowe pule podatkowe. Gdzie wpakować aktywa uciekając przed Belką?")
    
    col_a, col_space = st.columns([2, 1])
    with col_space:
        free_space = st.number_input("Dostępny limit IKE/IKZE", 0, 50000, 20000, 1000)
        
    assets = [
        {"name": "Vanguard S&P 500 ETF (VUAA)", "cagr": 0.08, "div_yield": 0.00, "value": 15000},
        {"name": "Global High Dividend ETF", "cagr": 0.07, "div_yield": 0.04, "value": 10000},
        {"name": "Obligacje Skarbowe EDO (10Y)", "cagr": 0.04, "div_yield": 0.00, "value": 20000}
    ]
    
    alloc = asset_location_optimizer(assets, free_space)
    
    df_alloc = pd.DataFrame(alloc['allocation'])
    df_alloc.rename(columns={"name": "Aktywo", "tax_free_account": "IKE/IKZE (PLN)", "taxable_account": "Zwykły Maklerski (PLN)", "reason": "Uzasadnienie"}, inplace=True)
    
    st.dataframe(df_alloc, use_container_width=True, hide_index=True)
    st.info(f"Pozostałe niewykorzystane miejsce na kontach emerytalnych: {alloc['unfilled_tax_free_space']:,.0f} PLN")

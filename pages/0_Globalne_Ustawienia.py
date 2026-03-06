"""
0_Globalne_Ustawienia.py — Strona Globalnych Ustawień Portfela
==============================================================
Centralny panel konfiguracji portfela Barbell Strategy.
Ustawienia zapisywane do global_settings.json i propagowane do wszystkich modułów.
"""
import copy
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from modules.styling import apply_styling
from modules.global_settings import (
    GlobalPortfolio,
    get_gs, set_gs,
    save_global_settings, load_global_settings,
    apply_gs_to_session, force_apply_gs_to_session,
    PRESET_PROFILES,
)
from config import GLOBAL_SETTINGS_PATH

st.set_page_config(
    page_title="Globalne Ustawienia | Barbell Strategy",
    page_icon="🌐",
    layout="wide",
)
st.markdown(apply_styling(), unsafe_allow_html=True)

# ── Inicjalizacja (ładuj z dysku przy pierwszym odwiedzeniu) ───────────────────
gs = get_gs()
apply_gs_to_session(gs)   # wstrzyknij jako domyślne (nie nadpisuje lokalnych zmian)

# ── Nagłówek ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(0,204,255,0.15) 0%, rgba(168,85,247,0.15) 100%);
    border: 1px solid rgba(0,204,255,0.3);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
">
    <h1 style="margin:0; font-size:2rem;">🌐 Globalne Ustawienia Portfela</h1>
    <p style="color:#94a3b8; margin:6px 0 0 0;">
    Skonfiguruj swój portfel raz — ustawienia automatycznie propagują się do
    Symulatora, Stress Testów, Wealth Optimizera i wszystkich pozostałych modułów.
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI STATUS BAR ─────────────────────────────────────────────────────────────
blended = gs.blended_rate
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🔒 Część Bezpieczna", f"{gs.alloc_safe_pct:.0%}")
k2.metric("⚡ Część Ryzykowna",  f"{gs.alloc_risky_pct:.0%}")
k3.metric("📈 Stopa Bezpieczna", f"{gs.safe_rate*100:.2f}%")
k4.metric("🎯 Blended Rate",     f"{blended*100:.2f}%",
          help="Efektywna oczekiwana stopa całego portfela: safe*rate + risky*8% (szacunek)")
k5.metric("💰 Kapitał Startowy", f"{gs.initial_capital:,.0f} PLN")

st.divider()

# ── ZAKŁADKI ──────────────────────────────────────────────────────────────────
tab_portfolio, tab_profiles, tab_status, tab_preview = st.tabs([
    "🏦 Portfel",
    "📋 Profile",
    "🔗 Status Propagacji",
    "📊 Podgląd",
])

# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKA 1: PORTFEL (główny edytor)
# ══════════════════════════════════════════════════════════════════════════════
with tab_portfolio:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ─── LEWA KOLUMNA ─────────────────────────────────────────────────────────
    with col_left:

        # ── Kapitał startowy ──────────────────────────────────────────────────
        st.markdown("### 💰 Kapitał Startowy")
        new_capital = st.number_input(
            "Kapitał Startowy (PLN)",
            min_value=1_000.0, max_value=100_000_000.0,
            value=float(gs.initial_capital),
            step=10_000.0,
            format="%.0f",
            key="gs_capital",
        )

        st.markdown("---")

        # ── Podział portfela ──────────────────────────────────────────────────
        st.markdown("### 📊 Podział Bezpieczna / Ryzykowna")

        alloc_safe_int = st.slider(
            "Część Bezpieczna (%)",
            min_value=0, max_value=100,
            value=int(round(gs.alloc_safe_pct * 100)),
            step=5,
            key="gs_alloc_safe",
            help="Przesuń suwak, aby określić podział między częścią bezpieczną (obligacje) a ryzykowną (akcje, krypto).",
        )
        alloc_safe_new = alloc_safe_int / 100.0
        alloc_risky_new = 1.0 - alloc_safe_new

        # Wizualny pasek podziału
        bar_html = f"""
        <div style="border-radius:8px; overflow:hidden; height:28px; display:flex; margin:4px 0 12px 0;">
            <div style="background:linear-gradient(90deg,#1e40af,#3b82f6); width:{alloc_safe_int}%;
                        display:flex; align-items:center; justify-content:center;
                        font-size:12px; font-weight:700; color:white;">
                🔒 {alloc_safe_int}%
            </div>
            <div style="background:linear-gradient(90deg,#dc2626,#f97316); width:{100-alloc_safe_int}%;
                        display:flex; align-items:center; justify-content:center;
                        font-size:12px; font-weight:700; color:white;">
                ⚡ {100-alloc_safe_int}%
            </div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)

        st.markdown("---")

        # ── Część Bezpieczna ──────────────────────────────────────────────────
        st.markdown("### 🔒 Część Bezpieczna")

        safe_type_opts = ["🏦 Obligacje Skarbowe RP 3-letnie (TOS — stała stopa)", "📁 Własny Koszyk Tickerów"]
        safe_type_idx = 0 if gs.safe_type == "fixed" else 1
        safe_type_sel = st.radio(
            "Rodzaj aktywa bezpiecznego",
            safe_type_opts,
            index=safe_type_idx,
            key="gs_safe_type",
            horizontal=False,
        )
        new_safe_type = "fixed" if safe_type_sel == safe_type_opts[0] else "tickers"

        new_safe_rate = gs.safe_rate
        new_safe_tickers = list(gs.safe_tickers)

        if new_safe_type == "fixed":
            st.markdown("##### Oprocentowanie obligacji TOS")
            new_safe_rate = st.number_input(
                "Oprocentowanie roczne (%)",
                min_value=0.01, max_value=20.0,
                value=float(gs.safe_rate * 100),
                step=0.01,
                format="%.2f",
                key="gs_safe_rate",
                help="Obligacje Skarbowe 3-letnie. Domyślnie TOS = 5.51%",
            ) / 100.0

            # EAR calc
            from config import TAX_BELKA
            rate_net = new_safe_rate * (1 - TAX_BELKA)
            ear = (1 + rate_net) - 1
            st.info(
                f"📈 **Efektywna stopa netto** po podatku Belki ({TAX_BELKA:.0%}): "
                f"**{ear*100:.3f}%** rocznie"
            )
        else:
            st.markdown("##### Koszyk bezpieczny (tickers Yahoo Finance)")
            safe_tickers_str_input = st.text_input(
                "Tickery bezpieczne (oddzielone przecinkami)",
                value=gs.safe_tickers_str or "TLT, IEF, GLD",
                key="gs_safe_tickers_str",
            )
            new_safe_tickers = [t.strip().upper() for t in safe_tickers_str_input.split(",") if t.strip()]
            st.caption("Przykłady: TLT (obligacje 20+), IEF (obligacje 7-10), GLD (złoto), TIPS (inflacja)")

    # ─── PRAWA KOLUMNA ────────────────────────────────────────────────────────
    with col_right:
        st.markdown("### ⚡ Część Ryzykowna")
        st.caption("Zdefiniuj papiery wartościowe i ich wagę w części ryzykownej portfela. Wagi powinny sumować się do 100%.")

        # Edytowalna tabela ryzykownych aktywów
        risky_df_default = pd.DataFrame(gs.risky_assets)
        if risky_df_default.empty or "ticker" not in risky_df_default.columns:
            risky_df_default = pd.DataFrame(
                [{"ticker": "SPY", "weight": 100.0, "asset_class": "ETF US"}]
            )

        # Upewnij się że kolumna asset_class istnieje
        if "asset_class" not in risky_df_default.columns:
            risky_df_default["asset_class"] = "N/A"

        risky_df_default = risky_df_default.rename(columns={
            "ticker": "Ticker",
            "weight": "Waga (%)",
            "asset_class": "Klasa Aktywu",
        })

        edited_risky = st.data_editor(
            risky_df_default,
            num_rows="dynamic",
            use_container_width=True,
            key="gs_risky_table",
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", help="Symbol Yahoo Finance, np. SPY, BTC-USD"),
                "Waga (%)": st.column_config.NumberColumn("Waga (%)", min_value=0.1, max_value=100.0, format="%.1f"),
                "Klasa Aktywu": st.column_config.SelectboxColumn(
                    "Klasa Aktywu",
                    options=["ETF US", "ETF Tech", "ETF Europe", "Akcja", "Krypto", "Obligacje", "Surowce", "Inne"],
                ),
            },
        )

        # Walidacja wag
        total_w = edited_risky["Waga (%)"].sum() if not edited_risky.empty else 0
        if abs(total_w - 100.0) > 0.5:
            st.error(f"⚠️ Suma wag = **{total_w:.1f}%** — musi wynosić 100%. Różnica: {total_w-100:.1f}%")
            wt_ok = False
        else:
            st.success(f"✅ Suma wag = {total_w:.1f}% — wagi prawidłowe.")
            wt_ok = True

        # Przygotuj nowe risky_assets z edytora
        new_risky_assets = []
        for _, row in edited_risky.iterrows():
            t = str(row.get("Ticker", "")).strip().upper()
            try:
                w = float(row.get("Waga (%)", 0))
            except (ValueError, TypeError):
                w = 0.0
            ac = str(row.get("Klasa Aktywu", "Inne"))
            if t:
                new_risky_assets.append({"ticker": t, "weight": w, "asset_class": ac})

        st.markdown("---")

        # ── Szybkie podpowiedzi do dodania aktywów ──────────────────────────
        st.markdown("##### 💡 Popularne aktywa do dodania")
        suggestion_cols = st.columns(4)
        suggestions = [
            ("SPY", "ETF US"), ("QQQ", "ETF Tech"), ("BTC-USD", "Krypto"), ("GLD", "Surowce"),
            ("NVDA", "Akcja"), ("AAPL", "Akcja"), ("VTI", "ETF US"), ("MSTR", "Akcja"),
        ]
        for i, (tick, cls) in enumerate(suggestions):
            suggestion_cols[i % 4].markdown(
                f"<span style='background:rgba(0,204,255,0.1);border:1px solid rgba(0,204,255,0.2);"
                f"border-radius:4px;padding:2px 6px;font-size:11px;color:#00ccff;'>"
                f"<b>{tick}</b> <span style='color:#6b7280'>{cls}</span></span>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── PRZYCISKI ZAPISU ───────────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3, _ = st.columns([2, 2, 2, 3])

    profile_name_input = st.text_input(
        "Nazwa profilu",
        value=gs.profile_name,
        key="gs_profile_name",
        help="Nadaj nazwę bieżącej konfiguracji dla łatwej identyfikacji",
    )

    save_ok = wt_ok  # allow save only when weights are valid

    with btn_col1:
        if st.button("💾 Zapisz jako Domyślne", type="primary", use_container_width=True, disabled=not save_ok):
            new_gs = GlobalPortfolio(
                safe_type=new_safe_type,
                safe_rate=new_safe_rate,
                safe_tickers=new_safe_tickers,
                risky_assets=new_risky_assets,
                alloc_safe_pct=alloc_safe_new,
                initial_capital=new_capital,
                profile_name=profile_name_input,
            )
            ok = save_global_settings(new_gs)
            set_gs(new_gs)
            force_apply_gs_to_session(new_gs)
            if ok:
                st.toast(f"✅ Ustawienia zapisane jako '{profile_name_input}' do {GLOBAL_SETTINGS_PATH}", icon="✅")
                st.rerun()
            else:
                st.error("❌ Błąd zapisu do pliku. Sprawdź uprawnienia do katalogu.")

    with btn_col2:
        if st.button("📤 Zastosuj (bez zapisu)", use_container_width=True, disabled=not save_ok):
            new_gs = GlobalPortfolio(
                safe_type=new_safe_type,
                safe_rate=new_safe_rate,
                safe_tickers=new_safe_tickers,
                risky_assets=new_risky_assets,
                alloc_safe_pct=alloc_safe_new,
                initial_capital=new_capital,
                profile_name=profile_name_input,
            )
            set_gs(new_gs)
            force_apply_gs_to_session(new_gs)
            st.toast("📤 Ustawienia zastosowane do tej sesji (nie zapisane na dysk).", icon="📤")
            st.rerun()

    with btn_col3:
        if st.button("🔄 Przywróć Domyślne Fabryczne", use_container_width=True):
            factory = GlobalPortfolio()
            set_gs(factory)
            force_apply_gs_to_session(factory)
            st.toast("🔄 Przywrócono wartości fabryczne.", icon="🔄")
            st.rerun()

    if gs.last_updated:
        st.caption(f"📅 Ostatni zapis: `{gs.last_updated}` | Profil: **{gs.profile_name}** | Plik: `{GLOBAL_SETTINGS_PATH}`")

# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKA 2: PROFILE
# ══════════════════════════════════════════════════════════════════════════════
with tab_profiles:
    st.markdown("### 📋 Gotowe Profile Inwestora")
    st.markdown(
        "Wczytaj jeden z predefiniowanych profili jednym kliknięciem. "
        "Po wczytaniu wróć do zakładki **🏦 Portfel** i kliknij **💾 Zapisz jako Domyślne**."
    )
    st.markdown("")

    for preset_name, preset_data in PRESET_PROFILES.items():
        risky_preview = ", ".join(
            f"{a['ticker']} {a['weight']:.0f}%" for a in preset_data["risky_assets"]
        )
        alloc_safe = preset_data["alloc_safe_pct"]

        with st.expander(f"{preset_name}  |  {alloc_safe:.0%} bezpieczna / {1-alloc_safe:.0%} ryzykowna", expanded=False):
            c1, c2, c3 = st.columns([2, 3, 2])
            with c1:
                st.metric("Część bezpieczna", f"{alloc_safe:.0%}")
                st.metric("Stopa TOS", f"{preset_data['safe_rate']*100:.2f}%")
            with c2:
                st.markdown(f"**Koszyk ryzykowny:**  \n{risky_preview}")
            with c3:
                if st.button(f"⬇️ Wczytaj", key=f"preset_{preset_name}", use_container_width=True):
                    preset_gs = GlobalPortfolio(
                        safe_type=preset_data["safe_type"],
                        safe_rate=preset_data["safe_rate"],
                        safe_tickers=preset_data.get("safe_tickers", []),
                        risky_assets=copy.deepcopy(preset_data["risky_assets"]),
                        alloc_safe_pct=preset_data["alloc_safe_pct"],
                        initial_capital=preset_data.get("initial_capital", gs.initial_capital),
                        profile_name=preset_data["profile_name"],
                    )
                    set_gs(preset_gs)
                    force_apply_gs_to_session(preset_gs)
                    st.toast(f"⬇️ Wczytano profil: {preset_name}", icon="⬇️")
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKA 3: STATUS PROPAGACJI
# ══════════════════════════════════════════════════════════════════════════════
with tab_status:
    st.markdown("### 🔗 Status Propagacji Globalnych Ustawień")
    st.markdown(
        "Poniżej widać, jakie wartości zostały wstrzyknięte do każdego modułu "
        "automatycznie na podstawie Twoich globalnych ustawień."
    )

    propagation_map = [
        {
            "Moduł": "1_Symulator.py (Monte Carlo)",
            "Parametr": "Alokacja bezpieczna",
            "Wartość": f"{gs.alloc_safe_pct:.0%}",
            "Klucz": "_s.mc_alloc_safe",
            "Status": "✅ Aktywny",
        },
        {
            "Moduł": "1_Symulator.py (AI Backtest)",
            "Parametr": "Stopa obligacji / koszyk Safe",
            "Wartość": f"{gs.safe_rate*100:.2f}%" if gs.safe_type == "fixed" else gs.safe_tickers_str,
            "Klucz": "_s.ai_safe_rate / ai_safe_type",
            "Status": "✅ Aktywny",
        },
        {
            "Moduł": "1_Symulator.py (AI Backtest)",
            "Parametr": "Koszyk ryzykowny",
            "Wartość": gs.risky_tickers_str[:40] + ("…" if len(gs.risky_tickers_str) > 40 else ""),
            "Klucz": "_s.ai_risky_tickers",
            "Status": "✅ Aktywny",
        },
        {
            "Moduł": "1_Symulator.py",
            "Parametr": "Kapitał startowy",
            "Wartość": f"{gs.initial_capital:,.0f} PLN",
            "Klucz": "_s.mc_cap / ai_cap",
            "Status": "✅ Aktywny",
        },
        {
            "Moduł": "3_Stress_Test.py",
            "Parametr": "Waga bezpieczna",
            "Wartość": f"{gs.alloc_safe_pct:.0%}",
            "Klucz": "_s.st_sw",
            "Status": "✅ Aktywny",
        },
        {
            "Moduł": "3_Stress_Test.py",
            "Parametr": "Koszyki Safe & Risky",
            "Wartość": f"Safe: {gs.safe_tickers_str or 'TLT, GLD'} | Risky: {gs.risky_tickers_str[:25]}",
            "Klucz": "_s.st_safe / st_risky",
            "Status": "✅ Aktywny",
        },
        {
            "Moduł": "4_Emerytura.py",
            "Parametr": "Kapitał startowy",
            "Wartość": f"{gs.initial_capital:,.0f} PLN",
            "Klucz": "rem_initial_capital",
            "Status": "✅ Aktywny",
        },
        {
            "Moduł": "19_Wealth_Optimizer.py",
            "Parametr": "Łączny majątek",
            "Wartość": f"{gs.initial_capital:,.0f} PLN",
            "Klucz": "_gs_wealth_total",
            "Status": "✅ Aktywny",
        },
    ]

    df_prop = pd.DataFrame(propagation_map)
    st.dataframe(df_prop, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### ℹ️ Jak działa propagacja?")
    st.markdown("""
    1. **Pierwsze odwiedzenie modułu** w sesji: wartości z Globalnych Ustawień są automatycznie wstrzykiwane jako domyślne.
    2. **Ręczna zmiana w module** (np. przesunięcie slidera w Symulatorze): zmiana jest **lokalna** i nie nadpisuje ustawień globalnych.
    3. **Przycisk ↩ Przywróć z Globalnych** (dostępny w każdym module): resetuje lokalne zmiany do wartości globalnych.
    4. **Po zapisaniu nowych ustawień globalnych**: odśwież moduł lub kliknij przycisk ↩, aby moduł załadował nowe wartości.
    """)

    if st.button("🔄 Force Sync — Synchronizuj wszystkie moduły teraz", type="secondary"):
        force_apply_gs_to_session(gs)
        st.success("✅ Session state zsynchronizowany z globalnymi ustawieniami.")


# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKA 4: PODGLĄD PORTFELA
# ══════════════════════════════════════════════════════════════════════════════
with tab_preview:
    st.markdown("### 📊 Wizualizacja Portfela")

    # ── Pie Chart główny ──────────────────────────────────────────────────────
    col_pie1, col_pie2 = st.columns(2)

    with col_pie1:
        st.markdown("#### Podział Bezpieczna / Ryzykowna")
        fig_split = go.Figure(go.Pie(
            labels=["🔒 Bezpieczna", "⚡ Ryzykowna"],
            values=[gs.alloc_safe_pct, gs.alloc_risky_pct],
            hole=0.55,
            marker_colors=["#3b82f6", "#f97316"],
            texttemplate="%{percent:.0%}",
            textfont_size=16,
        ))
        fig_split.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(font=dict(size=13)),
            annotations=[dict(
                text=f"{gs.blended_rate*100:.1f}%",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=22, color="white", family="Inter"),
            )],
        )
        st.plotly_chart(fig_split, use_container_width=True)
        st.caption("Liczba w środku = efektywna blended rate portfela")

    with col_pie2:
        st.markdown("#### Skład Części Ryzykownej")
        if gs.risky_assets:
            labels_r = [a["ticker"] for a in gs.risky_assets]
            values_r = [a["weight"] for a in gs.risky_assets]
            total_r = sum(values_r) or 1
            colors_r = [
                "#ff6384", "#ff9f40", "#ffcd56", "#4bc0c0",
                "#36a2eb", "#9966ff", "#ff6384", "#c9cbcf",
            ]
            fig_risky = go.Figure(go.Pie(
                labels=labels_r,
                values=values_r,
                hole=0.45,
                marker_colors=colors_r[:len(labels_r)],
                texttemplate="%{label}<br>%{percent:.0%}",
                textfont_size=12,
            ))
            fig_risky.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False,
            )
            st.plotly_chart(fig_risky, use_container_width=True)
        else:
            st.info("Brak aktywów ryzykownych do wyświetlenia.")

    st.markdown("---")

    # ── Tabela szczegółów ─────────────────────────────────────────────────────
    st.markdown("#### 📋 Szczegółowy Skład Portfela")

    cap = gs.initial_capital
    rows = []

    # Bezpieczna część
    safe_label = f"TOS {gs.safe_rate*100:.2f}%" if gs.safe_type == "fixed" else gs.safe_tickers_str
    rows.append({
        "Segment": "🔒 Bezpieczna",
        "Składnik": safe_label,
        "Waga w Portfelu": f"{gs.alloc_safe_pct:.1%}",
        "Kwota (PLN)": f"{gs.alloc_safe_pct * cap:,.0f}",
        "Klasa": "Obligacje / Skarbowe",
    })

    # Ryzykowna część (rozpisana per tytuł)
    for a in gs.risky_assets:
        weight_in_total = gs.alloc_risky_pct * (a["weight"] / max(sum(x["weight"] for x in gs.risky_assets), 1e-6))
        rows.append({
            "Segment": "⚡ Ryzykowna",
            "Składnik": a["ticker"],
            "Waga w Portfelu": f"{weight_in_total:.2%}",
            "Kwota (PLN)": f"{weight_in_total * cap:,.0f}",
            "Klasa": a.get("asset_class", "N/A"),
        })

    df_detail = pd.DataFrame(rows)
    st.dataframe(df_detail, use_container_width=True, hide_index=True)

    # ── Wykres słupkowy kwot ──────────────────────────────────────────────────
    labels_bar = [r["Składnik"] for r in rows]
    values_bar = [gs.alloc_safe_pct * cap] + [
        gs.alloc_risky_pct * (a["weight"] / max(sum(x["weight"] for x in gs.risky_assets), 1e-6)) * cap
        for a in gs.risky_assets
    ]
    colors_bar = ["#3b82f6"] + ["#f97316"] * len(gs.risky_assets)

    fig_bar = go.Figure(go.Bar(
        x=labels_bar,
        y=values_bar,
        marker_color=colors_bar,
        text=[f"{v:,.0f} PLN" for v in values_bar],
        textposition="outside",
    ))
    fig_bar.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,11,20,0.6)",
        height=350,
        yaxis_title="Kwota (PLN)",
        showlegend=False,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

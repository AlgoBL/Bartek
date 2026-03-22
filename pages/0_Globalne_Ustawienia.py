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
from modules.i18n import t
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
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(0,204,255,0.15) 0%, rgba(168,85,247,0.15) 100%);
    border: 1px solid rgba(0,204,255,0.3);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
">
    <h1 style="margin:0; font-size:2rem;">{t('gs_header')}</h1>
    <p style="color:#94a3b8; margin:6px 0 0 0;">
    {t('gs_subtitle')}
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI STATUS BAR ─────────────────────────────────────────────────────────────
blended = gs.blended_rate
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(t("gs_safe_pct"),    f"{gs.alloc_safe_pct:.0%}")
k2.metric(t("gs_risky_pct"),   f"{gs.alloc_risky_pct:.0%}")
k3.metric(t("gs_safe_rate"),   f"{gs.safe_rate*100:.2f}%")
k4.metric(t("gs_blended"),     f"{blended*100:.2f}%", help=t("gs_blended_help"))
k5.metric(t("capital_start_label"),  f"{gs.initial_capital:,.0f} PLN")

st.divider()

# ── ZAKŁADKI ──────────────────────────────────────────────────────────────────
tab_portfolio, tab_profiles, tab_status, tab_preview = st.tabs([
    t("gs_tab_portfolio"),
    t("gs_tab_profiles"),
    t("gs_tab_status"),
    t("gs_tab_preview"),
])

# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKA 1: PORTFEL (główny edytor)
# ══════════════════════════════════════════════════════════════════════════════
with tab_portfolio:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ─── LEWA KOLUMNA ─────────────────────────────────────────────────────────
    with col_left:

        st.markdown(f"### {t('capital_start_label')}")
        new_capital = st.number_input(
            t("capital_start"),
            min_value=1_000.0, max_value=100_000_000.0,
            value=float(gs.initial_capital),
            step=10_000.0,
            format="%.0f",
            key="gs_capital",
        )

        st.markdown("---")

        # ── Heartbeat Engine ──────────────────────────────────────────────────
        st.markdown(t("gs_heartbeat"))
        st.caption(t("gs_heartbeat_cap"))

        col_hb1, col_hb2 = st.columns([1, 2])
        with col_hb1:
            bg_enabled = st.toggle(t("gs_bg_toggle"), value=gs.bg_refresh_enabled, key="gs_bg_enabled")

        with col_hb2:
            bg_interval = st.slider(
                t("gs_bg_interval"),
                min_value=1, max_value=120,
                value=gs.bg_refresh_interval_minutes,
                step=1,
                disabled=not bg_enabled,
                key="gs_bg_interval"
            )

        st.markdown("---")

        # ── Język interfejsu ──────────────────────────────────────────────────
        st.markdown(f"### {t('lang_label')}")
        lang_opts = [t("lang_pl"), t("lang_en")]
        lang_idx = 0 if gs.language == "pl" else 1
        lang_sel = st.radio(
            t("lang_label"),
            lang_opts,
            index=lang_idx,
            key="gs_language",
            horizontal=True,
            label_visibility="collapsed",
        )
        new_language = "pl" if lang_sel == lang_opts[0] else "en"

        st.markdown("---")

        # ── Personalizacja Modułów (Menu) ─────────────────────────────────────
        st.markdown("### 🎛️ Objętość Menu")
        st.caption("Wybierz moduły naukowe do wyświetlenia na pasku bacznym.")
        
        all_possible_modules = [
            # Analiza Ryzyka
            "Factor Zoo & PCA", "EVT — Tail Risk", "Black-Litterman AI", 
            "DCC — Korelacje", "Stress Test",
            # Narzedzia
            "Symulator Barbell", "Skaner Rynku",
            # Planowanie
            "Emerytura / FIRE",
            # Ochrona Kapitału
            "Portfolio Health Monitor", "Concentration Risk", 
            "Drawdown Recovery", "Investment Clock",
            # Zarzadzanie
            "Regime Allocation", "Liquidity Risk", "Tail Risk Hedging", "Tax Optimizer PL",
            # Wzrost
            "Smart Rebalancing", "Sentiment & Flow", "Alt. Risk Premia", "Wealth Optimizer",
            # Life OS
            "Life OS — Łowca", "Day Trading", "Walk-Forward CPCV",
        ]
        
        # Jeśli lista w gs jest pusta, załóżmy że domyślnie wszystko włączone
        default_visible = gs.visible_modules if gs.visible_modules else all_possible_modules
        # Oczyść ze starych kluczy niewykorzystanych
        default_visible = [m for m in default_visible if m in all_possible_modules]
        
        new_visible_modules = st.multiselect(
            "Widoczne moduły",
            all_possible_modules,
            default=default_visible,
            key="gs_visible_modules",
            label_visibility="collapsed"
        )

        st.markdown("---")

        # ── Podział portfela ──────────────────────────────────────────────────
        st.markdown(t("gs_alloc_header"))

        alloc_safe_int = st.slider(
            t("gs_alloc_slider"),
            min_value=0, max_value=100,
            value=int(round(gs.alloc_safe_pct * 100)),
            step=5,
            key="gs_alloc_safe",
            help=t("gs_alloc_help"),
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
        st.markdown(t("gs_safe_header"))

        safe_type_opts = [t("gs_safe_fixed"), t("gs_safe_tickers")]
        safe_type_idx = 0 if gs.safe_type == "fixed" else 1
        safe_type_sel = st.radio(
            t("gs_safe_type_label"),
            safe_type_opts,
            index=safe_type_idx,
            key="gs_safe_type",
            horizontal=False,
        )
        new_safe_type = "fixed" if safe_type_sel == safe_type_opts[0] else "tickers"

        new_safe_rate = gs.safe_rate
        new_safe_tickers = list(gs.safe_tickers)

        if new_safe_type == "fixed":
            st.markdown(t("gs_bond_rate"))
            new_safe_rate = st.number_input(
                t("gs_bond_rate_input"),
                min_value=0.01, max_value=20.0,
                value=float(gs.safe_rate * 100),
                step=0.01,
                format="%.2f",
                key="gs_safe_rate",
                help=t("gs_bond_rate_help"),
            ) / 100.0

            from config import TAX_BELKA
            rate_net = new_safe_rate * (1 - TAX_BELKA)
            ear = (1 + rate_net) - 1
            st.info(
                f"{t('gs_ear_info')} ({TAX_BELKA:.0%}): "
                f"**{ear*100:.3f}%** {'rocznie' if gs.language == 'pl' else 'annually'}"
            )
        else:
            st.markdown(t("gs_ticker_basket"))
            safe_tickers_str_input = st.text_input(
                t("gs_ticker_input"),
                value=gs.safe_tickers_str or "TLT, IEF, GLD",
                key="gs_safe_tickers_str",
            )
            new_safe_tickers = [t_str.strip().upper() for t_str in safe_tickers_str_input.split(",") if t_str.strip()]
            st.caption(t("gs_ticker_examples"))

    # ─── PRAWA KOLUMNA ────────────────────────────────────────────────────────
    with col_right:
        st.markdown(t("gs_risky_header"))
        st.caption(t("gs_risky_caption"))

        # Edytowalna tabela ryzykownych aktywów
        risky_df_default = pd.DataFrame(gs.risky_assets)
        if risky_df_default.empty or "ticker" not in risky_df_default.columns:
            risky_df_default = pd.DataFrame(
                [{"ticker": "SPY", "weight": 100.0, "asset_class": "ETF US"}]
            )

        if "asset_class" not in risky_df_default.columns:
            risky_df_default["asset_class"] = "N/A"

        col_ticker  = t("ticker")
        col_weight  = t("weight_pct")
        col_class   = t("asset_class")

        risky_df_default = risky_df_default.rename(columns={
            "ticker": col_ticker,
            "weight": col_weight,
            "asset_class": col_class,
        })

        edited_risky = st.data_editor(
            risky_df_default,
            num_rows="dynamic",
            use_container_width=True,
            key="gs_risky_table",
            column_config={
                col_ticker: st.column_config.TextColumn(col_ticker, help="Yahoo Finance symbol, e.g. SPY, BTC-USD"),
                col_weight: st.column_config.NumberColumn(col_weight, min_value=0.1, max_value=100.0, format="%.1f"),
                col_class: st.column_config.SelectboxColumn(
                    col_class,
                    options=["ETF US", "ETF Tech", "ETF Europe", "Akcja", "Krypto", "Obligacje", "Surowce", "Inne"],
                ),
            },
        )

        # Walidacja wag
        total_w = edited_risky[col_weight].sum() if not edited_risky.empty else 0
        if abs(total_w - 100.0) > 0.5:
            st.error(t("gs_weight_err", w=total_w, d=total_w - 100))
            wt_ok = False
        else:
            st.success(t("gs_weight_ok", w=total_w))
            wt_ok = True

        # Przygotuj nowe risky_assets z edytora
        new_risky_assets = []
        for _, row in edited_risky.iterrows():
            tick = str(row.get(col_ticker, "")).strip().upper()
            try:
                w = float(row.get(col_weight, 0))
            except (ValueError, TypeError):
                w = 0.0
            ac = str(row.get(col_class, "Inne"))
            if tick:
                new_risky_assets.append({"ticker": tick, "weight": w, "asset_class": ac})

        st.markdown("---")

        # ── Szybkie podpowiedzi do dodania aktywów ──────────────────────────
        st.markdown(t("gs_popular"))
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
        t("gs_profile_name"),
        value=gs.profile_name,
        key="gs_profile_name",
        help=t("gs_profile_help"),
    )

    save_ok = wt_ok

    with btn_col1:
        if st.button(t("save_default"), type="primary", use_container_width=True, disabled=not save_ok):
            new_gs = GlobalPortfolio(
                safe_type=new_safe_type,
                safe_rate=new_safe_rate,
                safe_tickers=new_safe_tickers,
                risky_assets=new_risky_assets,
                alloc_safe_pct=alloc_safe_new,
                initial_capital=new_capital,
                bg_refresh_enabled=bg_enabled,
                bg_refresh_interval_minutes=bg_interval,
                language=new_language,
                visible_modules=new_visible_modules,
                profile_name=profile_name_input,
            )
            ok = save_global_settings(new_gs)
            set_gs(new_gs)
            force_apply_gs_to_session(new_gs)

            from modules.background_updater import bg_engine
            bg_engine.set_config(bg_enabled, bg_interval)

            if ok:
                st.toast(t("gs_toast_saved", name=profile_name_input, path=GLOBAL_SETTINGS_PATH), icon="✅")
                st.rerun()
            else:
                st.error(t("gs_write_err"))

    with btn_col2:
        if st.button(t("apply_no_save"), use_container_width=True, disabled=not save_ok):
            new_gs = GlobalPortfolio(
                safe_type=new_safe_type,
                safe_rate=new_safe_rate,
                safe_tickers=new_safe_tickers,
                risky_assets=new_risky_assets,
                alloc_safe_pct=alloc_safe_new,
                initial_capital=new_capital,
                bg_refresh_enabled=bg_enabled,
                bg_refresh_interval_minutes=bg_interval,
                language=new_language,
                profile_name=profile_name_input,
            )
            set_gs(new_gs)
            force_apply_gs_to_session(new_gs)

            from modules.background_updater import bg_engine
            bg_engine.set_config(bg_enabled, bg_interval)

            st.toast(t("gs_toast_applied"), icon="📤")
            st.rerun()

    with btn_col3:
        if st.button(t("restore_factory"), use_container_width=True):
            factory = GlobalPortfolio()
            set_gs(factory)
            force_apply_gs_to_session(factory)
            st.toast(t("gs_toast_factory"), icon="🔄")
            st.rerun()

    if gs.last_updated:
        st.caption(
            f"📅 {t('last_save')}: `{gs.last_updated}` | "
            f"{t('profile')}: **{gs.profile_name}** | "
            f"{t('file')}: `{GLOBAL_SETTINGS_PATH}`"
        )

# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKA 2: PROFILE
# ══════════════════════════════════════════════════════════════════════════════
with tab_profiles:
    st.markdown(t("gs_profiles_header"))
    st.markdown(t("gs_profiles_intro"))
    st.markdown("")

    for preset_name, preset_data in PRESET_PROFILES.items():
        risky_preview = ", ".join(
            f"{a['ticker']} {a['weight']:.0f}%" for a in preset_data["risky_assets"]
        )
        alloc_safe = preset_data["alloc_safe_pct"]

        with st.expander(f"{preset_name}  |  {alloc_safe:.0%} / {1-alloc_safe:.0%}", expanded=False):
            c1, c2, c3 = st.columns([2, 3, 2])
            with c1:
                st.metric(t("gs_safe_pct"), f"{alloc_safe:.0%}")
                st.metric("Stopa TOS", f"{preset_data['safe_rate']*100:.2f}%")
            with c2:
                st.markdown(f"**{t('gs_risky_header').replace('### ', '')}:**  \n{risky_preview}")
            with c3:
                if st.button(t("gs_load_profile"), key=f"preset_{preset_name}", use_container_width=True):
                    preset_gs = GlobalPortfolio(
                        safe_type=preset_data["safe_type"],
                        safe_rate=preset_data["safe_rate"],
                        safe_tickers=preset_data.get("safe_tickers", []),
                        risky_assets=copy.deepcopy(preset_data["risky_assets"]),
                        alloc_safe_pct=preset_data["alloc_safe_pct"],
                        initial_capital=preset_data.get("initial_capital", gs.initial_capital),
                        language=gs.language,
                        visible_modules=gs.visible_modules,
                        profile_name=preset_data["profile_name"],
                    )
                    set_gs(preset_gs)
                    force_apply_gs_to_session(preset_gs)
                    st.toast(t("gs_toast_loaded", name=preset_name), icon="⬇️")
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKA 3: STATUS PROPAGACJI
# ══════════════════════════════════════════════════════════════════════════════
with tab_status:
    st.markdown(t("gs_prop_header"))
    st.markdown(t("gs_prop_intro"))

    propagation_map = [
        {
            t("gs_prop_module"): "1_Symulator.py (Monte Carlo)",
            t("gs_prop_param"): t("gs_alloc_slider"),
            t("gs_prop_value"): f"{gs.alloc_safe_pct:.0%}",
            t("gs_prop_key"): "_s.mc_alloc_safe",
            t("gs_prop_status"): t("gs_prop_active"),
        },
        {
            t("gs_prop_module"): "1_Symulator.py (AI Backtest)",
            t("gs_prop_param"): t("gs_safe_rate"),
            t("gs_prop_value"): f"{gs.safe_rate*100:.2f}%" if gs.safe_type == "fixed" else gs.safe_tickers_str,
            t("gs_prop_key"): "_s.ai_safe_rate / ai_safe_type",
            t("gs_prop_status"): t("gs_prop_active"),
        },
        {
            t("gs_prop_module"): "1_Symulator.py (AI Backtest)",
            t("gs_prop_param"): t("gs_risky_header").replace("### ",""),
            t("gs_prop_value"): gs.risky_tickers_str[:40] + ("…" if len(gs.risky_tickers_str) > 40 else ""),
            t("gs_prop_key"): "_s.ai_risky_tickers",
            t("gs_prop_status"): t("gs_prop_active"),
        },
        {
            t("gs_prop_module"): "1_Symulator.py",
            t("gs_prop_param"): t("capital_start_label"),
            t("gs_prop_value"): f"{gs.initial_capital:,.0f} PLN",
            t("gs_prop_key"): "_s.mc_cap / ai_cap",
            t("gs_prop_status"): t("gs_prop_active"),
        },
        {
            t("gs_prop_module"): "3_Stress_Test.py",
            t("gs_prop_param"): t("st_safe_weight"),
            t("gs_prop_value"): f"{gs.alloc_safe_pct:.0%}",
            t("gs_prop_key"): "_s.st_sw",
            t("gs_prop_status"): t("gs_prop_active"),
        },
        {
            t("gs_prop_module"): "3_Stress_Test.py",
            t("gs_prop_param"): f"{t('st_safe_basket')} & {t('st_risky_basket')}",
            t("gs_prop_value"): f"Safe: {gs.safe_tickers_str or 'TLT, GLD'} | Risky: {gs.risky_tickers_str[:25]}",
            t("gs_prop_key"): "_s.st_safe / st_risky",
            t("gs_prop_status"): t("gs_prop_active"),
        },
        {
            t("gs_prop_module"): "4_Emerytura.py",
            t("gs_prop_param"): t("capital_start_label"),
            t("gs_prop_value"): f"{gs.initial_capital:,.0f} PLN",
            t("gs_prop_key"): "rem_initial_capital",
            t("gs_prop_status"): t("gs_prop_active"),
        },
        {
            t("gs_prop_module"): "19_Wealth_Optimizer.py",
            t("gs_prop_param"): t("wo_wealth"),
            t("gs_prop_value"): f"{gs.initial_capital:,.0f} PLN",
            t("gs_prop_key"): "_gs_wealth_total",
            t("gs_prop_status"): t("gs_prop_active"),
        },
    ]

    df_prop = pd.DataFrame(propagation_map)
    st.dataframe(df_prop, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(t("gs_how_header"))
    st.markdown(t("gs_how_text"))

    if st.button(t("gs_force_sync"), type="secondary"):
        force_apply_gs_to_session(gs)
        st.success(t("gs_sync_ok"))


# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKA 4: PODGLĄD PORTFELA
# ══════════════════════════════════════════════════════════════════════════════
with tab_preview:
    st.markdown(t("gs_preview_header"))

    # ── Pie Chart główny ──────────────────────────────────────────────────────
    col_pie1, col_pie2 = st.columns(2)

    with col_pie1:
        st.markdown(t("gs_pie_safe"))
        fig_split = go.Figure(go.Pie(
            labels=[t("gs_pie_safe_lbl"), t("gs_pie_risky_lbl")],
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
        st.caption(t("gs_pie_center_cap"))

    with col_pie2:
        st.markdown(t("gs_pie_risky"))
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
            st.info(t("gs_no_risky"))

    st.markdown("---")

    # ── Tabela szczegółów ─────────────────────────────────────────────────────
    st.markdown(t("gs_detail_header"))

    cap = gs.initial_capital
    rows = []

    # Bezpieczna część
    safe_label = f"TOS {gs.safe_rate*100:.2f}%" if gs.safe_type == "fixed" else gs.safe_tickers_str
    rows.append({
        t("gs_col_segment"):   t("gs_seg_safe"),
        t("gs_col_component"): safe_label,
        t("gs_col_weight"):    f"{gs.alloc_safe_pct:.1%}",
        t("gs_col_amount"):    f"{gs.alloc_safe_pct * cap:,.0f}",
        t("gs_col_class"):     t("gs_bonds_label"),
    })

    # Ryzykowna część (rozpisana per tytuł)
    for a in gs.risky_assets:
        weight_in_total = gs.alloc_risky_pct * (a["weight"] / max(sum(x["weight"] for x in gs.risky_assets), 1e-6))
        rows.append({
            t("gs_col_segment"):   t("gs_seg_risky"),
            t("gs_col_component"): a["ticker"],
            t("gs_col_weight"):    f"{weight_in_total:.2%}",
            t("gs_col_amount"):    f"{weight_in_total * cap:,.0f}",
            t("gs_col_class"):     a.get("asset_class", "N/A"),
        })

    df_detail = pd.DataFrame(rows)
    st.dataframe(df_detail, use_container_width=True, hide_index=True)

    # ── Wykres słupkowy kwot ──────────────────────────────────────────────────
    labels_bar = [r[t("gs_col_component")] for r in rows]
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
        yaxis_title=t("gs_bar_yaxis"),
        showlegend=False,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

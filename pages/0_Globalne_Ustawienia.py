"""
0_Globalne_Ustawienia.py — Centralny Panel Konfiguracji Portfela
================================================================
Nowy układ v2.0 — 5 zakładek:
  1. 💼 Mój Portfel   — ujednolicona tabela wszystkich aktywów + kapitał
  2. ⚙️ Ustawienia Techniczne — Engine, język, FX, tryb UI
  3. 📊 Podgląd & Analiza — wykresy i propagacja
  4. 🎯 Profile Presetów  — gotowe profile
  5. 🔍 Odkrywca ISIN     — wyszukiwarka ISIN
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
from config import GLOBAL_SETTINGS_PATH, TAX_BELKA

st.markdown(apply_styling(), unsafe_allow_html=True)

# ── Inicjalizacja ──────────────────────────────────────────────────────────────
gs = get_gs()
apply_gs_to_session(gs)

# ── Stałe stylu ───────────────────────────────────────────────────────────────
CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px 20px;margin-bottom:12px"
_SEC = "font-size:12px;color:#6b7280;letter-spacing:1.5px;text-transform:uppercase;font-weight:700;margin-bottom:6px"

# ── Nagłówek ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(0,204,255,0.12) 0%, rgba(168,85,247,0.12) 100%);
    border: 1px solid rgba(0,204,255,0.25);
    border-radius: 16px;
    padding: 22px 30px;
    margin-bottom: 20px;
">
    <h1 style="margin:0;font-size:1.9rem;">⚙️ Globalne Ustawienia Portfela</h1>
    <p style="color:#94a3b8;margin:6px 0 0 0;">
        Centralny panel konfiguracji Twojego portfela Barbell Strategy.
        Ustawienia zapisane tutaj propagują automatycznie do <b>wszystkich modułów</b> aplikacji.
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Bar ────────────────────────────────────────────────────────────────────
_assets = gs.all_assets_unified
_safe_w  = sum(a["weight"] for a in _assets if a.get("segment") == "Bezpieczny")
_risky_w = sum(a["weight"] for a in _assets if a.get("segment") == "Ryzykowny")
_total_w = _safe_w + _risky_w or 100
blended  = gs.blended_rate

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("🔒 Bezpieczny",   f"{_safe_w / _total_w:.0%}")
k2.metric("⚡ Ryzykowny",    f"{_risky_w / _total_w:.0%}")
k3.metric("📈 Stopa TOS",    f"{gs.safe_rate*100:.2f}%")
k4.metric("🔀 Blended",      f"{blended*100:.2f}%", help="Ważona stopa portfela (safe×TOS + risky×8%)")
k5.metric("💰 Kapitał",      f"{gs.initial_capital:,.0f} PLN")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKI
# ══════════════════════════════════════════════════════════════════════════════
tab_portfolio, tab_tech, tab_preview, tab_ret, tab_profiles, tab_isin = st.tabs([
    "💼 Mój Portfel",
    "⚙️ Ustawienia Techniczne",
    "📊 Podgląd & Analiza",
    "🏖️ Emerytura & FIRE",
    "🎯 Profile Presetów",
    "🔍 Odkrywca ISIN",
])

# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKA 1 — MÓJ PORTFEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_portfolio:

    st.markdown("""
    <div style="background:rgba(0,204,255,0.06);border:1px solid rgba(0,204,255,0.15);
                border-radius:10px;padding:12px 16px;margin-bottom:16px">
    📌 <b>Jedna tabela — pełny portfel.</b> Wpisz wszystkie swoje aktywa:
    obligacje skarbowe, ETF, akcje. Suma wag musi wynosić <b>100%</b>.
    Kolumna <i>Segment</i> decyduje, czy aktywo należy do części bezpiecznej czy ryzykownej.
    </div>
    """, unsafe_allow_html=True)

    # ── Kapitał ───────────────────────────────────────────────────────────────
    col_cap1, col_cap2 = st.columns([2, 3])
    with col_cap1:
        new_capital = st.number_input(
            "💰 Kapitał Całkowity (PLN)",
            min_value=1_000.0, max_value=100_000_000.0,
            value=float(gs.initial_capital),
            step=10_000.0, format="%.0f",
            key="gs_v2_capital",
            help="Łączna wartość Twojego portfela inwestycyjnego."
        )

    with col_cap2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"💡 Kwoty w tabeli poniżej wyliczane są automatycznie z kapitału {new_capital:,.0f} PLN")

    st.markdown("---")
    st.markdown("### 📋 Tabela Aktywów Portfela")
    st.caption("Dodaj wiersze przyciskiem ＋ w tabeli. Każde aktywo to jeden wiersz.")

    # Kolumny tabeli
    COL_NAME   = "Nazwa / Ticker"
    COL_TYPE   = "Typ Aktywa"
    COL_YIELD  = "Oprocentowanie %"
    COL_WEIGHT = "Waga %"
    COL_AMOUNT = "Kwota (PLN)"
    COL_SEG    = "Segment"

    # ── Inicjalizacja session_state tabeli ─────────────────────────────────
    if "gs_portfolio_df" not in st.session_state:
        # Załaduj z portfolio_assets lub zbuduj z legacy pól
        assets = gs.all_assets_unified
        if assets:
            rows = []
            for a in assets:
                amt = round(a.get("weight", 0) / 100.0 * new_capital, 0)
                rows.append({
                    COL_NAME:   a.get("name") or a.get("ticker", ""),
                    COL_TYPE:   a.get("type", "Inne"),
                    COL_YIELD:  float(a.get("yield_pct", 0.0)),
                    COL_WEIGHT: float(a.get("weight", 0.0)),
                    COL_AMOUNT: amt,
                    COL_SEG:    a.get("segment", "Ryzykowny"),
                })
        else:
            # Domyślne aktywa: Obligacje Skarbowe 3Y + SPY
            rows = [
                {COL_NAME: "Obligacje Skarbowe 3Y", COL_TYPE: "Obligacje Skarbowe PL",
                 COL_YIELD: 5.51,  COL_WEIGHT: 85.0,
                 COL_AMOUNT: round(0.85 * new_capital, 0), COL_SEG: "Bezpieczny"},
                {COL_NAME: "SPY", COL_TYPE: "ETF US",
                 COL_YIELD: 0.0,  COL_WEIGHT: 15.0,
                 COL_AMOUNT: round(0.15 * new_capital, 0), COL_SEG: "Ryzykowny"},
            ]
        st.session_state["gs_portfolio_df"] = pd.DataFrame(rows)

    # ── Edytor tabeli ─────────────────────────────────────────────────────
    TYPE_OPTS = ["Obligacje Skarbowe PL", "ETF US", "ETF Tech", "ETF Europe",
                 "Akcja", "Krypto", "Surowce", "Obligacje", "Inne"]
    SEG_OPTS  = ["Bezpieczny", "Ryzykowny"]

    edited_df = st.data_editor(
        st.session_state["gs_portfolio_df"].drop(columns=[COL_AMOUNT]),
        num_rows="dynamic",
        use_container_width=True,
        key="gs_portfolio_editor",
        column_config={
            COL_NAME:   st.column_config.TextColumn(
                COL_NAME, help="Ticker Yahoo Finance (np. SPY) lub opis własny (np. Obligacje Skarbowe 3Y)",
                max_chars=40
            ),
            COL_TYPE:   st.column_config.SelectboxColumn(COL_TYPE, options=TYPE_OPTS),
            COL_YIELD:  st.column_config.NumberColumn(
                COL_YIELD, min_value=0.0, max_value=30.0, step=0.01, format="%.2f",
                help="Oprocentowanie roczne brutto (%). Dla akcji/ETF wpisz 0."
            ),
            COL_WEIGHT: st.column_config.NumberColumn(
                COL_WEIGHT, min_value=0.01, max_value=100.0, step=0.5, format="%.2f",
                help="Udział procentowy w całym portfelu. Suma WSZYSTKICH wierszy = 100%."
            ),
            COL_SEG:    st.column_config.SelectboxColumn(COL_SEG, options=SEG_OPTS,
                help="Bezpieczny = obligacje/cash; Ryzykowny = akcje/ETF/krypto"),
        },
    )

    # Aktualizuj kwoty (wyliczone)
    if not edited_df.empty:
        edited_df[COL_AMOUNT] = (edited_df[COL_WEIGHT] / 100.0 * new_capital).round(0)
    st.session_state["gs_portfolio_df"] = edited_df if COL_AMOUNT not in edited_df.columns else edited_df

    # ── Podgląd kwot ──────────────────────────────────────────────────────
    if not edited_df.empty and COL_WEIGHT in edited_df.columns:
        preview_df = edited_df.copy()
        preview_df[COL_AMOUNT] = (preview_df[COL_WEIGHT] / 100.0 * new_capital).round(0).astype(int).apply(lambda x: f"{x:,} PLN")
        st.dataframe(
            preview_df[[COL_NAME, COL_SEG, COL_WEIGHT, COL_AMOUNT]].rename(columns={COL_WEIGHT: "Waga %"}),
            use_container_width=True, hide_index=True
        )

    # Walidacja sumy wag
    total_w = edited_df[COL_WEIGHT].sum() if not edited_df.empty and COL_WEIGHT in edited_df.columns else 0
    col_val1, col_val2 = st.columns([3, 2])
    with col_val1:
        if abs(total_w - 100.0) > 0.5:
            st.error(f"⚠️ Suma wag = **{total_w:.2f}%** (wymagane 100%). Różnica: {total_w-100:+.2f}%")
            wt_ok = False
        else:
            st.success(f"✅ Suma wag = {total_w:.2f}% — portfel poprawnie skonfigurowany")
            wt_ok = True

    # Pasek alokacji
    safe_allocated  = edited_df.loc[edited_df[COL_SEG]  == "Bezpieczny",  COL_WEIGHT].sum() if not edited_df.empty else 0
    risky_allocated = edited_df.loc[edited_df[COL_SEG] == "Ryzykowny", COL_WEIGHT].sum() if not edited_df.empty else 0
    if total_w > 0:
        sp = safe_allocated / total_w * 100
        rp = risky_allocated / total_w * 100
        st.markdown(f"""
        <div style="border-radius:8px;overflow:hidden;height:28px;display:flex;margin:8px 0 16px 0;">
            <div style="background:linear-gradient(90deg,#1e40af,#3b82f6);width:{sp:.1f}%;
                        display:flex;align-items:center;justify-content:center;
                        font-size:12px;font-weight:700;color:white;">
                🔒 {sp:.0f}% Bezpieczny
            </div>
            <div style="background:linear-gradient(90deg,#dc2626,#f97316);width:{rp:.1f}%;
                        display:flex;align-items:center;justify-content:center;
                        font-size:12px;font-weight:700;color:white;">
                ⚡ {rp:.0f}% Ryzykowny
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Szybkie skróty ────────────────────────────────────────────────────
    st.markdown("#### 💡 Popularne aktywa — kliknij żeby skopiować ticker")
    quick_cols = st.columns(6)
    QUICK_ASSETS = [
        ("SPY", "ETF US"), ("QQQ", "ETF Tech"), ("VTI", "ETF US"),
        ("GLD", "Surowce"), ("BTC-USD", "Krypto"), ("NVDA", "Akcja"),
        ("TLT", "Obligacje"), ("EEM", "ETF"), ("MSFT", "Akcja"),
        ("AAPL", "Akcja"), ("MSTR", "Akcja"), ("IAU", "Surowce"),
    ]
    for i, (tick, cls) in enumerate(QUICK_ASSETS):
        quick_cols[i % 6].markdown(
            f"<span style='background:rgba(0,204,255,0.08);border:1px solid rgba(0,204,255,0.2);"
            f"border-radius:5px;padding:2px 8px;font-size:11px;color:#00ccff;cursor:pointer;display:inline-block;margin:2px'>"
            f"<b>{tick}</b> <span style='color:#6b7280'>{cls}</span></span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Nazwa profilu i przyciski zapisu ──────────────────────────────────
    st.markdown("#### 💾 Zapis i Zastosowanie")
    pcol1, pcol2 = st.columns([2, 5])
    with pcol1:
        profile_name_input = st.text_input(
            "Nazwa profilu",
            value=gs.profile_name,
            key="gs_v2_profile_name",
            help="Własna nazwa profilu, np. 'Mój Portfel 2025'"
        )

    bcol1, bcol2, bcol3, _ = st.columns([2, 2, 2, 3])

    def _build_new_gs_from_editor():
        """Buduje nowy GlobalPortfolio z edytora tabeli."""
        df = edited_df.copy()
        if COL_AMOUNT not in df.columns:
            df[COL_AMOUNT] = (df[COL_WEIGHT] / 100.0 * new_capital).round(0)

        portfolio_assets = []
        for _, row in df.iterrows():
            name = str(row.get(COL_NAME, "")).strip()
            if not name:
                continue
            ticker = name.upper() if not any(c in name for c in [" ", "."]) else name
            portfolio_assets.append({
                "name":      name,
                "ticker":    ticker,
                "type":      str(row.get(COL_TYPE, "Inne")),
                "yield_pct": float(row.get(COL_YIELD, 0.0)),
                "weight":    float(row.get(COL_WEIGHT, 0.0)),
                "segment":   str(row.get(COL_SEG, "Ryzykowny")),
            })

        new_gs = GlobalPortfolio(
            safe_type=gs.safe_type,
            safe_rate=gs.safe_rate,
            safe_tickers=gs.safe_tickers,
            risky_assets=gs.risky_assets,
            alloc_safe_pct=gs.alloc_safe_pct,
            initial_capital=new_capital,
            base_currency=gs.base_currency,
            currency_risk_enabled=gs.currency_risk_enabled,
            usd_pln_vol=gs.usd_pln_vol,
            usd_pln_corr=gs.usd_pln_corr,
            bg_refresh_enabled=gs.bg_refresh_enabled,
            bg_refresh_interval_minutes=gs.bg_refresh_interval_minutes,
            language=gs.language,
            visible_modules=gs.visible_modules,
            ui_mode=gs.ui_mode,
            profile_name=profile_name_input,
            portfolio_assets=portfolio_assets,
        )
        new_gs.sync_from_portfolio_assets()
        return new_gs

    with bcol1:
        if st.button("💾 Zapisz domyślnie", type="primary",
                     use_container_width=True, disabled=not wt_ok):
            new_gs = _build_new_gs_from_editor()
            ok = save_global_settings(new_gs)
            set_gs(new_gs)
            force_apply_gs_to_session(new_gs)
            if "gs_portfolio_df" in st.session_state:
                del st.session_state["gs_portfolio_df"]
            if ok:
                st.toast(f"✅ Portfel '{profile_name_input}' zapisany do {GLOBAL_SETTINGS_PATH}", icon="✅")
                st.rerun()
            else:
                st.error("❌ Błąd zapisu pliku.")

    with bcol2:
        if st.button("📤 Zastosuj (bez zapisu)", use_container_width=True, disabled=not wt_ok):
            new_gs = _build_new_gs_from_editor()
            set_gs(new_gs)
            force_apply_gs_to_session(new_gs)
            st.toast("📤 Ustawienia zastosowane (sesja)", icon="📤")
            st.rerun()

    with bcol3:
        if st.button("🔄 Przywróć fabryczne", use_container_width=True):
            factory = GlobalPortfolio()
            set_gs(factory)
            force_apply_gs_to_session(factory)
            for k in ["gs_portfolio_df", "gs_v2_capital"]:
                st.session_state.pop(k, None)
            st.toast("🔄 Przywrócono ustawienia fabryczne", icon="🔄")
            st.rerun()

    if gs.last_updated:
        st.caption(
            f"📅 Ostatni zapis: `{gs.last_updated}` | "
            f"Profil: **{gs.profile_name}** | Plik: `{GLOBAL_SETTINGS_PATH}`"
        )

# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKA 2 — USTAWIENIA TECHNICZNE
# ══════════════════════════════════════════════════════════════════════════════
with tab_tech:
    st.markdown("### ⚙️ Ustawienia Techniczne Aplikacji")
    st.caption("Parametry silnika, interfejsu i integracji FX. Nie wpływają bezpośrednio na wyniki analiz portfelowych.")

    col_tl, col_tr = st.columns(2, gap="large")

    with col_tl:
        # ── Heartbeat Engine ──────────────────────────────────────────────
        st.markdown(f"<p class='{_SEC}'>🫀 Heartbeat Engine (Dane Makro)</p>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:rgba(0,230,118,0.06);border:1px solid rgba(0,230,118,0.15);
                    border-radius:8px;padding:10px;margin-bottom:12px;font-size:12px;color:#94a3b8;">
        Silnik aktualizuje dane makroekonomiczne (VIX, FCI, krzywa dochodowości itp.)
        w tle co określony interwał. Używane przez <b>Control Center</b> i <b>Doradcę</b>.
        </div>
        """, unsafe_allow_html=True)

        c_hb1, c_hb2 = st.columns([1, 2])
        with c_hb1:
            bg_enabled = st.toggle("🟢 Włącz Engine", value=gs.bg_refresh_enabled, key="gs_v2_bg_en")
        with c_hb2:
            bg_interval = st.slider(
                "Interwał (minuty)", 1, 120,
                value=gs.bg_refresh_interval_minutes,
                disabled=not bg_enabled, key="gs_v2_bg_int",
            )

        st.markdown("---")

        # ── Język ─────────────────────────────────────────────────────────
        st.markdown(f"<p class='{_SEC}'>🌐 Język Interfejsu</p>", unsafe_allow_html=True)
        lang_opts = ["🇵🇱 Polski", "🇬🇧 English"]
        lang_idx  = 0 if gs.language == "pl" else 1
        lang_sel  = st.radio("Język", lang_opts, index=lang_idx, key="gs_v2_lang",
                             horizontal=True, label_visibility="collapsed")
        new_language = "pl" if lang_sel == lang_opts[0] else "en"

        st.markdown("---")

        # ── Tryb UI ───────────────────────────────────────────────────────
        st.markdown(f"<p class='{_SEC}'>🎓 Tryb Interfejsu</p>", unsafe_allow_html=True)
        st.caption("Expert = kompaktowy. Educational = pełne objaśnienia matematyczne.")
        ui_opts = ["🔬 Expert (Kompaktowy)", "📚 Educational (Szczegółowy)"]
        ui_idx  = 0 if gs.ui_mode == "expert" else 1
        ui_sel  = st.radio("Tryb", ui_opts, index=ui_idx, key="gs_v2_ui",
                           horizontal=True, label_visibility="collapsed")
        new_ui_mode = "expert" if ui_sel == ui_opts[0] else "educational"

    with col_tr:
        # ── Ryzyko walutowe ───────────────────────────────────────────────
        st.markdown(f"<p class='{_SEC}'>💱 Ryzyko Walutowe USD/PLN</p>", unsafe_allow_html=True)
        st.caption("Parametry symulacji wpływu kursu USD/PLN na aktywa denominowane w dolarach.")
        new_currency_risk = st.toggle(
            "Uwzględnij ryzyko walutowe FX w symulacjach",
            value=gs.currency_risk_enabled, key="gs_v2_fx_en"
        )
        col_fx1, col_fx2 = st.columns(2)
        with col_fx1:
            new_fx_vol = st.number_input(
                "Vol USD/PLN (roczna)", min_value=0.01, max_value=0.50,
                value=float(gs.usd_pln_vol), step=0.01, format="%.2f",
                key="gs_v2_fx_vol", disabled=not new_currency_risk
            )
        with col_fx2:
            new_fx_corr = st.number_input(
                "Korelacja FX / S&P500", min_value=-1.0, max_value=1.0,
                value=float(gs.usd_pln_corr), step=0.05, format="%.2f",
                key="gs_v2_fx_corr", disabled=not new_currency_risk
            )

        st.markdown("---")

        # ── Widoczne moduły ───────────────────────────────────────────────
        st.markdown(f"<p class='{_SEC}'>🎛️ Widoczność Modułów w Menu</p>", unsafe_allow_html=True)
        all_mods = [
            "Factor Zoo & PCA", "EVT — Tail Risk", "Black-Litterman AI",
            "DCC — Korelacje", "Stress Test", "Symulator Barbell", "Skaner Rynku",
            "Emerytura / FIRE", "Portfolio Health Monitor", "Concentration Risk",
            "Drawdown Recovery", "Investment Clock", "Regime Allocation",
            "Liquidity Risk", "Tail Risk Hedging", "Tax Optimizer PL",
            "Smart Rebalancing", "Sentiment & Flow", "Alt. Risk Premia",
            "Wealth Optimizer", "Life OS — Łowca", "Day Trading",
            "Walk-Forward CPCV", "Systemic Risk & CoVaR",
            "Recession Nowcasting", "Decumulation / SWR", "Doradca AI",
        ]
        default_vis = [m for m in (gs.visible_modules or all_mods) if m in all_mods]
        new_visible = st.multiselect(
            "Widoczne moduły", all_mods, default=default_vis,
            key="gs_v2_mods", label_visibility="collapsed"
        )

    st.markdown("---")

    # ── Zapis ustawień technicznych ───────────────────────────────────────
    if st.button("💾 Zapisz Ustawienia Techniczne", type="primary"):
        new_gs = GlobalPortfolio(
            safe_type=gs.safe_type,
            safe_rate=gs.safe_rate,
            safe_tickers=gs.safe_tickers,
            risky_assets=gs.risky_assets,
            alloc_safe_pct=gs.alloc_safe_pct,
            initial_capital=gs.initial_capital,
            base_currency=gs.base_currency,
            currency_risk_enabled=new_currency_risk,
            usd_pln_vol=new_fx_vol,
            usd_pln_corr=new_fx_corr,
            bg_refresh_enabled=bg_enabled,
            bg_refresh_interval_minutes=bg_interval,
            language=new_language,
            visible_modules=new_visible,
            ui_mode=new_ui_mode,
            profile_name=gs.profile_name,
            portfolio_assets=gs.portfolio_assets,
        )
        ok = save_global_settings(new_gs)
        set_gs(new_gs)
        force_apply_gs_to_session(new_gs)
        from modules.background_updater import bg_engine
        bg_engine.set_config(bg_enabled, bg_interval)
        if ok:
            st.toast("✅ Ustawienia techniczne zapisane!", icon="✅")
            st.rerun()
        else:
            st.error("❌ Błąd zapisu.")

    st.markdown("---")
    if st.button("🔁 Wymuś Synchronizację Sesji", type="secondary"):
        force_apply_gs_to_session(gs)
        st.success("✅ Ustawienia ponownie wstrzyknięte do sesji wszystkich modułów.")

# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKA 3 — PODGLĄD & ANALIZA
# ══════════════════════════════════════════════════════════════════════════════
with tab_preview:
    st.markdown("### 📊 Wizualizacja Portfela")

    assets = gs.all_assets_unified
    cap    = gs.initial_capital

    if not assets:
        st.info("Brak aktywów. Dodaj je w zakładce 💼 Mój Portfel.")
    else:
        # ── Pie chart safe vs risky ─────────────────────────────────────
        col_p1, col_p2, col_p3 = st.columns(3)

        safe_assets  = [a for a in assets if a.get("segment") == "Bezpieczny"]
        risky_assets_list = [a for a in assets if a.get("segment") == "Ryzykowny"]
        safe_total   = sum(a["weight"] for a in safe_assets) or 0
        risky_total  = sum(a["weight"] for a in risky_assets_list) or 0

        with col_p1:
            st.markdown("**Podział Bezpieczny / Ryzykowny**")
            fig_split = go.Figure(go.Pie(
                labels=["🔒 Bezpieczny", "⚡ Ryzykowny"],
                values=[safe_total, risky_total],
                hole=0.55,
                marker_colors=["#3b82f6", "#f97316"],
                texttemplate="%{percent:.0%}",
                textfont_size=14,
            ))
            fig_split.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                height=280, margin=dict(t=10, b=10, l=10, r=10),
                legend=dict(font=dict(size=12)),
                annotations=[dict(
                    text=f"{gs.blended_rate*100:.1f}%\nblended",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=15, color="white", family="Inter"),
                )],
            )
            st.plotly_chart(fig_split, use_container_width=True)

        with col_p2:
            if risky_assets_list:
                st.markdown("**Struktura Części Ryzykownej**")
                colors_r = ["#ff6384","#ff9f40","#ffcd56","#4bc0c0","#36a2eb","#9966ff","#c9cbcf","#ff6384"]
                fig_r = go.Figure(go.Pie(
                    labels=[a.get("name", a.get("ticker","?")) for a in risky_assets_list],
                    values=[a["weight"] for a in risky_assets_list],
                    hole=0.40,
                    marker_colors=colors_r[:len(risky_assets_list)],
                    texttemplate="%{label}<br>%{percent:.0%}",
                    textfont_size=11,
                ))
                fig_r.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
                )
                st.plotly_chart(fig_r, use_container_width=True)

        with col_p3:
            if safe_assets:
                st.markdown("**Struktura Części Bezpiecznej**")
                colors_s = ["#3b82f6","#60a5fa","#93c5fd","#bfdbfe"]
                fig_s = go.Figure(go.Pie(
                    labels=[a.get("name", a.get("ticker","?")) for a in safe_assets],
                    values=[a["weight"] for a in safe_assets],
                    hole=0.40,
                    marker_colors=colors_s[:len(safe_assets)],
                    texttemplate="%{label}<br>%{percent:.0%}",
                    textfont_size=11,
                ))
                fig_s.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
                )
                st.plotly_chart(fig_s, use_container_width=True)

        st.markdown("---")

        # ── Tabela szczegółowa ─────────────────────────────────────────────
        st.markdown("**📋 Szczegółowa Tabela Portfela**")
        rows_detail = []
        for a in assets:
            w = a.get("weight", 0)
            rows_detail.append({
                "Segment":        "🔒 Bezpieczny" if a.get("segment") == "Bezpieczny" else "⚡ Ryzykowny",
                "Nazwa":          a.get("name", a.get("ticker", "?")),
                "Ticker":         a.get("ticker", "—"),
                "Typ":            a.get("type", "Inne"),
                "Oprocent. %":    f"{a.get('yield_pct', 0):.2f}%" if a.get("yield_pct") else "—",
                "Waga %":         f"{w:.2f}%",
                "Kwota PLN":      f"{w / 100 * cap:,.0f}",
            })
        st.dataframe(pd.DataFrame(rows_detail), use_container_width=True, hide_index=True)

        # ── Bar chart ──────────────────────────────────────────────────────
        labels_bar = [a.get("name", a.get("ticker","?")) for a in assets]
        values_bar = [a["weight"] / 100 * cap for a in assets]
        colors_bar = ["#3b82f6" if a.get("segment") == "Bezpieczny" else "#f97316" for a in assets]

        fig_bar = go.Figure(go.Bar(
            x=labels_bar, y=values_bar,
            marker_color=colors_bar,
            text=[f"{v:,.0f} PLN" for v in values_bar],
            textposition="outside",
        ))
        fig_bar.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,11,20,0.6)",
            height=350, showlegend=False,
            yaxis_title="Kwota (PLN)",
            xaxis_title="Aktywo",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # ── Tabela propagacji ──────────────────────────────────────────────
        st.markdown("**🔁 Status Propagacji do Modułów**")
        prop_data = [
            {"Moduł": "1_Symulator (MC)", "Parametr": "Alokacja bezpieczna",
             "Wartość": f"{gs.alloc_safe_pct:.0%}", "Klucz": "_s.mc_alloc_safe", "Status": "✅ Aktywny"},
            {"Moduł": "1_Symulator (MC)", "Parametr": "Kapitał startowy",
             "Wartość": f"{gs.initial_capital:,.0f} PLN", "Klucz": "_s.mc_cap", "Status": "✅ Aktywny"},
            {"Moduł": "1_Symulator (AI)", "Parametr": "Stopa TOS",
             "Wartość": f"{gs.safe_rate*100:.2f}%", "Klucz": "_s.ai_safe_rate", "Status": "✅ Aktywny"},
            {"Moduł": "1_Symulator (AI)", "Parametr": "Tickery ryzykowne",
             "Wartość": gs.risky_tickers_str[:35]+"…" if len(gs.risky_tickers_str)>35 else gs.risky_tickers_str,
             "Klucz": "_s.ai_risky_tickers", "Status": "✅ Aktywny"},
            {"Moduł": "3_Stress Test", "Parametr": "Wagi Safe/Risky",
             "Wartość": f"{gs.alloc_safe_pct:.0%}/{gs.alloc_risky_pct:.0%}", "Klucz": "_s.st_sw", "Status": "✅ Aktywny"},
            {"Moduł": "19_Wealth Optimizer", "Parametr": "Majątek startowy",
             "Wartość": f"{gs.initial_capital:,.0f} PLN", "Klucz": "_gs_wealth_total", "Status": "✅ Aktywny"},
            {"Moduł": "28_Doradca AI", "Parametr": "Cały portfel",
             "Wartość": "portfolio_assets + macro cache", "Klucz": "AdvisorEngine(gs)", "Status": "✅ Aktywny"},
        ]
        st.dataframe(pd.DataFrame(prop_data), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKA 4 — EMERYTURA / FIRE (NOWA)
# ══════════════════════════════════════════════════════════════════════════════
with tab_ret:
    st.markdown("### 🏖️ Parametry Emerytalne (FIRE)")
    st.caption("Ustawienia dla modułów planowania finansowego, decumulacji, SWR oraz symulacji wieku emerytalnego.")

    ret_c1, ret_c2 = st.columns(2)
    with ret_c1:
        st.markdown(f"<p class='{_SEC}'>Wiek i Czas</p>", unsafe_allow_html=True)
        new_curr_age = st.number_input(
            "Obecny wiek (Lata)", min_value=18, max_value=100, value=gs.ret_current_age, step=1
        )
        new_target_age = st.number_input(
            "Docelowy wiek emerytury / FIRE (Lata)", min_value=new_curr_age, max_value=120, value=max(gs.ret_target_age, new_curr_age), step=1
        )
        years_to_fire = new_target_age - new_curr_age
        st.info(f"⏳ Czas do emerytury (akumulacja): **{years_to_fire} lat**")

        st.markdown(f"<p class='{_SEC}'>Przepływy Miesięczne</p>", unsafe_allow_html=True)
        new_monthly_contrib = st.number_input(
            "Miesięczne wpłaty do portfela (PLN)", min_value=0.0, value=float(gs.ret_monthly_contribution), step=500.0, format="%.2f"
        )
        new_monthly_exp = st.number_input(
            "Miesięczne zapotrzebowanie / Wydatki na emeryturze (PLN)", min_value=0.0, value=float(gs.ret_monthly_expense), step=500.0, format="%.2f"
        )

    with ret_c2:
        st.markdown(f"<p class='{_SEC}'>Zmienne Ekonomiczne</p>", unsafe_allow_html=True)
        new_inflation = st.slider(
            "Szacowana Długoterminowa Inflacja (%)", min_value=0.0, max_value=15.0, value=gs.ret_inflation_rate*100, step=0.1
        ) / 100.0
        new_swr = st.slider(
            "Safe Withdrawal Rate (SWR) (%)", min_value=1.0, max_value=10.0, value=gs.ret_swr_rate*100, step=0.1,
            help="Zalecany standard to 4%. Oznacza jaki procent zgromadzonego kapitału rocznie (skorygowany o inflację) jest bezpiecznie wypłacany przez lata emerytury bez ryzyka bankructwa."
        ) / 100.0
        
        needed_capital = (new_monthly_exp * 12) / (new_swr if new_swr > 0 else 0.04)
        st.success(f"🎯 Szacowany potrzebny kapitał docelowy (wg SWR): **{needed_capital:,.0f} PLN**")

    st.markdown("---")
    if st.button("💾 Zapisz Parametry Emerytury", type="primary"):
        # Budujemy kopię, by nie nadpisać niechcący tabel
        new_gs = GlobalPortfolio(
            safe_type=gs.safe_type,
            safe_rate=gs.safe_rate,
            safe_tickers=gs.safe_tickers,
            risky_assets=gs.risky_assets,
            alloc_safe_pct=gs.alloc_safe_pct,
            initial_capital=gs.initial_capital,
            base_currency=gs.base_currency,
            currency_risk_enabled=gs.currency_risk_enabled,
            usd_pln_vol=gs.usd_pln_vol,
            usd_pln_corr=gs.usd_pln_corr,
            bg_refresh_enabled=gs.bg_refresh_enabled,
            bg_refresh_interval_minutes=gs.bg_refresh_interval_minutes,
            language=gs.language,
            visible_modules=gs.visible_modules,
            ui_mode=gs.ui_mode,
            profile_name=gs.profile_name,
            portfolio_assets=gs.portfolio_assets,
            ret_current_age=new_curr_age,
            ret_target_age=new_target_age,
            ret_monthly_contribution=new_monthly_contrib,
            ret_monthly_expense=new_monthly_exp,
            ret_inflation_rate=new_inflation,
            ret_swr_rate=new_swr,
        )
        ok = save_global_settings(new_gs)
        set_gs(new_gs)
        force_apply_gs_to_session(new_gs)
        if ok:
            st.toast("✅ Parametry FIRE zapisane i spropagowane na wszystkie moduły!", icon="✅")
            st.rerun()
        else:
            st.error("❌ Błąd zapisu danych.")

# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKA 5 — PROFILE PRESETÓW
# ══════════════════════════════════════════════════════════════════════════════
with tab_profiles:
    st.markdown("### 🎯 Gotowe Profile Portfela")
    st.markdown("Załaduj jeden z predefiniowanych profili jako punkt startowy, "
                "a następnie dostosuj go w zakładce **💼 Mój Portfel**.")

    for preset_name, preset_data in PRESET_PROFILES.items():
        risky_preview = ", ".join(
            f"{a['ticker']} {a['weight']:.0f}%" for a in preset_data["risky_assets"]
        )
        alloc_safe = preset_data["alloc_safe_pct"]

        with st.expander(f"**{preset_name}**  |  🔒 {alloc_safe:.0%} / ⚡ {1-alloc_safe:.0%}", expanded=False):
            c1, c2, c3 = st.columns([2, 4, 2])
            with c1:
                st.metric("Bezpieczna", f"{alloc_safe:.0%}")
                st.metric("Stopa TOS",  f"{preset_data['safe_rate']*100:.2f}%")
            with c2:
                st.markdown(f"**Aktywa Ryzykowne:**  \n{risky_preview}")
                safe_info = "Obligacje Skarbowe PL (stała stopa)" if preset_data["safe_type"] == "fixed" \
                            else ", ".join(preset_data.get("safe_tickers", []))
                st.markdown(f"**Bezpieczna:**  \n{safe_info}")
            with c3:
                if st.button("⬇️ Załaduj Profil", key=f"preset_{preset_name}", use_container_width=True):
                    # Buduj portfolio_assets dla presetu
                    pa = [{
                        "name": "Obligacje Skarbowe 3Y" if preset_data["safe_type"] == "fixed" else tkr,
                        "ticker": "TOS_PL" if preset_data["safe_type"] == "fixed" else tkr,
                        "type": "Obligacje Skarbowe PL" if preset_data["safe_type"] == "fixed" else "ETF",
                        "yield_pct": preset_data["safe_rate"] * 100 if preset_data["safe_type"] == "fixed" else 0.0,
                        "weight": preset_data["alloc_safe_pct"] * 100,
                        "segment": "Bezpieczny",
                    }]
                    risky_total = sum(a["weight"] for a in preset_data["risky_assets"]) or 1
                    for ra in preset_data["risky_assets"]:
                        pa.append({
                            "name": ra["ticker"],
                            "ticker": ra["ticker"],
                            "type": ra.get("asset_class", "ETF US"),
                            "yield_pct": 0.0,
                            "weight": ra["weight"] / risky_total * preset_data.get("alloc_risky_pct",
                                       1 - preset_data["alloc_safe_pct"]) * 100,
                            "segment": "Ryzykowny",
                        })

                    preset_gs = GlobalPortfolio(
                        safe_type=preset_data["safe_type"],
                        safe_rate=preset_data["safe_rate"],
                        safe_tickers=preset_data.get("safe_tickers", []),
                        risky_assets=copy.deepcopy(preset_data["risky_assets"]),
                        alloc_safe_pct=preset_data["alloc_safe_pct"],
                        initial_capital=preset_data.get("initial_capital", gs.initial_capital),
                        language=gs.language,
                        visible_modules=gs.visible_modules,
                        ui_mode=gs.ui_mode,
                        profile_name=preset_data["profile_name"],
                        portfolio_assets=pa,
                    )
                    set_gs(preset_gs)
                    force_apply_gs_to_session(preset_gs)
                    for k in ["gs_portfolio_df"]:
                        st.session_state.pop(k, None)
                    st.toast(f"⬇️ Załadowano profil: {preset_name}", icon="⬇️")
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# ZAKŁADKA 6 — ODKRYWCA ISIN
# ══════════════════════════════════════════════════════════════════════════════
with tab_isin:
    st.markdown("### 🔍 Wyszukiwarka i Weryfikacja ISIN")
    st.markdown(
        "Dzięki systemowi ISIN Resolver, cała aplikacja obsługuje **przezroczyste tłumaczenie numerów ISIN** "
        "(np. `IE00B4L5Y983`) na Tickery giełdowe w locie. "
        "Możesz bez obaw wklejać identyfikatory w pola **Ticker**."
    )

    from modules.ui.widgets import ticker_input
    isin_query = ticker_input("Podaj numer ISIN:", placeholder="np. IE00B4L5Y983", key="gs_isin_q")

    if st.button("🔎 Szukaj odpowiednika", type="primary"):
        from modules.isin_resolver import ISINResolver
        if not ISINResolver.is_isin(isin_query):
            st.warning("⚠️ Podany ciąg nie wygląda na prawidłowy 12-znakowy kod ISIN.")
        else:
            with st.spinner("Przeszukiwanie baz Yahoo Finance..."):
                res = ISINResolver.search_isin(isin_query)
                if res:
                    sym = res.get("symbol", "Brak")
                    st.success(f"Odnaleziono: **{isin_query.upper()}** ➔ **{sym}**")
                    df_res = pd.DataFrame([{
                        "Ticker (Zmapowany)": sym,
                        "Pełna Nazwa": res.get("longname") or res.get("shortname", "-"),
                        "Giełda": res.get("exchange", "-"),
                        "Typ": res.get("quoteType", "-"),
                    }])
                    st.dataframe(df_res, use_container_width=True, hide_index=True)
                    st.info(f"💡 Po wklejeniu `{isin_query.upper()}` w innych modułach, "
                            f"automatycznie użyte zostaną notowania `{sym}`.")
                else:
                    st.error(f"Nie znaleziono Tickera dla: **{isin_query.upper()}**. "
                             f"Użyj ręcznie oficjalnego skrótu (np. CSPX.AS).")

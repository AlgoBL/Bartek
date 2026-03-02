
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from modules.styling import apply_styling
from modules.logger import setup_logger

logger = setup_logger(__name__)
from modules.simulation import simulate_barbell_strategy, calculate_metrics, run_ai_backtest, calculate_individual_metrics
from modules.metrics import (
    calculate_trade_stats, calculate_omega, calculate_ulcer_index,
    calculate_pain_index, calculate_drawdown_analytics, calculate_max_drawdown
)
from modules.ai.data_loader import load_data
from modules.analysis_content import display_analysis_report, display_scanner_methodology, display_chart_guide
from modules.scanner import calculate_convecity_metrics, score_asset, compute_hierarchical_dendrogram
from modules.ai.scanner_engine import ScannerEngine
from modules.ai.asset_universe import (
    get_sp500_tickers, get_global_etfs, get_top100_etfs,
    get_wig20_tickers, get_stoxx50_tickers, get_crypto_tickers
)
from modules.ui.status_manager import StatusManager
from modules.ai.observer import REGIME_BULL_QUIET, REGIME_BULL_VOL, REGIME_BEAR, REGIME_CRISIS

# ... existsing code ...

# 1. Page Configuration (handled by app.py)

# 2. Apply Custom Styling
st.markdown(apply_styling(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Persystencja ustawień między modułami
# Streamlit USUWA klucze widżetów z session_state gdy widżet nie jest renderowany.
# Rozwiązanie: on_change zapisuje do "_s.<key>", a value= czyta z tego klucza.
# ---------------------------------------------------------------------------
def _save(wk):
    """Callback on_change: kopiuje wartość widżetu do trwałego klucza."""
    st.session_state[f"_s.{wk}"] = st.session_state[wk]

def _saved(wk, default):
    """Zwraca ostatnio zapisaną wartość lub domyślną."""
    return st.session_state.get(f"_s.{wk}", default)

# Klucze pomocnicze (nie-widżetowe) dla modułu Emerytura
if "rem_initial_capital" not in st.session_state:
    st.session_state["rem_initial_capital"] = 1000000.0
if "rem_expected_return" not in st.session_state:
    st.session_state["rem_expected_return"] = 0.07
if "rem_volatility" not in st.session_state:
    st.session_state["rem_volatility"] = 0.15


# Navigation handled by Streamlit natively.

if "custom_stress_scenarios" not in st.session_state:
    st.session_state["custom_stress_scenarios"] = {}

# 3. Main Navigation

st.header("🌍 Skaner V6.0")
st.markdown("""
**Architektura V6**: Wielu-agentowy system globalnej syntezy rynkowej.
Oracle → Ekonomista → Geopolityk (FinBERT NLP) → CIO → EVT Engine → Composite Barbell Score.
Dane przetwarzane przez **Polars** (Arrow backend) dla maksymalnej wydajności.
""")

# ── Backend Status Badges ─────────────────────────────────────────────
from modules.ai.agents import get_sentiment_backend
_sent_backend = get_sentiment_backend()
_sent_badge = "🟢 **FinBERT** (ProsusAI — NLP finansowy)" if _sent_backend == "finbert" \
    else ("🟡 **VADER** (fallback — brak transformers)" if _sent_backend == "vader" \
    else "🔴 **Brak NLP** (brak bibliotek)")

try:
    import polars as _pl
    _polars_badge = "🟢 **Polars** (Arrow — 10–50× szybszy)"
except ImportError:
    _polars_badge = "🟡 **Pandas** (fallback — brak polars)"

_badge_col1, _badge_col2 = st.columns(2)
_badge_col1.markdown(f"🧠 NLP Sentyment: {_sent_badge}")
_badge_col2.markdown(f"⚡ DataFrame Engine: {_polars_badge}")
st.markdown("---")

# ── Celery async toggle ───────────────────────────────────────────────
try:
    from tasks import scan_task, get_task_status, is_celery_available
    _celery_ok = is_celery_available()
except Exception as e:
    logger.warning(f"Błąd inicjalizacji Celery: {e}")
    _celery_ok = False
    scan_task = None

_use_celery = False
if _celery_ok:
    st.sidebar.markdown("### ⚙️ Ustawienia")
    _use_celery = st.sidebar.checkbox(
        "⚡ Tryb Async (Celery + Redis)",
        value=False,
        help="Uruchamia skan w tle przez kolejkę Redis. Wymaga: `celery -A tasks worker --pool=solo`"
    )
else:
    st.sidebar.info("ℹ️ Async (Celery): Redis niedostępny — tryb synchroniczny.")

col_scan1, col_scan2 = st.columns([3, 1])

with col_scan2:
    scan_months = st.slider(
        "📅 Horyzont Inwestycyjny (Miesiące)",
        min_value=1, max_value=120, value=60, step=1,
        help="Ilość miesięcy do prześwietlenia historii aktywów (1 = 1M, 12 = 1 rok, 60 = 5 lat)"
    )
    scan_years = round(scan_months / 12, 2)
    if scan_months < 12:
        st.caption(f"⏱️ {scan_months} mies. (≈{scan_months * 30} dni)")
    elif scan_months % 12 == 0:
        st.caption(f"⏱️ {scan_months} mies. = {scan_months // 12} lat")
    else:
        st.caption(f"⏱️ {scan_months} mies. = {scan_years:.1f} roku")
    st.markdown("")
    scan_btn = st.button("🚀 Uruchom Globalną Syntezę", type="primary")

# ── Celery async polling ──────────────────────────────────────────────
if _use_celery and 'celery_scan_task_id' in st.session_state:
    _tid = st.session_state['celery_scan_task_id']
    _status = get_task_status(_tid)
    if _status['state'] == 'SUCCESS':
        import pandas as _pd_celery
        _raw = _status['result']
        if 'metrics_df_json' in _raw:
            _raw['metrics_df'] = _pd_celery.read_json(_raw.pop('metrics_df_json'), orient='records')
        st.session_state['v5_scanner_results'] = _raw
        st.session_state.pop('celery_scan_task_id', None)
        st.toast("✅ Skan async zakończony!")
        st.rerun()
    elif _status['state'] == 'PROGRESS':
        st.progress(_status['progress'] / 100, text=_status.get('message', ''))
        st.info("⏳ Skan działa w tle... możesz korzystać z innych modułów.")
        st.rerun()
    elif _status['state'] == 'FAILURE':
        st.error(f"❌ Błąd Celery: {_status.get('error', '')}") 
        st.session_state.pop('celery_scan_task_id', None)

if scan_btn:
    if _use_celery and scan_task is not None:
        # ── ASYNC: wyślij do Celery ───────────────────────────────
        _task = scan_task.delay(horizon_years=int(scan_years))
        st.session_state['celery_scan_task_id'] = _task.id
        st.info(f"⚡ Skan wysłany do kolejki Celery (task_id: `{_task.id}`). Możesz korzystać z aplikacji.")
        st.rerun()
    else:
        # ── SYNC: tradycyjne uruchomienie ─────────────────────────
        engine = ScannerEngine()
        status_scan = StatusManager("Autonomiczna fuzja danych V6 uruchomiona...", expanded=True)
        progress_scan = st.progress(0)

        def terminal_update(pct, msg):
            progress_scan.progress(pct, text=msg)
            if pct < 0.2: status_scan.info_data(msg)
            elif pct < 0.5: status_scan.info_ai(msg)
            elif pct < 0.8: status_scan.info_math(msg)

        try:
            v5_results = engine.run_v5_autonomous_scan(int(scan_years), progress_callback=terminal_update)
            status_scan.success("Kwantowy re-balancing i selekcja zakończone!")
            st.toast("System Algorytmiczny podjął decyzje inwestycyjne.")
            st.session_state['v5_scanner_results'] = v5_results
        except Exception as e:
            status_scan.error(f"Krytyczny błąd w Silniku V6: {e}")
         
# Renderowanie Wyników V5 (API-Free)
if 'v5_scanner_results' in st.session_state:
    res = st.session_state['v5_scanner_results']

    econ = res['econ_report']
    geo  = res['geo_report']
    cio  = res['cio_thesis']
    macro = res.get('macro_snapshot', {})

    # ── Zegary Instrumentalne ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🎛️ Zegary Instrumentalne: Barbell Nowcast")

    def make_gauge(val, title, r_min, r_max, steps, suffix=""):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            title={'text': title, 'font': {'size': 18}},
            number={'font': {'size': 32}, 'suffix': suffix},
            gauge={
                'axis': {'range': [r_min, r_max], 'tickwidth': 1},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': steps,
                'threshold': {
                    'line': {'color': "white", 'width': 5},
                    'thickness': 0.85, 'value': val
                }
            }
        ))
        fig.update_layout(
            height=280, margin=dict(l=20, r=20, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}
        )
        return fig

    col_c1, col_c2, col_c3 = st.columns(3)

    with col_c1:
        st.markdown("**📊 Nowcast Ryzyka Makro (7 czynników)**")
        fig_econ = make_gauge(
            econ['score'], "Ryzyko Recesji", 0, 8,
            [{'range': [0, 3],   'color': "#2ecc71"},
             {'range': [3, 5.5], 'color': "#f39c12"},
             {'range': [5.5, 8], 'color': "#e74c3c"}]
        )
        st.plotly_chart(fig_econ, use_container_width=True)
        st.info(f"**Stan**: {econ['phase']}")
        with st.expander("🔍 Szczegóły Ekonomisty + FRED", expanded=False):
            for d in econ['details']:
                st.write(f"- {d}")
            if macro:
                st.markdown("**Dane FRED (makro leading):**")
                csp = macro.get('FRED_Credit_Spread_BAA_AAA')
                jcl = macro.get('FRED_Initial_Jobless_Claims')
                vix_ts = macro.get('VIX_TS_Ratio')
                cu_au = macro.get('CuAu_Ratio')
                bkwd = macro.get('VIX_Backwardation', False)
                if csp:    st.metric("Credit Spread (BAA-10Y)", f"{csp:.2f}%", help="> 3.5% = kryzys korporacyjny")
                
                ted = macro.get('FRED_TED_Spread')
                if ted:    st.metric("TED Spread 🆕", f"{ted:.2f}%", help="Ryzyko kredytowe bankowe (LIBOR-TBill)")
                
                hy = macro.get('FRED_HY_Spread')
                if hy:     st.metric("High Yield Spread 🆕", f"{hy:.2f}%", help="BAML Option-Adjusted Spread. > 6% = stres rynkowy")
                
                mv = macro.get('MOVE_Index')
                if mv:     st.metric("MOVE Index 🆕", f"{mv:.1f}", help="VIX dla obligacji (zmienność stóp)")
                
                bdi = macro.get('Baltic_Dry')
                if bdi:    st.metric("Baltic Dry (BDRY) 🆕", f"{bdi:.2f}", help="Proxy handlu globalnego")
                
                fng = macro.get('Crypto_FearGreed')
                if fng:    st.metric("Crypto Fear & Greed 🆕", f"{fng}", delta="Fear" if fng < 40 else "Greed")
                
                if jcl:    st.metric("Wnioski o zasiłek (tygod.)", f"{jcl:,.0f}", help="> 300k = ryzyko recesji")
                if vix_ts: st.metric("VIX Term Structure", f"{vix_ts:.2f}",
                                    delta="BACKWARDATION ⚠️" if bkwd else "Contango ✅")
                if cu_au:  st.metric("Copper/Gold Ratio", f"{cu_au:.4f}",
                                    help="Róśnie = wzrost globalny (risk-on), maleje = risk-off")
                                    
                gex = macro.get("total_gex_billions")
                skew = macro.get("skew_index")
                if gex is not None and skew is not None:
                    st.divider()
                    st.markdown("**🌑 Dark Pools & Options (GEX):**")
                    st.metric("Net Gamma Exposure (SPY GEX)", f"${gex:.1f} Mld", 
                              delta=macro.get("gex_status", ""), 
                              delta_color="normal" if gex > 0 else "inverse",
                              help="Zastępcze szacowanie pozycji Market Makerów na SPY. Dodatni GEX to niska zmienność (kupują spadki, sprzedają wzrosty). Ujemny GEX to rynkowe eldorado zmienności.")
                    msg_skew = "Strach (Drogie Puts)" if skew > 1.2 else ("Chciwość" if skew < 0.8 else "Neutralnie")
                    st.metric("Volatility Skew Index", f"{skew:.2f}", 
                              delta=msg_skew,
                              delta_color="inverse" if skew > 1.2 else "normal",
                              help="Stosunek Implied Volatility: OTM Put / OTM Call. Im wyższy, tym droższe ubezpieczenie od krachu (Smart Money się boi).")

    with col_c2:
        _geo_backend_badge = res.get('sentiment_backend', 'unknown')
        _geo_label = '🟢 FinBERT' if _geo_backend_badge == 'finbert' else ('🟡 VADER' if _geo_backend_badge == 'vader' else '🔴 brak')
        st.markdown(f"**🌍 Geopolityk (NLP: {_geo_label})**")
        fig_geo = make_gauge(
            geo['compound_sentiment'], "Globalny Sentyment", -1, 1,
            [{'range': [-1, -0.15], 'color': "#e74c3c"},
             {'range': [-0.15,  0.15], 'color': "#f39c12"},
             {'range': [ 0.15,  1.0], 'color': "#2ecc71"}]
        )
        st.plotly_chart(fig_geo, use_container_width=True)
        st.info(f"{geo['label']}")
        with st.expander("📰 Prasa Szczegółowo", expanded=False):
            c1g, c2g, c3g = st.columns(3)
            c1g.metric("🟢 Pozytywne", f"{geo.get('positive_pct', 0):.0f}%")
            c2g.metric("🔴 Negatywne", f"{geo.get('negative_pct', 0):.0f}%")
            c3g.metric("⚪ Neutralne",  f"{geo.get('neutral_pct', 0):.0f}%")
            st.caption(f"Przeanalizowano {geo['analyzed_articles']} nagłówków RSS")

    with col_c3:
        st.markdown("**🤵 Dyrektywa Barbella (CIO)**")
        fig_cio = make_gauge(
            cio['gauge_risk_percent'], "Defensywność (%)", 0, 100,
            [{'range': [0,  30], 'color': "#2ecc71"},
             {'range': [30, 70], 'color': "#f39c12"},
             {'range': [70, 100], 'color': "#e74c3c"}],
            suffix="%"
        )
        st.plotly_chart(fig_cio, use_container_width=True)
        st.info(f"**Tryb**: {cio['mode']}")
        with st.expander("🎯 Cele Risky Sleeve", expanded=True):
            st.markdown(f"*{cio['description']}*")
            etf_focus = cio.get('etf_focus', [])
            if etf_focus:
                st.success(f"🔎 ETF Focus: `{'`, `'.join(etf_focus)}`")
            kelly_m = cio.get('kelly_multiplier', 1.0)
            st.metric("Mnożnik Kelly (CIO)", f"{kelly_m:.0%}",
                      help="CIO skaluje wielkość pozycji wg reżimu makro. Risk-Off = zmniejsz ryzyko.")

    # ── TDA Crash Indicator ──────────────────────────────────────────────────
    tda = res.get('tda_results', {})
    if tda and not tda.get('indicator', pd.Series(dtype=float)).empty:
        st.divider()
        st.subheader("🪐 Topological Data Analysis (Betti-0 Crash Indicator)")
        
        tda_c1, tda_c2 = st.columns([1, 2])
        fragility = tda.get('current_fragility', 0.0)
        is_crash = tda.get('crash_warning', False)
        threshold = tda.get('threshold_10p', 0.0)
        
        with tda_c1:
            st.metric("Indeks Kruchości Rynku (Betti-0 Death)", f"{fragility:.4f}",
                      delta="KRACH (Topologiczny)" if is_crash else "Spokojnie",
                      delta_color="inverse" if is_crash else "normal")
            st.info("TDA bada 'kształt' chmury danych rynkowych. Niski czas życia komponentów Betti-0 sugeruje, że rynek zapada się w jeden, silnie skorelowany monolit (Contagion/Zaraza).")
            st.caption(f"Próg ostrzegawczy (10. percentyl): {threshold:.4f}")
            
        with tda_c2:
            tda_series = tda['indicator']
            fig_tda = px.line(x=tda_series.index, y=tda_series.values, 
                              title="Krzywa Persystencji Betti-0 w Czasie")
            fig_tda.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Strefa Krachu")
            fig_tda.update_layout(template="plotly_dark", height=250, margin=dict(t=30, b=10),
                                  yaxis_title="Dystans Śmierci Betti-0", xaxis_title="")
            st.plotly_chart(fig_tda, use_container_width=True)

    # ── Wyniki Rankingu EVT ──────────────────────────────────────────────────
    df_metrics = res['metrics_df']
    selected_tickers = res['top_picks']

    if cio.get('regime') == 'risk_off':
        st.warning("🛑 **Tryb BUNKIER**: CIO wstrzymał skan części ryzykownej Barbella."
                   " W obecnym reżimie makro nie zaleca się ekspozycji na ryzykowne aktywa.")

    if df_metrics.empty:
        st.warning("EVT nie znalazł wystarczająco dużo danych do oceny lub wystąpił błąd.")
    else:
        sort_col = "Barbell Score" if "Barbell Score" in df_metrics.columns else "Score"
        if selected_tickers:
            df_res = df_metrics[df_metrics['Ticker'].isin(selected_tickers)].sort_values(sort_col, ascending=False)
        else:
            df_res = df_metrics.sort_values(sort_col, ascending=False).head(10)

        if 'scanner_data' not in st.session_state:
            start_date = pd.Timestamp.now() - pd.DateOffset(months=int(scan_months))
            st.session_state['scanner_data'] = load_data(
                df_res['Ticker'].tolist(),
                start_date=start_date.strftime("%Y-%m-%d")
            )
        st.session_state['scanner_results'] = df_res
        
        # --- Zapis historii skanów ---
        hist_df = df_res.copy()
        if 'Barbell Score' in hist_df.columns:
            hist_rows = hist_df[['Ticker', 'Barbell Score']].copy()
            hist_rows['ScanDate'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            if 'scan_history' not in st.session_state:
                st.session_state['scan_history'] = hist_rows
            else:
                st.session_state['scan_history'] = pd.concat([st.session_state['scan_history'], hist_rows], ignore_index=True)


if 'scanner_results' in st.session_state:
    df_res = st.session_state['scanner_results']
    data = st.session_state.get('scanner_data', pd.DataFrame()) # Retrieve data for charts
                    
    st.divider()
    st.subheader("🕵️‍♂️ Narzędzia Śledcze (Filtry & Watchlist)")
    
    if 'watchlist' not in st.session_state:
        st.session_state['watchlist'] = ["BTC-USD", "NVDA"] # Default examples
        
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        asset_filter = st.selectbox(
            "Filtruj klasę aktywów:", 
            ["Wszystkie", "Akcje US (S&P 500)", "Europa (STOXX 50)", "Polska (WIG20)", "Krypto", "ETF / Surowce (Top 100)"]
        )
    with col_f2:
        new_watch = st.text_input("Dodaj ticker do Watchlisty (np. AAPL):").strip().upper()
        if new_watch and new_watch not in st.session_state['watchlist']:
            st.session_state['watchlist'].append(new_watch)
            st.rerun()

        watch_disp = ", ".join([f"`{t}`" for t in st.session_state['watchlist']])
        st.markdown(f"**Obserwowane**: {watch_disp if watch_disp else 'Brak'}")
        if st.button("Wyczyść Watchlistę", key="clear_watch"):
            st.session_state['watchlist'] = []
            st.rerun()

    # ── Dedykowane universe dla każdej klasy aktywów ──────────────────────────
    _CLASS_UNIVERSE = {
        "Krypto": get_crypto_tickers(),
        "Polska (WIG20)": get_wig20_tickers(),
        "Akcje US (S&P 500)": get_sp500_tickers(),
        "Europa (STOXX 50)": get_stoxx50_tickers(),
        "ETF / Surowce (Top 100)": get_top100_etfs(),
    }

    # Przycisk „Skanuj tę klasę“ — dostępny gdy filtr != Wszystkie
    if asset_filter != "Wszystkie":
        _uni = _CLASS_UNIVERSE.get(asset_filter, [])
        _btn_col, _info_col = st.columns([1, 3])
        with _btn_col:
            rescan_btn = st.button(
                f"🔍 Skanuj: {asset_filter}",
                key="rescan_class_btn",
                type="primary",
                help=f"Uruchamia pełną analizę EVT tylko dla klasy „{asset_filter}“ ({len(_uni)} aktywów)."
            )
        with _info_col:
            st.caption(
                f"📊 Dedykowany universe: {len(_uni)} aktywów klasy **{asset_filter}**. "
                f"Wynik nadpisze aktualny ranking dla tej klasy."
            )

        if rescan_btn and _uni:
            engine_cls = ScannerEngine()
            _prog = st.progress(0, text=f"Skanowanie {asset_filter}...")
            def _cb(p, m): _prog.progress(min(p, 1.0), text=m)
            with st.spinner(f"Analiza EVT: {asset_filter} ({len(_uni)} aktywów)..."):
                _new_df = engine_cls.scan_markets(_uni, progress_callback=_cb)
            _prog.empty()
            if not _new_df.empty:
                # scalamy z istniejącymi wynikami (usuwamy stare wiersze tej klasy)
                _old = st.session_state['scanner_results'].copy()
                _old_without = _old[~_old['Ticker'].isin(_uni)]
                st.session_state['scanner_results'] = pd.concat(
                    [_old_without, _new_df], ignore_index=True
                ).sort_values('Barbell Score', ascending=False)
                # odśwież dane wykresów dla nowej klasy
                st.session_state.pop('scanner_data', None)
                st.toast(f"✅ Zaktualizowano wyniki dla: {asset_filter} ({len(_new_df)} aktywów)")
                st.rerun()
            else:
                st.error(f"❌ **Skanowanie zakończone, ale żadne z aktywów z klasy `{asset_filter}` nie spełniło minimalnych kryteriów analizy.** (Wymagane m.in. >50 dni notowań, minimum płynności).")

    # Apply Filters
    df_filtered = df_res.copy()
    if asset_filter != "Wszystkie":
        _uni = _CLASS_UNIVERSE.get(asset_filter, [])
        if _uni:
            df_filtered = df_filtered[df_filtered['Ticker'].isin(_uni)]
        else:
            # Fallback w razie braku zdefiniowanej listy
            if "Krypto" in asset_filter:
                df_filtered = df_filtered[df_filtered['Ticker'].str.endswith('-USD')]
            elif "Polska" in asset_filter:
                df_filtered = df_filtered[df_filtered['Ticker'].str.endswith('.WA')]
                
    if df_filtered.empty:
        st.warning(f"⚠️ **Brak wyników do wyświetlenia dla klasy: `{asset_filter}`**\n\nMożliwe przyczyny:\n1. **Nie skanowano jeszcze tej klasy.** (Kliknij widoczny wyżej przycisk *🔍 Skanuj*)\n2. **Żadne aktywo z tej klasy nie spełniło rygorów analizy EVT.** Model automatycznie odrzuca aktywa ze zbyt krótką historią notowań (<50 dni) lub brakiem płynności.")
        st.stop() # Zatrzymaj renderowanie pustych wykresów w dół strony 

    st.divider()
    st.subheader("🏆 Ranking Antykruchości (Barbell Score)")
    st.caption("Barbell Score = ważony Z-Score: EVT Prawy Ogón (+), Skewness (+), Omega (+), Momentum (+), Hurst (+), EVT Lewy Ogón (kara), Amihud (kara).")

    def highlight_watchlist(val):
        # Highlight tickers in watchlist
        if val in st.session_state.get('watchlist', []):
            return 'background-color: #3498db; color: white; font-weight: bold'
        return ''

    def highlight_barbell(val):
        if isinstance(val, (int, float)):
            if val > 0.5:   return 'color: #2ecc71; font-weight: bold'
            if val < -0.3:  return 'color: #e74c3c; font-weight: bold'
        return ''

    def highlight_hurst(val):
        if isinstance(val, (int, float)):
            if val > 0.55:  return 'color: #2ecc71'
            if val < 0.45:  return 'color: #3498db'
        return ''

    display_cols = [
        "Ticker", "Barbell Score", "EVT Shape (Tail)", "EVT Left Tail",
        "Omega", "Hurst", "Momentum_1Y",
        "Skewness", "Annual Return", "Volatility",
        "Sharpe", "Sortino", "Max Drawdown", "Kelly Safe (50%)"
    ]
    display_cols = [c for c in display_cols if c in df_filtered.columns]

    df_display = df_filtered[display_cols].copy()
    df_display.insert(0, "Wybierz", False)

    format_dict = {}
    for col_ in ["Annual Return", "Volatility", "Momentum_1Y", "Max Drawdown", "Kelly Safe (50%)"]:
        if col_ in df_display.columns: format_dict[col_] = "{:.1%}"
    for col_ in ["Barbell Score", "Skewness", "EVT Shape (Tail)", "EVT Left Tail",
                 "Omega", "Hurst", "Sharpe", "Sortino"]:
        if col_ in df_display.columns: format_dict[col_] = "{:.2f}"
    dynamic_height = (len(df_display) + 1) * 38 + 10

    try:
        styled = df_display.style.format(format_dict)
        if "Barbell Score" in df_display.columns:
            styled = styled.applymap(highlight_barbell, subset=["Barbell Score"])
        if "Hurst" in df_display.columns:
            styled = styled.applymap(highlight_hurst, subset=["Hurst"])
    except Exception as e:
        logger.warning(f"Błąd stylowania tabeli: {e}")
        styled = df_display

    edited_df_scan = st.data_editor(
        styled, use_container_width=True, height=dynamic_height,
        column_config={
            "Wybierz": st.column_config.CheckboxColumn("Wybierz", help="Zaznacz aby przenieść do Symulatora", default=False),
            "Hurst":           st.column_config.NumberColumn("Hurst H", help="> 0.55 🟢 Trend | 0.45-0.55 Losowy | < 0.45 🔵 MR"),
            "EVT Left Tail":   st.column_config.NumberColumn("Crash Risk", help="Niższy = bezpieczniejszy. > 0.5 = dyskwalifikacja z Risky Sleeve."),
            "Barbell Score":   st.column_config.NumberColumn("🦸 Barbell Score", help="Główny wskaźnik selekcji (AQR-style ważony Z-Score, 7 czynników)"),
            "Momentum_1Y":     st.column_config.NumberColumn("Momentum 12M", help="Zwrot z ostatnich 12M minus 1M (Jegadeesh & Titman 1993)"),
            "Omega":           st.column_config.NumberColumn("Omega Ratio", help="> 1.0 = więcej zysku niż straty. Nie zakłada normalności (Shadwick 2002)."),
        },
        disabled=display_cols,
    )

    selected_rows = edited_df_scan[edited_df_scan["Wybierz"]]

    if not selected_rows.empty:
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button(f"📉 Przenieś ({len(selected_rows)}) do Symulatora", type="primary", use_container_width=True):
                tickers_to_transfer = selected_rows["Ticker"].tolist()
                weight = 100.0 / len(tickers_to_transfer)
                transfer_data = [{"Ticker": t, "Waga (%)": weight} for t in tickers_to_transfer]
                st.session_state['transfer_data'] = pd.DataFrame(transfer_data)
                st.switch_page("pages/1_Symulator.py")
                
        with btn_col2:
            if st.button(f"⚡ Przenieś ({len(selected_rows)}) do Stress Testów", use_container_width=True):
                tickers_to_transfer = selected_rows["Ticker"].tolist()
                st.session_state["st_risky_transfer"] = ", ".join(tickers_to_transfer)
                st.switch_page("pages/3_Stress_Test.py")

    # --- Historia Skanów (Wykres Liniowy) ---
    if 'scan_history' in st.session_state and not st.session_state['scan_history'].empty:
        st.divider()
        st.subheader("📈 Historia Barbell Score (Ewolucja w czasie)")
        hist_df = st.session_state['scan_history'].copy()
        hist_df['ScanDate'] = pd.to_datetime(hist_df['ScanDate'])
        
        # Pokaż tylko z Watchlisty, jeśli są, by nie zamazać wykresu
        wl = st.session_state.get('watchlist', [])
        if wl:
            hist_plot = hist_df[hist_df['Ticker'].isin(wl)]
        else:
            top5_recent = df_res.head(5)['Ticker'].tolist()
            hist_plot = hist_df[hist_df['Ticker'].isin(top5_recent)]
            
        fig_hist_score = px.line(hist_plot, x='ScanDate', y='Barbell Score', color='Ticker', markers=True,
                                title="Ewolucja Barbell Score dla obserwowanych walorów")
        fig_hist_score.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_hist_score, use_container_width=True)

    # --- New Visualization: 3D Antifragile Scatter ---
    st.divider()
    st.subheader("🧊 Mapa Antykruchości 3D")
    st.caption("Szukamy aktywów w prawym górnym rogu (Wysoki Skew, Wysoka Kurtoza, Niskie Hill Alpha jako duży bąbel).")
    
    # Prepare Data for 3D Plot
    # X: Skewness, Y: Kurtosis, Z: Annual Return
    # Color: Score, Size: Inverse Hill Alpha (or just fixed if nan)
    
    plot_df = df_res.copy()
    plot_df['EVT Shape (Tail)'] = plot_df['EVT Shape (Tail)'].fillna(0.0) 
    # Create Size dimension: Higher EVT Shape = Bigger Bubble (Fatter tail)
    plot_df['Size'] = (plot_df['EVT Shape (Tail)'] * 50).clip(upper=30, lower=5)
    
    fig_3d_scan = px.scatter_3d(
        plot_df,
        x='Skewness',
        y='Kurtosis',
        z='Annual Return',
        color='Score',
        size='Size', # Dynamic size
        hover_name='Ticker',
        hover_data=['EVT Shape (Tail)', 'Kelly Safe (50%)'],
        color_continuous_scale='Viridis',
        title='Przestrzeń Wypukłości (Convexity Space)'
    )
    fig_3d_scan.update_layout(
         scene=dict(
            xaxis_title='Skośność (Skew)',
            yaxis_title='Kurtoza (Kurt)',
            zaxis_title='Zwrot Roczny'
        ),
        template="plotly_dark",
        height=600
    )
    st.plotly_chart(fig_3d_scan, use_container_width=True)
    
    display_chart_guide("Mapa Antykruchości 3D", """
    *   **Szukaj Baniek**: Szukamy aktywów w prawym górnym rogu (Wysoka Skośność, Wysoka Kurtoza).
    *   **Rozmiar Bańki**: Większa bańka = Bardzo "Gruby Ogon" Mierzony Teorią EVT (Wyższe EVT Shape). To są potencjalne "rakiety".
    """)
    
    st.markdown("""
    ### 📖 Legenda Metryk (Słownik)
    
    *   **Annual Return**: Średni roczny zwrot geometryczny.
    *   **Volatility (Zmienność)**: Zmienność roczna. W strategii sztangi traktujemy ją jako **zasób**.
    *   **Skewness (Skośność)**: Mierzy asymetrię. >0 to nasz cel (częste małe straty, rzadkie wielkie zyski).
    *   **Kurtosis (Kurtoza)**: Mierzy "grubość" ogonów. Im wyższa, tym więcej ekstremalnych zdarzeń.
    *   **EVT Shape (Tail)**: Kluczowa metryka Teorii EVT (Peaks Over Threshold). Im wyższa, tym większa szansa na wybitne zdarzenia (Black Swans).
    *   **Sharpe Ratio**: Wynik > 1.0 jest dobry. Mierzy zysk na jednostkę całkowitego ryzyka (zmienności).
    *   **Sortino Ratio**: Lepsza wersja Sharpe'a. Mierzy zysk na jednostkę "złej zmienności" (tylko spadki).
    *   **Max Drawdown**: Maksymalne obsunięcie kapitału. Mówi o tym, jak bardzo zaboli w najgorszym momencie.
    """)
    
    # Best Asset Charts — używamy df_filtered by wykresy odpowiadały aktualnej klasie
    _df_for_chart = df_filtered if not df_filtered.empty else df_res
    best_asset = _df_for_chart.iloc[0]
    class_label = f" [{asset_filter}]" if asset_filter != "Wszystkie" else ""
    st.subheader(f"💎 Najlepsze Aktywo{class_label}: {best_asset['Ticker']}")
    
    col_chart1, col_chart2 = st.columns(2)
    
    # Returns Histogram
    asset_data = data[best_asset['Ticker']].pct_change().dropna()
    fig_hist = px.histogram(asset_data, nbins=100, title=f"Rozkład Zwrotów {best_asset['Ticker']}")
    col_chart1.plotly_chart(fig_hist, use_container_width=True)
    
    # Cumulative Return (Log scale)
    cum_ret = (1 + asset_data).cumprod()
    fig_line = px.line(cum_ret, log_y=True, title=f"Wzrost Kapitału (Skala Log) {best_asset['Ticker']}")
    col_chart2.plotly_chart(fig_line, use_container_width=True)
    
    display_chart_guide("Wykresy Najlepszego Aktywa", """
    *   **Histogram**: Szukamy prawego "długiego ogona" (wiele słupków po prawej stronie zera).
    *   **Skala Logarytmiczna**: Linia prosta oznacza stabilne tempo wzrostu procentowego ($CAGR$).
    """)

    # 2a. Correlation Heatmap (Scanner)
    if len(data.columns) > 1:
        st.divider()
        st.subheader("🔥 Mapa Korelacji (Skaner)")
        st.caption("Sprawdź, czy wybrane aktywa są ze sobą powiązane.")
        
        # Calculate correlation matrix
        corr_matrix_scan = data.pct_change().corr()
        
        fig_corr_scan = px.imshow(
            corr_matrix_scan, 
            text_auto=".2f", 
            color_continuous_scale='RdBu_r', 
            zmin=-1, zmax=1,
            title="Macierz Korelacji"
        )
        fig_corr_scan.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig_corr_scan, use_container_width=True)
        
        display_chart_guide("Mapa Korelacji (Skaner)", """
        *   **Cel**: Budowa portfela wymaga niskiej korelacji. Jeśli wszystko jest czerwone (kor > 0.8), dywersyfikacja nie działa.
        *   **Szukaj Błękitu**: Wartości bliskie 0 lub ujemne to idealni kandydaci do pary w strategii Barbell.
        """)

    # 3. Log-Log Tail Plot (Power Law Visualizer)
    st.markdown("**Analiza Ogonów (Log-Log Plot)**")
    st.caption("Jeśli linia jest prosta (opada liniowo na skali log-log), mamy do czynienia z Rozkładem Potęgowym (Power Law) i Grubymi Ogonami. Krzywa opadająca szybko (jak parabola) to rozkład Normalny (Gaussa).")
    
    # Calculate Tail Survival Function
    # We look at right tail (positive returns)
    pos_rets = asset_data[asset_data > 0].sort_values(ascending=False)
    if len(pos_rets) > 10:
        rank = np.arange(1, len(pos_rets) + 1)
        prob = rank / len(pos_rets)
        
        fig_loglog = go.Figure()
        fig_loglog.add_trace(go.Scattergl(
            x=pos_rets,
            y=prob,
            mode='markers',
            name='Empiryczne Dane',
            marker=dict(color='#00ff88', size=5)
        ))
        fig_loglog.update_layout(
            title=f"Ogon Prawy: {best_asset['Ticker']} (Log-Log)",
            xaxis_title="Zwrot Dzienny (Log)",
            yaxis_title="P(X > x) (Log)",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_loglog, use_container_width=True)
        
        display_chart_guide("Log-Log Tail Plot", """
        *   **Test Potęgowy**: To najważniejszy test "Antykruchości".
        *   **Linia Prosta**: Oznacza, że ryzyko nie maleje wykładniczo. Krachy i Rakiety są bardziej prawdopodobne niż sądzisz (Mandlebrot/Taleb).
        *   **Parabola w dół**: Oznacza "Bezpieczny" rozkład normalny (mało niespodzianek).
        """)
    else:
        st.info("Za mało danych do wygenerowania wykresu Log-Log.")

st.markdown("---")
display_scanner_methodology()

# ─────────────────────────────────────────────────────────────────
# 🆕 HIERARCHICAL DENDROGRAM (Zamiast płaskiego MST)
# ─────────────────────────────────────────────────────────────────
if 'bcs_returns' in st.session_state:
    st.divider()
    st.subheader("🌳 Dendrogram Klastrowy (Hierarchical Risk Parity) 🆕")
    st.caption("Maszyny uczące budują zagnieżdżoną strukturę ryzyka. Szukaj aktywów, które odłączają się najwcześniej na dole (najdłuższe pionowe gałęzie) — to prawdziwe, nieskorelowane dywersyfikatory wg Teorematu Lopeza de Prado.")
    mst_fig = compute_hierarchical_dendrogram(st.session_state['bcs_returns'])
    
    if mst_fig:
        st.plotly_chart(mst_fig, use_container_width=True)
        st.caption("🟢 Odkryj prawdziwą architekturę rynku: Drzewo pokazuje które rynki zachowują się jak jedno aktywo, a które faktycznie różnią się od reszty koszyka.")



# ─────────────────────────────────────────────────────────────────
# ⚡ STRESS TEST TAB
# ─────────────────────────────────────────────────────────────────

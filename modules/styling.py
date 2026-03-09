"""
styling.py — Premium Dark Theme dla Barbell Strategy Dashboard v9.5

Zawiera:
  - Glassmorphism karty
  - Gradient radial background (#05060d → #0d0f1a)
  - Neon glow na kluczowych wskaźnikach
  - Animated gauge needle pulse
  - Pulsujący alert badge (czerwony blink)
  - CSS animacje (fadeIn, pulse, neonFlicker)
  - Live ticker bar base style
  - Expander math explainer style
"""


def inject_accordion_js():
    """
    Wstrzykuje JavaScript sterujacy accordion sidebar (st.navigation).
    Uzywa MutationObserver + click capture do wymuszenia:
    - zamknięcia wszystkich sekcji menu przy starcie
    - otwarcia tylko sekcji z aktywnym linkiem (aria-current)
    - prawdziwego akordeonu przy kliknięciach (jedno rozwinięte naraz)
    """
    import streamlit.components.v1 as components

    js_code = """
    <script>
    (function() {
        var doc = window.parent.document;
        var _observer = null;
        var _clickHandler = null;

        function collapseAll(nav) {
            nav.querySelectorAll('details').forEach(function(d) {
                d.removeAttribute('open');
            });
        }

        function openActive(nav) {
            var activeLink = nav.querySelector('a[aria-current="page"]');
            if (activeLink) {
                var det = activeLink.closest('details');
                if (det) { det.setAttribute('open', ''); return; }
            }
            // Fallback: strona glowna — otwieramy sekcje Dashboard (index 1)
            var allDet = nav.querySelectorAll('details');
            if (allDet.length > 1) allDet[1].setAttribute('open', '');
            else if (allDet.length === 1) allDet[0].setAttribute('open', '');
        }

        function applyAccordion() {
            var nav = doc.querySelector('[data-testid="stSidebarNav"]');
            if (!nav) return false;
            if (nav.querySelectorAll('details').length === 0) return false;
            collapseAll(nav);
            openActive(nav);
            return true;
        }

        function attachAccordionClick(nav) {
            if (_clickHandler) nav.removeEventListener('click', _clickHandler, true);
            _clickHandler = function(e) {
                var summary = e.target.closest('summary');
                if (!summary) return;
                var clickedDet = summary.closest('details');
                if (!clickedDet) return;
                // Zapobiegamy domyslnemu dzialaniu przegladarki
                e.preventDefault();
                e.stopPropagation();
                var wasOpen = clickedDet.hasAttribute('open');
                // Zamknij wszystkie sekcje
                nav.querySelectorAll('details').forEach(function(d) {
                    d.removeAttribute('open');
                });
                // Jesli nie bylo otwarte — otworz klikniete (toggle)
                if (!wasOpen) {
                    clickedDet.setAttribute('open', '');
                }
            };
            nav.addEventListener('click', _clickHandler, true);
        }

        function attachObserver(nav) {
            if (_observer) { _observer.disconnect(); }
            _observer = new MutationObserver(function(mutations) {
                var needsUpdate = false;
                for (var i = 0; i < mutations.length; i++) {
                    var m = mutations[i];
                    if (m.type === 'attributes' && m.attributeName === 'aria-current') {
                        needsUpdate = true; break;
                    }
                    if (m.type === 'childList' && m.addedNodes.length > 0) {
                        needsUpdate = true; break;
                    }
                }
                if (needsUpdate) { setTimeout(applyAccordion, 100); }
            });
            _observer.observe(nav, {
                subtree: true, childList: true,
                attributes: true, attributeFilter: ['aria-current']
            });
        }

        // Glowna inicjalizacja z retry co 150ms (max 6 sekund)
        var attempts = 0;
        var initTimer = setInterval(function() {
            attempts++;
            var nav = doc.querySelector('[data-testid="stSidebarNav"]');
            if (nav && nav.querySelectorAll('details').length > 0) {
                clearInterval(initTimer);
                applyAccordion();
                attachAccordionClick(nav);
                attachObserver(nav);
            }
            if (attempts > 40) clearInterval(initTimer);
        }, 150);

    })();
    </script>
    """
    components.html(js_code, height=0, width=0)

def apply_styling() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

    /* ═══════════════════════════════════════════════════════════════
       ROOT VARIABLES — Premium Cyberpunk Palette
    ═══════════════════════════════════════════════════════════════ */
    :root {
        --bg-deep:     #05060d;
        --bg-mid:      #0d0f1a;
        --bg-surface:  #0f111a;
        --bg-card:     rgba(15, 17, 28, 0.85);
        --border:      rgba(255,255,255,0.06);
        --border-glow: rgba(0, 230, 118, 0.25);
        --green:       #00e676;
        --green-dim:   rgba(0,230,118,0.08);
        --cyan:        #00ccff;
        --yellow:      #ffea00;
        --red:         #ff1744;
        --purple:      #a855f7;
        --text:        #e2e4f0;
        --text-dim:    #6b7280;
        --font:        'Inter', sans-serif;
        --mono:        'JetBrains Mono', monospace;
    }

    /* ═══════════════════════════════════════════════════════════════
       GRADIENT BACKGROUND (radial + linear depth)
    ═══════════════════════════════════════════════════════════════ */
    .stApp {
        background:
            radial-gradient(ellipse at 20% 10%, rgba(0,230,118,0.04) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(0,204,255,0.04) 0%, transparent 50%),
            linear-gradient(160deg, #05060d 0%, #0a0b14 40%, #0d0f1a 100%) !important;
        color: var(--text) !important;
        font-family: var(--font) !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       GLASSMORPHISM CARDS  (wszystkie st.container, st.expander)
    ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stVerticalBlockBorderWrapper"],
    div.element-container > div[data-testid="stExpander"] {
        background: var(--bg-card) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        box-shadow: 0 4px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.04) !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(0,230,118,0.18) !important;
        box-shadow: 0 6px 40px rgba(0,0,0,0.45), 0 0 20px rgba(0,230,118,0.06) !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       LAYOUT — Odpowiedni padding aby treść nie była za wysoko
    ═══════════════════════════════════════════════════════════════ */
    div.block-container {
        padding-top: 2.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    div[data-testid="stVerticalBlock"] > div { gap: 0.2rem; }
    .stPlotlyChart { margin-bottom: -8px; margin-top: -14px; }
    h4 { margin-bottom: 4px !important; margin-top: 4px !important; }

    /* ═══════════════════════════════════════════════════════════════
       HEADINGS — gradient text
    ═══════════════════════════════════════════════════════════════ */
    h1 {
        background: linear-gradient(90deg, var(--green) 0%, var(--cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.6rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.5px;
        padding-bottom: 0.3rem;
    }
    h2, h3 {
        color: var(--text) !important;
        font-weight: 700 !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       SIDEBAR — Premium branded navigation
    ═══════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08090f 0%, #0b0d18 50%, #0d0f1a 100%) !important;
        border-right: 1px solid rgba(0,230,118,0.10) !important;
    }
    /* Paddin top dla opcji sidebara */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem !important;
    }
    /* Header for options */
    .sidebar-options-header {
        font-size: 10px !important;
        font-weight: 700 !important;
        color: var(--green) !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        margin-bottom: 8px !important;
        opacity: 0.8;
    }
    /* Navigation group headers — bold, uppercase, spaced */
    [data-testid="stSidebarNavLink"] {
        border-radius: 8px !important;
        margin: 2px 8px !important;
        padding: 8px 14px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        color: #9ca3af !important;
        transition: background 0.2s ease, color 0.2s ease !important;
    }
    [data-testid="stSidebarNavLink"]:hover {
        background: rgba(0,230,118,0.08) !important;
        color: #e2e4f0 !important;
    }
    [data-testid="stSidebarNavLink"][aria-selected="true"] {
        background: rgba(0,230,118,0.12) !important;
        color: #00e676 !important;
        border-left: 3px solid #00e676 !important;
        font-weight: 600 !important;
    }
    /* Section labels in navigation */
    [data-testid="stSidebarNavSeparator"] {
        color: rgba(255,255,255,0.3) !important;
        font-size: 10px !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        padding: 12px 14px 4px 14px !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       NEON GLOW — klasy do ręcznego stosowania
       Użycie: <span class="neon-green">wartość</span>
    ═══════════════════════════════════════════════════════════════ */
    .neon-green {
        color: var(--green);
        text-shadow: 0 0 8px rgba(0,230,118,0.8), 0 0 20px rgba(0,230,118,0.4);
    }
    .neon-red {
        color: var(--red);
        text-shadow: 0 0 8px rgba(255,23,68,0.8), 0 0 20px rgba(255,23,68,0.4);
    }
    .neon-cyan {
        color: var(--cyan);
        text-shadow: 0 0 8px rgba(0,204,255,0.7), 0 0 18px rgba(0,204,255,0.3);
    }
    .neon-yellow {
        color: var(--yellow);
        text-shadow: 0 0 8px rgba(255,234,0,0.7), 0 0 18px rgba(255,234,0,0.3);
    }

    /* ═══════════════════════════════════════════════════════════════
       ANIMATED GAUGE WRAPPER — fade-in przy renderowaniu
    ═══════════════════════════════════════════════════════════════ */
    @keyframes gaugeAppear {
        from { opacity: 0; transform: scale(0.97); }
        to   { opacity: 1; transform: scale(1.00); }
    }
    .js-plotly-plot {
        animation: gaugeAppear 0.5s ease-out;
    }

    /* ═══════════════════════════════════════════════════════════════
       PULSE ALERT BADGE — czerwone blink gdy alarm
    ═══════════════════════════════════════════════════════════════ */
    @keyframes pulseRed {
        0%   { box-shadow: 0 0 0 0 rgba(255,23,68,0.7); }
        50%  { box-shadow: 0 0 0 10px rgba(255,23,68,0.0); }
        100% { box-shadow: 0 0 0 0 rgba(255,23,68,0.0); }
    }
    @keyframes blinkBg {
        0%, 100% { background: rgba(255,23,68,0.15); }
        50%       { background: rgba(255,23,68,0.32); }
    }
    .alert-badge-red {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 50px;
        background: rgba(255,23,68,0.15);
        border: 1px solid rgba(255,23,68,0.5);
        color: #ff1744;
        font-weight: 700;
        font-size: 12px;
        animation: pulseRed 1.5s infinite, blinkBg 1.5s infinite;
        letter-spacing: 1px;
    }
    @keyframes pulseGreen {
        0%   { box-shadow: 0 0 0 0 rgba(0,230,118,0.6); }
        50%  { box-shadow: 0 0 0 8px rgba(0,230,118,0.0); }
        100% { box-shadow: 0 0 0 0 rgba(0,230,118,0.0); }
    }
    .alert-badge-green {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 50px;
        background: rgba(0,230,118,0.12);
        border: 1px solid rgba(0,230,118,0.4);
        color: #00e676;
        font-weight: 700;
        font-size: 12px;
        animation: pulseGreen 2.5s infinite;
        letter-spacing: 1px;
    }
    .alert-badge-yellow {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 50px;
        background: rgba(255,234,0,0.10);
        border: 1px solid rgba(255,234,0,0.35);
        color: #ffea00;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 1px;
    }

    /* ═══════════════════════════════════════════════════════════════
       NEON METRIC CARDS
    ═══════════════════════════════════════════════════════════════ */
    .metric-card {
        background: var(--bg-card);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px 12px;
        text-align: center;
        transition: border-color 0.25s ease, box-shadow 0.25s ease;
    }
    .metric-card:hover {
        border-color: rgba(0,230,118,0.25);
        box-shadow: 0 0 16px rgba(0,230,118,0.08);
    }
    .metric-value {
        font-family: var(--mono);
        font-size: 22px;
        font-weight: 700;
        color: white;
    }
    .metric-label {
        font-size: 11px;
        color: var(--text-dim);
        margin-bottom: 4px;
        letter-spacing: 0.5px;
    }

    /* ═══════════════════════════════════════════════════════════════
       GLASS INFO CARD (business cycle / regime cards)
    ═══════════════════════════════════════════════════════════════ */
    .glass-card {
        background: linear-gradient(135deg, rgba(15,17,26,0.9) 0%, rgba(20,22,35,0.85) 100%);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 18px 14px;
        text-align: center;
        transition: border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(255,255,255,0.12);
    }

    /* ═══════════════════════════════════════════════════════════════
       EXPANDER — Math Explainer ("Skąd pochodzi ta liczba?")
    ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stExpander"] {
        background: rgba(10,11,20,0.6) !important;
        border: 1px solid rgba(0,204,255,0.12) !important;
        border-radius: 10px !important;
        margin-top: 4px !important;
    }
    div[data-testid="stExpander"] summary {
        color: var(--cyan) !important;
        font-size: 12px !important;
        font-family: var(--mono) !important;
        letter-spacing: 0.5px;
    }
    div[data-testid="stExpander"] summary:hover {
        color: white !important;
    }
    .math-formula {
        font-family: var(--mono);
        font-size: 13px;
        color: var(--cyan);
        background: rgba(0,204,255,0.06);
        border-left: 3px solid var(--cyan);
        padding: 8px 14px;
        border-radius: 0 6px 6px 0;
        margin: 8px 0;
    }
    .math-note {
        font-size: 12px;
        color: #7c8096;
        line-height: 1.5;
        margin: 6px 0;
    }
    .math-source {
        font-size: 10px;
        color: #4a4e6a;
        font-style: italic;
        margin-top: 8px;
        border-top: 1px solid rgba(255,255,255,0.05);
        padding-top: 6px;
    }

    /* ═══════════════════════════════════════════════════════════════
       LIVE TICKER BAR
    ═══════════════════════════════════════════════════════════════ */
    .ticker-bar {
        background: linear-gradient(90deg, rgba(5,6,13,0.95), rgba(13,15,26,0.95));
        border-bottom: 1px solid rgba(0,230,118,0.12);
        padding: 4px 0;
        overflow: hidden;
        white-space: nowrap;
        font-family: var(--mono);
        font-size: 12px;
    }
    @keyframes tickerScroll {
        0%   { transform: translateX(0); }
        100% { transform: translateX(-50%); }
    }
    .ticker-inner {
        display: inline-block;
        animation: tickerScroll 30s linear infinite;
    }
    .ticker-item {
        display: inline-block;
        margin: 0 24px;
        color: #9ca3af;
    }
    .ticker-item .ticker-name { color: #e2e4f0; font-weight: 600; }
    .ticker-item .ticker-pos { color: #00e676; }
    .ticker-item .ticker-neg { color: #ff1744; }

    /* ═══════════════════════════════════════════════════════════════
       BUTTONS — premium gradient
    ═══════════════════════════════════════════════════════════════ */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0,230,118,0.15) 0%, rgba(0,204,255,0.10) 100%) !important;
        color: var(--green) !important;
        border: 1px solid rgba(0,230,118,0.3) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: var(--font) !important;
        letter-spacing: 0.5px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0,230,118,0.25) 0%, rgba(0,204,255,0.18) 100%) !important;
        border-color: rgba(0,230,118,0.6) !important;
        box-shadow: 0 0 20px rgba(0,230,118,0.2) !important;
        transform: translateY(-1px) !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       SLIDERS
    ═══════════════════════════════════════════════════════════════ */
    .stSlider [data-baseweb="slider"] div[data-testid="stTickBar"] {
        color: var(--green) !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       METRICS (st.metric)
    ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stMetricValue"] {
        color: var(--green) !important;
        font-family: var(--mono) !important;
        font-weight: 700 !important;
        text-shadow: 0 0 12px rgba(0,230,118,0.4);
    }
    div[data-testid="stMetricDelta"] svg { display: none; }
    div[data-testid="metric-container"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 14px;
        backdrop-filter: blur(8px);
    }

    /* ═══════════════════════════════════════════════════════════════
       DIVIDERS
    ═══════════════════════════════════════════════════════════════ */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,230,118,0.2), transparent) !important;
        margin: 12px 0 !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       SCROLLBAR custom
    ═══════════════════════════════════════════════════════════════ */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0a0b12; }
    ::-webkit-scrollbar-thumb { background: rgba(0,230,118,0.25); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0,230,118,0.45); }

    /* ═══════════════════════════════════════════════════════════════
       FADEUP ANIMATION — elementy wchodzą od dołu
    ═══════════════════════════════════════════════════════════════ */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-up { animation: fadeUp 0.4s ease-out; }

    /* ═══════════════════════════════════════════════════════════════
       TABLE styles
    ═══════════════════════════════════════════════════════════════ */
    table {
        border-collapse: collapse;
        width: 100%;
        font-family: var(--font);
        font-size: 13px;
    }
    thead tr {
        background: rgba(0,230,118,0.07);
        border-bottom: 1px solid rgba(0,230,118,0.15);
    }
    thead th {
        color: var(--green);
        font-weight: 600;
        padding: 8px 12px;
        letter-spacing: 0.5px;
        text-align: left;
    }
    tbody tr:nth-child(even) { background: rgba(255,255,255,0.02); }
    tbody td {
        padding: 7px 12px;
        color: var(--text);
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    tbody tr:hover { background: rgba(0,230,118,0.04); }

    /* ═══════════════════════════════════════════════════════════════
       SIDEBAR ACCORDION — smooth CSS transition for details/summary
    ═══════════════════════════════════════════════════════════════ */
    [data-testid="stSidebarNav"] details {
        transition: all 0.2s ease;
    }
    [data-testid="stSidebarNav"] details summary {
        cursor: pointer;
        user-select: none;
    }

    </style>

    """


def alert_badge_html(score: float) -> str:
    """Zwraca HTML pulsującego badge z klasą CSS zależną od score."""
    if score > 65:
        return f"<span class='alert-badge-red'>⚠ ALARM · SCORE {score:.0f}</span>"
    elif score > 35:
        return f"<span class='alert-badge-yellow'>⚡ NEUTRAL · SCORE {score:.0f}</span>"
    else:
        return f"<span class='alert-badge-green'>✓ RISK-ON · SCORE {score:.0f}</span>"


def math_explainer(
    title: str,
    formula: str,
    explanation: str,
    source: str = "",
) -> str:
    """
    Zwraca HTML bloku matematycznego wyjaśnienia metryki.
    Użycie: st.markdown(math_explainer(...), unsafe_allow_html=True)
    """
    src_html = f"<div class='math-source'>📚 {source}</div>" if source else ""
    return f"""
    <div class='fade-up'>
        <div class='math-formula'>{formula}</div>
        <div class='math-note'>{explanation}</div>
        {src_html}
    </div>
    """


def ticker_bar_html(items: list[dict]) -> str:
    """
    Zwraca HTML live ticker bar.

    Parameters
    ----------
    items : lista {'name': str, 'value': float, 'change': float, 'suffix': str}
    """
    def item_html(d: dict) -> str:
        chg = d.get("change", 0.0)
        chg_str = f"{chg:+.2f}%"
        chg_cls = "ticker-pos" if chg >= 0 else "ticker-neg"
        chg_arrow = "▲" if chg >= 0 else "▼"
        val_str = f"{d.get('suffix','')}{d.get('value',0):.2f}"
        return (
            f"<span class='ticker-item'>"
            f"<span class='ticker-name'>{d['name']}</span> "
            f"{val_str} "
            f"<span class='{chg_cls}'>{chg_arrow} {chg_str}</span>"
            f"</span>"
        )

    # Duplikujemy dla seamless loop
    content = "".join(item_html(i) for i in items) * 2
    return f"""
    <div class='ticker-bar'>
        <div class='ticker-inner'>{content}</div>
    </div>
    """

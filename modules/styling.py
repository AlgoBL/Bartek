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


import streamlit as st

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
        
        function enforceAccordion(nav) {
            var activeLink = nav.querySelector('a[aria-current="page"]');
            var activeDetails = activeLink ? activeLink.closest('details') : null;
            
            nav.querySelectorAll('details').forEach(function(d) {
                // Listeners for manual clicks
                if (!d.dataset.accordionAttached) {
                    d.addEventListener('toggle', function(e) {
                        if (e.target.open) {
                            nav.querySelectorAll('details').forEach(function(other) {
                                if (other !== e.target && other.open) {
                                    other.removeAttribute('open');
                                }
                            });
                        }
                    }, true);
                    d.dataset.accordionAttached = "true";
                }
            });
            
            // On navigation or load, force only the active section to be open
            if (activeDetails && !activeDetails.open) {
                activeDetails.setAttribute('open', '');
            }
            
            if (activeDetails) {
                nav.querySelectorAll('details').forEach(function(d) {
                    if (d !== activeDetails && d.open) {
                        d.removeAttribute('open');
                    }
                });
            } else {
                // Fallback if no active link: open the first details if multiple, or close all but first
                var allDet = nav.querySelectorAll('details');
                if (allDet.length > 0) {
                    var anyOpen = false;
                    for(var i=0; i<allDet.length; i++) {
                        if (allDet[i].open) anyOpen = true;
                    }
                    if (!anyOpen && allDet.length > 1) {
                        allDet[1].setAttribute('open', '');
                    } else if (!anyOpen && allDet.length === 1) {
                        allDet[0].setAttribute('open', '');
                    }
                }
            }
        }

        // The interval ensures even when Streamlit completely unmounts and remounts the navigation, 
        // the accordion logic is instantly re-applied and listeners attached.
        setInterval(function() {
            var nav = doc.querySelector('[data-testid="stSidebarNav"]');
            if (nav) {
                enforceAccordion(nav);
            }
        }, 300);

    })();
    </script>
    """
    components.html(js_code, height=0, width=0)

def inject_aos_js():
    """
    Wstrzykuje bibliotekę AOS (Animate On Scroll)
    Pozwala m.in. na data-aos="fade-up" wewnątrz wstrzykiwanych HTML.
    """
    import streamlit.components.v1 as components
    js_code = """
    <script>
    (function() {
        var doc = window.parent.document;
        if (doc.getElementById("aos-css-injected")) return;
        
        var aosLink = doc.createElement("link");
        aosLink.id = "aos-css-injected";
        aosLink.rel = "stylesheet";
        aosLink.href = "https://unpkg.com/aos@2.3.1/dist/aos.css";
        doc.head.appendChild(aosLink);
        
        var aosScript = doc.createElement("script");
        aosScript.src = "https://unpkg.com/aos@2.3.1/dist/aos.js";
        aosScript.onload = function() {
            var initScript = doc.createElement("script");
            initScript.innerHTML = "AOS.init({duration: 800, once: true});";
            doc.body.appendChild(initScript);
        };
        doc.head.appendChild(aosScript);
    })();
    </script>
    """
    components.html(js_code, height=0, width=0)


def inject_command_palette_js():
    import streamlit.components.v1 as components
    js_code = """
    <script>
    (function() {
        var doc = window.parent.document;
        if (doc.getElementById("cmd-palette-overlay")) return;

        var overlay = doc.createElement("div");
        overlay.id = "cmd-palette-overlay";
        Object.assign(overlay.style, {
            position: "fixed", top: "0", left: "0", width: "100vw", height: "100vh",
            backgroundColor: "rgba(5, 6, 13, 0.7)", backdropFilter: "blur(5px)",
            zIndex: "999999", display: "none", alignItems: "flex-start",
            justifyContent: "center", paddingTop: "15vh"
        });

        var modal = doc.createElement("div");
        Object.assign(modal.style, {
            width: "600px", maxWidth: "90vw", backgroundColor: "#0f111a",
            border: "1px solid rgba(0, 230, 118, 0.4)", borderRadius: "12px",
            boxShadow: "0 10px 40px rgba(0, 0, 0, 0.5), 0 0 20px rgba(0, 230, 118, 0.15)",
            overflow: "hidden", display: "flex", flexDirection: "column"
        });

        var inputWrapper = doc.createElement("div");
        Object.assign(inputWrapper.style, {
            display: "flex", alignItems: "center", borderBottom: "1px solid rgba(255, 255, 255, 0.05)",
            padding: "0 20px"
        });
        
        var searchIcon = doc.createElement("span");
        searchIcon.innerHTML = "⚡";
        Object.assign(searchIcon.style, { fontSize: "20px", marginRight: "10px", opacity: "0.8" });

        var input = doc.createElement("input");
        input.type = "text";
        input.placeholder = "Wpisz nazwę modułu... (np. Skaner, Symulator)";
        Object.assign(input.style, {
            width: "100%", padding: "20px 0", fontSize: "18px", backgroundColor: "transparent",
            border: "none", color: "#e2e4f0", outline: "none"
        });
        
        inputWrapper.appendChild(searchIcon);
        inputWrapper.appendChild(input);

        var listWrapper = doc.createElement("div");
        Object.assign(listWrapper.style, { maxHeight: "400px", overflowY: "auto", paddingBottom: "10px" });

        modal.appendChild(inputWrapper);
        modal.appendChild(listWrapper);
        overlay.appendChild(modal);
        doc.body.appendChild(overlay);

        var links = [];
        var activeIndex = 0;
        var flatList = []; // Flattened list for keyboard navigation
        
        // ── MRU (Most Recently Used) Logic ──
        function getMRU() {
            try {
                return JSON.parse(localStorage.getItem('barbell_mru_v1')) || [];
            } catch (e) { return []; }
        }
        function saveMRU(text) {
            var mru = getMRU();
            mru = mru.filter(function(item) { return item !== text; });
            mru.unshift(text);
            if (mru.length > 5) mru = mru.slice(0, 5); // Keep up to 5
            localStorage.setItem('barbell_mru_v1', JSON.stringify(mru));
        }

        // ── Group Categories ──
        function getCategory(text) {
            var t = text.toLowerCase();
            if (t.includes('skaner') || t.includes('factor') || t.includes('day trading') || t.includes('cykliczność')) return '📊 Analiza Rynkowa';
            if (t.includes('symulator') || t.includes('stress test') || t.includes('health') || t.includes('emerytura')) return '🛡️ Portfel i Ochrona';
            if (t.includes('life os')) return '🧬 Osobiste';
            if (t.includes('ustawienia') || t.includes('dashboard')) return '⚙️ System';
            return '📁 Inne';
        }

        function getNavLinks() {
            var items = doc.querySelectorAll('[data-testid="stSidebarNavLink"]');
            var result = [];
            items.forEach(function(item) {
                var rawText = item.textContent || item.innerText;
                result.push({ element: item, text: rawText.trim() });
            });
            return result;
        }

        function renderList(query) {
            listWrapper.innerHTML = "";
            var q = query.trim().toLowerCase();
            var filtered = links.filter(function(l) { return l.text.toLowerCase().includes(q); });
            
            flatList = []; // reset
            var isSearch = q.length > 0;
            
            var groups = {};
            var mru = getMRU();
            
            if (isSearch) {
                // If searching, group by category
                filtered.forEach(function(l) {
                    var cat = getCategory(l.text);
                    if (!groups[cat]) groups[cat] = [];
                    groups[cat].push(l);
                });
            } else {
                // If empty search, show MRU and default grouping
                if (mru.length > 0) {
                    groups['🕒 Ostatnio używane'] = [];
                    mru.forEach(function(mText) {
                        var found = links.find(function(l) { return l.text === mText; });
                        if (found) groups['🕒 Ostatnio używane'].push(found);
                    });
                }
                groups['⭐ Wszystkie Moduły'] = links; // show all
            }
            
            var groupNames = Object.keys(groups);
            if (groupNames.length === 0 || (groupNames.length === 1 && groups[groupNames[0]].length === 0)) {
                listWrapper.innerHTML = "<div style='padding:15px 20px;color:#666;font-family:sans-serif;'>Brak modułu pasującego do wyszukiwania...</div>";
                return;
            }
            
            groupNames.forEach(function(groupName) {
                var items = groups[groupName];
                if (items.length === 0) return;
                
                var header = doc.createElement("div");
                header.textContent = groupName;
                Object.assign(header.style, {
                    padding: "10px 20px 5px", fontSize: "12px", color: "#888",
                    textTransform: "uppercase", letterSpacing: "1px", fontWeight: "bold"
                });
                listWrapper.appendChild(header);
                
                items.forEach(function(l) {
                    // Avoid duplicating MRU items in the "All" list if empty search
                    if (!isSearch && groupName === '⭐ Wszystkie Moduły' && mru.includes(l.text)) return;
                    
                    var indexInFlat = flatList.length;
                    flatList.push(l);
                    
                    var item = doc.createElement("div");
                    item.textContent = l.text;
                    
                    var isActive = (indexInFlat === activeIndex);
                    
                    Object.assign(item.style, {
                        padding: "12px 20px", cursor: "pointer", fontFamily: "sans-serif",
                        color: isActive ? "#00e676" : "#e2e4f0",
                        backgroundColor: isActive ? "rgba(0, 230, 118, 0.1)" : "transparent",
                        borderLeft: isActive ? "3px solid #00e676" : "3px solid transparent",
                        transition: "all 0.1s ease",
                        margin: "2px 8px", borderRadius: "4px"
                    });

                    item.onmouseenter = function() { activeIndex = indexInFlat; renderList(query); };
                    item.onclick = function() { saveMRU(l.text); closeModal(); l.element.click(); };
                    listWrapper.appendChild(item);
                });
            });
            
            // Adjust bounds
            if (flatList.length > 0) {
                if (activeIndex < 0) activeIndex = 0;
                if (activeIndex >= flatList.length) activeIndex = flatList.length - 1;
            }
            
            // Auto scroll to active element
            var activeEl = listWrapper.children[activeIndex + groupNames.length]; // rough approx for scroll
            if (activeEl && activeEl.scrollIntoView) {
                activeEl.scrollIntoView({ block: "nearest" });
            }
        }

        function openModal() {
            links = getNavLinks();
            overlay.style.display = "flex";
            input.value = "";
            activeIndex = 0;
            renderList("");
            setTimeout(function() { input.focus(); }, 100);
        }

        function closeModal() {
            overlay.style.display = "none";
            input.blur();
        }

        doc.addEventListener("keydown", function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === "k") {
                e.preventDefault();
                overlay.style.display === "none" ? openModal() : closeModal();
            } else if (e.key === "Escape" && overlay.style.display !== "none") {
                e.preventDefault();
                closeModal();
            }
        });

        input.addEventListener("input", function(e) {
            activeIndex = 0;
            renderList(e.target.value);
        });

        input.addEventListener("keydown", function(e) {
            if (e.key === "ArrowDown") {
                e.preventDefault();
                if (activeIndex < flatList.length - 1) activeIndex++;
                renderList(input.value);
            } else if (e.key === "ArrowUp") {
                e.preventDefault();
                if (activeIndex > 0) activeIndex--;
                renderList(input.value);
            } else if (e.key === "Enter") {
                e.preventDefault();
                if (flatList.length > 0) {
                    saveMRU(flatList[activeIndex].text);
                    closeModal();
                    flatList[activeIndex].element.click();
                }
            }
        });

        overlay.addEventListener("click", function(e) {
            if (e.target === overlay) closeModal();
        });
    })();
    </script>
    """
    components.html(js_code, height=0, width=0)


def inject_keyboard_shortcuts_js():
    """Wstrzykuje JavaScript skrótów klawiszowych oraz overlay '?'."""
    import streamlit.components.v1 as components
    js_code = """
    <script>
    (function() {
        var doc = window.parent.document;
        if (doc.getElementById('kb-shortcuts-overlay')) return;

        var overlay = doc.createElement('div');
        overlay.id = 'kb-shortcuts-overlay';
        Object.assign(overlay.style, {
            position: 'fixed', top: '0', left: '0',
            width: '100vw', height: '100vh',
            backgroundColor: 'rgba(5,6,13,0.75)',
            backdropFilter: 'blur(6px)',
            zIndex: '999998',
            display: 'none',
            alignItems: 'center',
            justifyContent: 'center'
        });

        var modal = doc.createElement('div');
        Object.assign(modal.style, {
            background: '#0f111a',
            border: '1px solid rgba(0,204,255,0.3)',
            borderRadius: '14px',
            padding: '28px 32px',
            minWidth: '380px',
            boxShadow: '0 20px 60px rgba(0,0,0,0.6)'
        });

        var rowStyle = 'display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:13px;color:#9ca3af;font-family:Inter,sans-serif;';
        var keyStyle = 'background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);border-radius:5px;padding:2px 8px;font-family:monospace;color:#e2e4f0;font-size:12px;';

        modal.innerHTML =
            '<h3 style="color:#00ccff;font-size:15px;margin:0 0 18px 0;letter-spacing:1px;font-family:Inter,sans-serif;">⌨️ SKRÓTY KLAWISZOWE</h3>' +
            '<div style="' + rowStyle + '"><span>Wyszukaj moduł</span><span style="' + keyStyle + '">Ctrl + K</span></div>' +
            '<div style="' + rowStyle + '"><span>Uruchom symulację</span><span style="' + keyStyle + '">Ctrl + Enter</span></div>' +
            '<div style="' + rowStyle + '"><span>Ten dialog (Pomoc)</span><span style="' + keyStyle + '">?</span></div>' +
            '<div style="' + rowStyle + '"><span>Zamknij overlay</span><span style="' + keyStyle + '">Esc</span></div>' +
            '<div style="' + rowStyle.replace('border-bottom:1px solid rgba(255,255,255,0.05);','') + '"><span>Nawigacja + Wybierz</span><span style="' + keyStyle + '">↑↓ + Enter</span></div>' +
            '<p style="margin:18px 0 0 0;font-size:10px;color:#4a4e6a;font-family:Inter,sans-serif;">Barbell Strategy Dashboard v9.5+</p>';

        overlay.appendChild(modal);
        doc.body.appendChild(overlay);

        function openKb() { overlay.style.display = 'flex'; }
        function closeKb() { overlay.style.display = 'none'; }

        doc.addEventListener('keydown', function(e) {
            var tag = ((doc.activeElement || {}).tagName || '').toUpperCase();
            var isInput = ['INPUT','TEXTAREA','SELECT'].indexOf(tag) >= 0;
            if (e.key === '?' && !isInput && !e.ctrlKey && !e.metaKey) {
                e.preventDefault();
                overlay.style.display === 'none' ? openKb() : closeKb();
            }
            if (e.key === 'Escape' && overlay.style.display !== 'none') {
                e.preventDefault(); closeKb();
            }
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                var btn = doc.querySelector('[data-testid="stAppViewContainer"] [data-testid="stButton"] button') ||
                          doc.querySelector('[data-testid="stMainBlockContainer"] .stButton button');
                if (btn) { e.preventDefault(); btn.click(); }
            }
        });

        overlay.addEventListener('click', function(e) {
            if (e.target === overlay) closeKb();
        });
    })();
    </script>
    """
    components.html(js_code, height=0, width=0)


@st.cache_resource
def apply_styling() -> str:
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&family=JetBrains+Mono:wght@400;700&display=swap');

    /* ═══════════════════════════════════════════════════════════════
       ROOT VARIABLES — Premium Cyberpunk Palette (OKLCH V.1)
    ═══════════════════════════════════════════════════════════════ */
    :root {
        --bg-deep:     #05060d;
        --bg-mid:      #0d0f1a;
        --bg-surface:  #0f111a;
        --bg-card:     rgba(15, 17, 28, 0.85);
        --border:      rgba(255,255,255,0.06);
        --border-glow: rgba(0, 230, 118, 0.25);
        
        /* Modern OKLCH color system — perceptually uniform (V.1) */
        --green:       oklch(85% 0.25 142);
        --green-dim:   rgba(0, 230, 118, 0.08); /* fallback hex: #00e676 */
        --cyan:        oklch(82% 0.14 220);     /* original: #00ccff */
        --yellow:      oklch(85% 0.18 95);      /* original: #ffea00 */
        --red:         oklch(58% 0.24 25);      /* original: #ff1744 */
        --purple:      oklch(65% 0.20 303);     /* original: #a855f7 */
        
        --text:        #e2e4f0;
        /* WCAG/APCA Contrast fixes (V.9) */
        --text-dim:    #8b95a5;
        --text-note:   #9da8b8;
        --text-caption:#7a839a;
        
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
    /* UI FIX: Usunięto margin-top: -14px który przesuwał wykresy w górę zakrywając nagłówki sekcji.
       Gauges w Control Center mają własny override lokalnie w app.py. */
    .stPlotlyChart { margin-bottom: 0; margin-top: 0; }
    h4 { margin-bottom: 4px !important; margin-top: 4px !important; }

    /* ═══════════════════════════════════════════════════════════════
       HEADINGS — gradient text & variable fonts (V.2)
    ═══════════════════════════════════════════════════════════════ */
    h1 {
        background: linear-gradient(90deg, var(--green) 0%, var(--cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.6rem !important;
        font-weight: 900 !important;
        font-optical-sizing: auto;
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

    /* ═══════════════════════════════════════════════════════════════
       MODULE HEADER — breadcrumb + title + subtitle
    ═══════════════════════════════════════════════════════════════ */
    .module-header {
        padding: 0 0 16px 0;
        border-bottom: 1px solid rgba(0,230,118,0.12);
        margin-bottom: 18px;
    }
    .module-breadcrumb {
        font-size: 11px;
        color: var(--text-dim);
        letter-spacing: 0.5px;
        margin-bottom: 6px;
        opacity: 0.7;
    }
    .module-breadcrumb a {
        color: var(--green);
        text-decoration: none;
    }
    .module-title {
        font-size: 1.8rem;
        font-weight: 900;
        background: linear-gradient(90deg, var(--green) 0%, var(--cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
        letter-spacing: -0.3px;
    }
    .module-subtitle {
        font-size: 13px;
        color: #6b7280;
        margin-top: 6px;
        line-height: 1.5;
    }
    .module-badge {
        display: inline-block;
        margin-top: 8px;
        padding: 3px 10px;
        background: rgba(0,230,118,0.08);
        border: 1px solid rgba(0,230,118,0.2);
        border-radius: 20px;
        font-size: 11px;
        color: var(--green);
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* ═══════════════════════════════════════════════════════════════
       ACTION PANEL — Panel Akcji (Control Center)
    ═══════════════════════════════════════════════════════════════ */
    .action-panel {
        background: linear-gradient(135deg, rgba(10,11,20,0.95), rgba(15,17,28,0.95));
        border: 1px solid rgba(0,230,118,0.15);
        border-left: 4px solid var(--green);
        border-radius: 12px;
        padding: 16px 20px;
        margin-top: 8px;
    }
    .action-panel.alarm {
        border-left-color: var(--red);
        border-color: rgba(255,23,68,0.2);
    }
    .action-panel.warning {
        border-left-color: var(--yellow);
        border-color: rgba(255,234,0,0.15);
    }
    .action-rec-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255,255,255,0.03);
        font-size: 13px;
        color: var(--text);
    }
    .action-rec-item:last-child { border-bottom: none; }

    /* ═══════════════════════════════════════════════════════════════
       KEYBOARD SHORTCUTS OVERLAY
    ═══════════════════════════════════════════════════════════════ */
    #kb-shortcuts-overlay {
        position: fixed; top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: rgba(5,6,13,0.75);
        backdrop-filter: blur(6px);
        z-index: 999998;
        display: none;
        align-items: center;
        justify-content: center;
    }
    #kb-shortcuts-modal {
        background: #0f111a;
        border: 1px solid rgba(0,204,255,0.3);
        border-radius: 14px;
        padding: 28px 32px;
        min-width: 380px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.6), 0 0 30px rgba(0,204,255,0.08);
    }
    #kb-shortcuts-modal h3 {
        color: #00ccff;
        font-size: 15px;
        margin: 0 0 18px 0;
        letter-spacing: 1px;
    }
    .kb-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 7px 0;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        font-size: 13px;
        color: #9ca3af;
    }
    .kb-row:last-child { border-bottom: none; }
    .kb-key {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 5px;
        padding: 2px 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #e2e4f0;
    }

    /* ═══════════════════════════════════════════════════════════════
       TYPOGRAFIA — Tabular Nums (V.2)
    ═══════════════════════════════════════════════════════════════ */
    .stMetric label { color: var(--text-dim) !important; }
    .stMetric [data-testid="stMetricValue"] {
        color: var(--green) !important;
        font-family: var(--mono);
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        text-shadow: 0 0 10px var(--green-dim);
    }
    .stMetric [data-testid="stMetricDelta"] svg { fill: var(--green) !important; }
    .stMetric [data-testid="stMetricDelta"] div { color: var(--green) !important; }

    .gauge-number, .tabular-nums, .st-emotional-cache-1wivap2 {
        font-variant-numeric: tabular-nums;
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


@st.cache_resource
def module_header(title: str, subtitle: str = "", icon: str = "", badge: str = "") -> str:
    """
    Zwraca HTML nagłówka modułu z breadcrumb, tytułem i opisem.
    Użycie: st.markdown(module_header(...), unsafe_allow_html=True)
    """
    badge_html = f"<span class='module-badge'>{badge}</span>" if badge else ""
    subtitle_html = f"<div class='module-subtitle'>{subtitle}</div>" if subtitle else ""
    icon_prefix = f"{icon} " if icon else ""
    return f"""
    <div class='module-header fade-up'>
        <div class='module-breadcrumb'>🏠 Dashboard → {title}</div>
        <div class='module-title'>{icon_prefix}{title}</div>
        {subtitle_html}
        {badge_html}
    </div>
    """


# Słownik historycznych kryzysów finansowych dla adnotacji na wykresach
CRISIS_EVENTS: dict = {
    "COVID-19 Crash": ("2020-02-19", "2020-03-23", "#ff1744"),
    "Bear Market 2022": ("2022-01-03", "2022-10-13", "#f39c12"),
    "SVB Crisis": ("2023-03-08", "2023-03-17", "#a855f7"),
    "GFC 2008–09": ("2008-09-15", "2009-03-09", "#ff1744"),
    "Flash Crash 2010": ("2010-05-06", "2010-05-10", "#00ccff"),
}


def add_crisis_annotations(fig, show: bool = True, opacity: float = 0.10) -> None:
    """
    Nakłada kolorowe prostokąty i etykiety historycznych kryzysów na wykres Plotly.

    Parameters
    ----------
    fig      : go.Figure — wykres do modyfikacji (in-place)
    show     : bool — czy rysować adnotacje
    opacity  : float — przezroczystość wypełnienia (domyślnie 0.10)
    """
    if not show:
        return
    for name, (start, end, color) in CRISIS_EVENTS.items():
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor=color,
            opacity=opacity,
            layer="below",
            line_width=0,
            annotation_text=name,
            annotation_position="top left",
            annotation=dict(
                font=dict(size=9, color=color, family="Inter"),
                opacity=0.85,
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  V.3 — UJEDNOLICONY PLOTLY THEME (Grammar of Graphics, Wilkinson 2005/2024)
# ─────────────────────────────────────────────────────────────────────────────

_CHART_AXIS_BASE = dict(
    gridcolor="#1c1c2e", gridwidth=0.5,
    zerolinecolor="#2a2a3a", zerolinewidth=1,
    tickfont=dict(size=10, color="#9ca3af"),
    linecolor="#2a2a3a", linewidth=1,
)

_CHART_BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e4f0", size=12),
    legend=dict(
        bgcolor="rgba(0,0,0,0)", borderwidth=0,
        font=dict(size=10, color="#9ca3af"),
        orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1.0,
    ),
    margin=dict(l=50, r=20, t=48, b=40),
    colorway=["#00e676", "#00ccff", "#a855f7", "#f39c12", "#ff1744",
              "#3498db", "#2ecc71", "#e67e22"],
    hoverlabel=dict(
        bgcolor="#0f111a", bordercolor="#2a2a3a",
        font=dict(family="Inter, monospace", size=11, color="#e2e4f0"),
    ),
)


def plotly_theme(height: int = 380, title: str = "", show_legend: bool = True) -> dict:
    """
    Zwraca dict gotowy do fig.update_layout(**plotly_theme(...)).
    Jeden standard ciemnego designu dla WSZYSTKICH wykresów projektu.
    Ref: Wilkinson (2005) "Grammar of Graphics" — spójny system stylu.

    Użycie:
        fig.update_layout(**plotly_theme(height=350, title="Mój wykres"))
    """
    layout = dict(**_CHART_BASE_LAYOUT)
    layout["height"]     = height
    layout["showlegend"] = show_legend
    if title:
        layout["title"] = dict(
            text=title,
            font=dict(size=13, color="#e2e4f0", family="Inter"),
            x=0.0, xanchor="left",
        )
    return layout


def apply_chart_theme(fig, height: int = 380, title: str = "") -> None:
    """
    Stosuje ujednolicony ciemny motyw do istniejącego go.Figure (in-place).
    Aktualizuje wszystkie osie (w tym subploty).

    Użycie:
        apply_chart_theme(fig, height=320, title="Volatility Clustering")
    """
    fig.update_layout(**plotly_theme(height=height, title=title))
    fig.update_xaxes(**_CHART_AXIS_BASE)
    fig.update_yaxes(**_CHART_AXIS_BASE)


# ─────────────────────────────────────────────────────────────────────────────
#  V.5 — PROGRESSIVE DISCLOSURE — SciCard
#  Ref: Miller (1956) Magical Number 7; Sweller (2011) Cognitive Load Theory
# ─────────────────────────────────────────────────────────────────────────────

def scicard(
    title: str,
    icon: str,
    level0_html: str,
    chart_fn=None,
    explanation_md: str = "",
    formula_code: str = "",
    reference: str = "",
    accent_color: str = "#00e676",
    key_prefix: str = "",
) -> None:
    """
    Renderuje 3-poziomową kartę naukową z Progressive Disclosure.

    Poziom 0 (zawsze)   : kluczowa metryka / wynik (level0_html)
    Poziom 1 (expander) : wykres + wyjaśnienie
    Poziom 2 (expander) : wzory matematyczne + referencje naukowe

    Użycie:
        scicard(
            title="Permutation Entropy",
            icon="🌀",
            level0_html="<b style='color:#00e676;font-size:24px'>0.42</b>",
            chart_fn=lambda: st.plotly_chart(fig, use_container_width=True),
            explanation_md="**Entropia permutacyjna** mierzy...",
            formula_code="PermEn(m,τ) = -Σ p(π) · log₂ p(π)",
            reference="Bandt & Pompe (2002), PRL 88:174102",
        )
    """
    import streamlit as _st

    c = accent_color.lstrip("#")
    try:
        r_c, g_c, b_c = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        border_col = f"rgba({r_c},{g_c},{b_c},0.25)"
        bg_col     = f"rgba({r_c},{g_c},{b_c},0.04)"
    except ValueError:
        border_col = "rgba(0,230,118,0.25)"
        bg_col     = "rgba(0,230,118,0.04)"

    _st.markdown(f"""
    <div data-aos="fade-up" style="background:{bg_col};border:1px solid {border_col};
                border-left:3px solid {accent_color};border-radius:10px;
                padding:12px 16px;margin-bottom:4px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
        <span style="font-size:20px">{icon}</span>
        <span style="font-size:13px;font-weight:700;color:#e2e4f0;
               letter-spacing:0.5px">{title}</span>
      </div>
      {level0_html}
    </div>
    """, unsafe_allow_html=True)

    if chart_fn is not None or explanation_md:
        with _st.expander("📊 Rozwiń — wykres i opis", expanded=False):
            if chart_fn is not None:
                chart_fn()
            if explanation_md:
                _st.markdown(explanation_md)

    if formula_code or reference:
        with _st.expander("🧮 Matematyka i Źródło Naukowe"):
            if formula_code:
                _st.markdown(
                    f"<div style=\"font-family:'JetBrains Mono',monospace;"
                    f"font-size:13px;color:#00ccff;background:rgba(0,204,255,0.06);"
                    f"border-left:3px solid #00ccff;padding:10px 14px;"
                    f"border-radius:0 6px 6px 0;margin-bottom:8px;\">"
                    f"{formula_code}</div>",
                    unsafe_allow_html=True,
                )
            if reference:
                _st.caption(f"📚 Źródło: {reference}")


# ─────────────────────────────────────────────────────────────────────────────
#  V.6 — METRIC SPARK CARDS (Few 2024 "Show Me the Numbers")
# ─────────────────────────────────────────────────────────────────────────────

def _spark_svg(history: list, color: str = "#00e676",
               width: int = 80, height: int = 28) -> str:
    """Generuje miniaturowy sparkline SVG."""
    if not history or len(history) < 2:
        return ""
    h   = history[-20:] if len(history) > 20 else history
    n   = len(h)
    mn, mx = min(h), max(h)
    rng = mx - mn if mx != mn else 1e-9
    pad = 3
    pts = []
    for i, v in enumerate(h):
        x = pad + (i / (n - 1)) * (width - 2 * pad)
        y = (height - pad) - ((v - mn) / rng) * (height - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    poly = " ".join(pts)
    return (f'<svg width="{width}" height="{height}" style="display:block">'
            f'<polyline points="{poly}" fill="none" stroke="{color}" '
            f'stroke-width="1.8" stroke-linecap="round" '
            f'stroke-linejoin="round" opacity="0.85"/></svg>')


def metric_spark_html(
    label: str,
    value: str,
    suffix: str = "",
    delta: str = "",
    delta_positive: bool = True,
    history: list = None,
    accent_color: str = "#00e676",
    help_text: str = "",
    width: str = "100%",
) -> str:
    """
    Zwraca HTML premium karty metrykowej ze sparkline i deltą.
    Ref: Few, S. (2024) "Show Me the Numbers" — spark charts +
         contextual coloring for rapid data comprehension.

    Użycie:
        st.markdown(metric_spark_html(
            "VIX 1M", "23.4", suffix="", delta="+1.2",
            delta_positive=False, history=[18,19,21,22,23.4],
            accent_color="#ff1744",
        ), unsafe_allow_html=True)
    """
    spark = ""
    if history:
        scol = accent_color if delta_positive else "#ff1744"
        spark = f"<div style='margin-top:4px;opacity:0.8'>{_spark_svg(history, scol)}</div>"

    d_color = "#00e676" if delta_positive else "#ff1744"
    d_arrow = "▲" if delta_positive else "▼"
    d_html  = (
        f"<div style='font-size:10px;color:{d_color};font-weight:600;margin-top:2px;'>"
        f"{d_arrow} {delta}</div>"
    ) if delta else ""

    safe_help = help_text.replace('"', "'").replace("\n", " ")

    return (
        f'<div title="{safe_help}" style="'
        f'background:linear-gradient(135deg,rgba(15,17,26,0.95),rgba(20,22,35,0.9));'
        f'border:1px solid rgba(255,255,255,0.06);border-top:2px solid {accent_color};'
        f'border-radius:10px;padding:10px 12px 8px 12px;width:{width};min-width:90px;'
        f'cursor:default;transition:border-color 0.2s ease,box-shadow 0.2s ease;" '
        f'onmouseover="this.style.borderColor=\'{accent_color}\';" '
        f'onmouseout="this.style.borderColor=\'rgba(255,255,255,0.06)\';">'
        f'<div style="font-size:9px;color:#6b7280;letter-spacing:0.8px;'
        f'text-transform:uppercase;font-weight:600;margin-bottom:4px;">{label}</div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:20px;'
        f'font-weight:700;color:{accent_color};font-variant-numeric:tabular-nums;'
        f'line-height:1.1;">{value}'
        f'<span style="font-size:11px;color:#9ca3af;font-weight:400;margin-left:2px;">'
        f'{suffix}</span></div>'
        f'{d_html}{spark}'
        f'</div>'
    )


def metric_spark_row(cards: list, columns: int = 5) -> None:
    """
    Renderuje rząd kart sparkline w kolumnach Streamlit.

    Parameters
    ----------
    cards   : lista dict z kluczami: label, value, suffix, delta,
                delta_positive, history, accent_color, help_text
    columns : liczba kolumn layoutu

    Użycie:
        metric_spark_row([
            {"label": "VIX 1M", "value": "23.4", "delta": "+1.2",
             "delta_positive": False, "history": [18,19,21,22,23.4],
             "accent_color": "#ff1744"},
            ...
        ], columns=5)
    """
    import streamlit as _st
    cols = _st.columns(columns)
    for i, card in enumerate(cards):
        with cols[i % columns]:
            _st.markdown(
                metric_spark_html(
                    label          = card.get("label", ""),
                    value          = str(card.get("value", "—")),
                    suffix         = card.get("suffix", ""),
                    delta          = card.get("delta", ""),
                    delta_positive = card.get("delta_positive", True),
                    history        = card.get("history"),
                    accent_color   = card.get("accent_color", "#00e676"),
                    help_text      = card.get("help_text", ""),
                ),
                unsafe_allow_html=True,
            )

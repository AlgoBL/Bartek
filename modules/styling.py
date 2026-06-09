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
    """Legacy stub — zastąpiona przez inject_spotlight_js(). Zachowana dla kompatybilności."""
    pass  # inject_spotlight_js() jest teraz wywoływana bezpośrednio w app.py


def inject_spotlight_js(search_index_json: str = "[]"):
    """
    Wstrzykuje Spotlight v2 — pełna wyszukiwarka projektu Barbell Dashboard.

    Funkcje:
    - Fuzzy search (wszystkie znaki query w kolejności w target)
    - Score ranking (exact > prefix > fuzzy > fallback)
    - Kategorie wyników (Moduły, Wskaźniki, Metody Quant, Akcje)
    - Panel podglądu po prawej stronie (dwukolumnowy layout Spotlight)
    - Pełna obsługa klawiatury (↑↓ Enter Esc, Tab kategorii)
    - Podświetlanie dopasowanych znaków (<mark>)
    - MRU — ostatnio używane (localStorage)
    - Indeks z search_index_json — pełne metadane projektu
    - Fallback na linki nawigacyjne gdy indeks pusty

    Args:
        search_index_json: JSON string z wynikiem build_search_index()
    """
    import streamlit.components.v1 as components

    # Escapuj backtick'i w JSON żeby nie psuf template literal JS
    safe_json = search_index_json.replace("\\", "\\\\").replace("`", "\\`")

    js_code = f"""
    <script>
    (function() {{
        var doc = window.parent.document;
        
        // CLEANUP PREVIOUS INJECTIONS TO PREVENT DEAD EVENT LISTENERS
        if (doc.getElementById("spotlight-overlay-v2")) doc.getElementById("spotlight-overlay-v2").remove();
        if (doc.getElementById("spotlight-trigger-btn")) doc.getElementById("spotlight-trigger-btn").remove();
        if (doc.getElementById("spotlight-styles-v2")) doc.getElementById("spotlight-styles-v2").remove();
        if (doc._spotlightKeydownV2) doc.removeEventListener('keydown', doc._spotlightKeydownV2, true);

        // ═══════════════════════════════════════════════════════
        //  SEARCH INDEX — wstrzykiwany z Pythona
        // ═══════════════════════════════════════════════════════
        var SEARCH_INDEX = [];
        try {{
            SEARCH_INDEX = JSON.parse(`{safe_json}`);
        }} catch(e) {{
            console.warn("Spotlight: błąd parsowania indeksu", e);
        }}

        // ═══════════════════════════════════════════════════════
        //  CSS STYLES
        // ═══════════════════════════════════════════════════════
        if (!doc.getElementById("spotlight-styles-v2")) {{
            var styleEl = doc.createElement("style");
            styleEl.id = "spotlight-styles-v2";
            styleEl.textContent = `
                #spotlight-overlay-v2 {{
                    position: fixed; top: 0; left: 0;
                    width: 100vw; height: 100vh;
                    background: rgba(5, 6, 13, 0.75);
                    backdrop-filter: blur(8px);
                    -webkit-backdrop-filter: blur(8px);
                    z-index: 9999999;
                    display: none;
                    align-items: flex-start;
                    justify-content: center;
                    padding-top: 12vh;
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                }}
                #spotlight-modal-v2 {{
                    width: 780px; max-width: 94vw;
                    background: #0d0f1c;
                    border: 1px solid rgba(0, 230, 118, 0.35);
                    border-radius: 16px;
                    box-shadow: 0 24px 80px rgba(0,0,0,0.7),
                                0 0 0 1px rgba(0,230,118,0.08),
                                0 0 40px rgba(0,230,118,0.10);
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                    max-height: 70vh;
                    animation: spotlightIn 0.18s cubic-bezier(0.16,1,0.3,1);
                }}
                @keyframes spotlightIn {{
                    from {{ opacity: 0; transform: scale(0.96) translateY(-8px); }}
                    to   {{ opacity: 1; transform: scale(1) translateY(0); }}
                }}
                #spotlight-input-row {{
                    display: flex;
                    align-items: center;
                    padding: 0 18px;
                    border-bottom: 1px solid rgba(255,255,255,0.06);
                    flex-shrink: 0;
                }}
                #spotlight-search-icon {{
                    font-size: 18px; margin-right: 12px; opacity: 0.6;
                    flex-shrink: 0;
                }}
                #spotlight-input {{
                    flex: 1; padding: 18px 0; font-size: 17px;
                    background: transparent; border: none;
                    color: #e2e4f0; outline: none; font-family: inherit;
                }}
                #spotlight-input::placeholder {{ color: #4a5068; }}
                #spotlight-shortcut-hint {{
                    font-size: 11px; color: #3a3f5a;
                    background: rgba(255,255,255,0.04);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 5px; padding: 3px 8px;
                    flex-shrink: 0; font-family: monospace;
                }}
                #spotlight-body {{
                    display: flex;
                    flex: 1;
                    overflow: hidden;
                    min-height: 0;
                }}
                #spotlight-results {{
                    flex: 0 0 55%;
                    overflow-y: auto;
                    padding: 6px 0 12px;
                    border-right: 1px solid rgba(255,255,255,0.05);
                }}
                #spotlight-results::-webkit-scrollbar {{ width: 4px; }}
                #spotlight-results::-webkit-scrollbar-thumb {{
                    background: rgba(0,230,118,0.2); border-radius: 2px;
                }}
                #spotlight-preview {{
                    flex: 0 0 45%;
                    padding: 24px 20px;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    overflow-y: auto;
                    min-width: 0;
                }}
                .spotlight-group-header {{
                    padding: 10px 16px 4px;
                    font-size: 10px;
                    color: #4a5068;
                    text-transform: uppercase;
                    letter-spacing: 1.5px;
                    font-weight: 700;
                }}
                .spotlight-item {{
                    display: flex;
                    align-items: center;
                    padding: 9px 14px;
                    margin: 1px 8px;
                    cursor: pointer;
                    border-radius: 8px;
                    border-left: 3px solid transparent;
                    transition: background 0.08s ease, border-color 0.08s ease;
                }}
                .spotlight-item:hover, .spotlight-item.active {{
                    background: rgba(0,230,118,0.08);
                    border-left-color: #00e676;
                }}
                .spotlight-item.active .spotlight-item-title {{
                    color: #00e676;
                }}
                .spotlight-item-icon {{
                    font-size: 16px;
                    margin-right: 10px;
                    flex-shrink: 0;
                    width: 22px;
                    text-align: center;
                }}
                .spotlight-item-text {{ flex: 1; min-width: 0; }}
                .spotlight-item-title {{
                    font-size: 13.5px;
                    color: #d4d8ef;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    font-weight: 500;
                }}
                .spotlight-item-cat {{
                    font-size: 10.5px;
                    color: #4a5068;
                    margin-top: 1px;
                }}
                .spotlight-item-title mark {{
                    background: rgba(0,230,118,0.25);
                    color: #00e676;
                    border-radius: 2px;
                    padding: 0 1px;
                }}
                #spotlight-preview-icon {{
                    font-size: 42px;
                    margin-bottom: 12px;
                    line-height: 1;
                }}
                #spotlight-preview-title {{
                    font-size: 16px;
                    font-weight: 700;
                    color: #e2e4f0;
                    margin-bottom: 6px;
                    line-height: 1.3;
                }}
                #spotlight-preview-subtitle {{
                    font-size: 12px;
                    color: #6b7591;
                    line-height: 1.55;
                    flex: 1;
                }}
                #spotlight-preview-open {{
                    margin-top: 16px;
                    padding: 8px 16px;
                    background: rgba(0,230,118,0.12);
                    border: 1px solid rgba(0,230,118,0.3);
                    border-radius: 8px;
                    color: #00e676;
                    font-size: 12px;
                    font-weight: 600;
                    cursor: pointer;
                    font-family: inherit;
                    transition: background 0.15s, border-color 0.15s;
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    width: fit-content;
                }}
                #spotlight-preview-open:hover {{
                    background: rgba(0,230,118,0.2);
                    border-color: rgba(0,230,118,0.5);
                }}
                #spotlight-footer {{
                    display: flex;
                    align-items: center;
                    gap: 16px;
                    padding: 8px 16px;
                    border-top: 1px solid rgba(255,255,255,0.05);
                    flex-shrink: 0;
                }}
                .spotlight-kb {{
                    display: inline-flex;
                    align-items: center;
                    gap: 5px;
                    font-size: 10px;
                    color: #3a3f5a;
                }}
                .spotlight-kb kbd {{
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.10);
                    border-radius: 4px;
                    padding: 2px 6px;
                    font-family: monospace;
                    font-size: 10px;
                    color: #6b7591;
                }}
                #spotlight-no-results {{
                    padding: 30px 20px;
                    text-align: center;
                    color: #3a3f5a;
                    font-size: 13px;
                }}
                #spotlight-trigger-btn {{
                    position: fixed;
                    bottom: 24px;
                    right: 24px;
                    z-index: 99998;
                    background: rgba(13,15,28,0.92);
                    border: 1px solid rgba(0,230,118,0.35);
                    border-radius: 12px;
                    padding: 8px 14px;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    font-family: 'Inter', sans-serif;
                    font-size: 12px;
                    color: #6b7591;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
                    transition: all 0.2s ease;
                    backdrop-filter: blur(8px);
                }}
                #spotlight-trigger-btn:hover {{
                    border-color: rgba(0,230,118,0.7);
                    color: #00e676;
                    box-shadow: 0 4px 24px rgba(0,230,118,0.15);
                }}
                #spotlight-trigger-btn kbd {{
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 4px;
                    padding: 1px 5px;
                    font-size: 10px;
                    font-family: monospace;
                }}
            `;
            doc.head.appendChild(styleEl);
        }}

        // ═══════════════════════════════════════════════════════
        //  DOM STRUCTURE
        // ═══════════════════════════════════════════════════════
        var overlay = doc.createElement("div");
        overlay.id = "spotlight-overlay-v2";

        var modal = doc.createElement("div");
        modal.id = "spotlight-modal-v2";

        // — Input row —
        var inputRow = doc.createElement("div");
        inputRow.id = "spotlight-input-row";

        var searchIcon = doc.createElement("span");
        searchIcon.id = "spotlight-search-icon";
        searchIcon.textContent = "🔍";

        var inputEl = doc.createElement("input");
        inputEl.id = "spotlight-input";
        inputEl.type = "text";
        inputEl.placeholder = "Szukaj modułów, wskaźników, metod... (Ctrl+K)";
        inputEl.autocomplete = "off";
        inputEl.spellcheck = false;

        var hintEl = doc.createElement("span");
        hintEl.id = "spotlight-shortcut-hint";
        hintEl.textContent = "Ctrl+K";

        inputRow.appendChild(searchIcon);
        inputRow.appendChild(inputEl);
        inputRow.appendChild(hintEl);

        // — Body (results + preview) —
        var body = doc.createElement("div");
        body.id = "spotlight-body";

        var resultsPane = doc.createElement("div");
        resultsPane.id = "spotlight-results";

        var previewPane = doc.createElement("div");
        previewPane.id = "spotlight-preview";
        previewPane.innerHTML = `
            <div>
                <div id="spotlight-preview-icon">🔦</div>
                <div id="spotlight-preview-title">Barbell Spotlight</div>
                <div id="spotlight-preview-subtitle">
                    Wyszukaj moduły, wskaźniki makro, metody quantitative, ustawienia portfela i więcej.<br><br>
                    Indeks zawiera <strong style="color:#00e676">${{SEARCH_INDEX.length}}</strong> elementów projektu.
                </div>
            </div>
            <div></div>
        `;

        body.appendChild(resultsPane);
        body.appendChild(previewPane);

        // — Footer —
        var footer = doc.createElement("div");
        footer.id = "spotlight-footer";
        footer.innerHTML = `
            <span class="spotlight-kb"><kbd>↑</kbd><kbd>↓</kbd> nawigacja</span>
            <span class="spotlight-kb"><kbd>Enter</kbd> otwórz</span>
            <span class="spotlight-kb"><kbd>Esc</kbd> zamknij</span>
            <span style="flex:1"></span>
            <span style="font-size:10px;color:#2a2f45;">Barbell Spotlight v2</span>
        `;

        modal.appendChild(inputRow);
        modal.appendChild(body);
        modal.appendChild(footer);
        overlay.appendChild(modal);
        doc.body.appendChild(overlay);

        // — Floating trigger button —
        var triggerBtn = doc.createElement("button");
        triggerBtn.id = "spotlight-trigger-btn";
        triggerBtn.innerHTML = `🔍 Szukaj &nbsp; <kbd>Ctrl+K</kbd>`;
        triggerBtn.title = "Otwórz Spotlight (Ctrl+K)";
        doc.body.appendChild(triggerBtn);
        triggerBtn.addEventListener("click", openModal);

        // ═══════════════════════════════════════════════════════
        //  STATE
        // ═══════════════════════════════════════════════════════
        var activeIndex = 0;
        var flatList = [];
        var currentSelected = null;

        // ═══════════════════════════════════════════════════════
        //  MRU (Most Recently Used)
        // ═══════════════════════════════════════════════════════
        function getMRU() {{
            try {{ return JSON.parse(localStorage.getItem('barbell_spotlight_mru_v2')) || []; }}
            catch(e) {{ return []; }}
        }}
        function saveMRU(id) {{
            var mru = getMRU().filter(function(x) {{ return x !== id; }});
            mru.unshift(id);
            if (mru.length > 8) mru = mru.slice(0, 8);
            localStorage.setItem('barbell_spotlight_mru_v2', JSON.stringify(mru));
        }}

        // ═══════════════════════════════════════════════════════
        //  FUZZY SEARCH ENGINE
        // ═══════════════════════════════════════════════════════
        function fuzzyScore(query, target) {{
            if (!query) return 50;
            var q = query.toLowerCase();
            var t = target.toLowerCase();

            // Exact match
            if (t === q) return 100;
            // Starts with
            if (t.startsWith(q)) return 90;
            // Contains substring
            if (t.includes(q)) return 75;

            // Fuzzy: all chars in order
            var qi = 0;
            var score = 0;
            var lastMatchPos = -1;
            for (var ti = 0; ti < t.length && qi < q.length; ti++) {{
                if (t[ti] === q[qi]) {{
                    // Bonus for consecutive chars
                    score += (lastMatchPos === ti - 1) ? 3 : 1;
                    lastMatchPos = ti;
                    qi++;
                }}
            }}
            if (qi < q.length) return 0; // not all chars found
            // Scale to 40-70 range
            return Math.min(70, 40 + score * 2);
        }}

        function scoreItem(item, query) {{
            if (!query) return 50 * (item.score_boost || 1.0);

            var q = query.toLowerCase();
            var maxScore = 0;

            // Score title
            var titleScore = fuzzyScore(q, item.title);
            if (titleScore > maxScore) maxScore = titleScore;

            // Score subtitle
            if (item.subtitle) {{
                var subScore = fuzzyScore(q, item.subtitle) * 0.7;
                if (subScore > maxScore) maxScore = subScore;
            }}

            // Score tags
            if (item.tags) {{
                for (var i = 0; i < item.tags.length; i++) {{
                    var tagScore = fuzzyScore(q, item.tags[i]) * 0.85;
                    if (tagScore > maxScore) maxScore = tagScore;
                }}
            }}

            // Score category
            if (item.category) {{
                var catScore = fuzzyScore(q, item.category) * 0.5;
                if (catScore > maxScore) maxScore = catScore;
            }}

            return maxScore * (item.score_boost || 1.0);
        }}

        function highlightMatches(text, query) {{
            if (!query) return escapeHtml(text);
            var q = query.toLowerCase();
            var t = text;

            // Try exact substring highlight first
            var lowerT = t.toLowerCase();
            var idx = lowerT.indexOf(q);
            if (idx >= 0) {{
                return escapeHtml(t.slice(0, idx))
                     + '<mark>' + escapeHtml(t.slice(idx, idx + q.length)) + '</mark>'
                     + escapeHtml(t.slice(idx + q.length));
            }}

            // Fuzzy char-by-char highlight
            var result = '';
            var qi = 0;
            var chars = t.split('');
            for (var i = 0; i < chars.length; i++) {{
                if (qi < q.length && chars[i].toLowerCase() === q[qi]) {{
                    result += '<mark>' + escapeHtml(chars[i]) + '</mark>';
                    qi++;
                }} else {{
                    result += escapeHtml(chars[i]);
                }}
            }}
            return result;
        }}

        function escapeHtml(str) {{
            return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
                      .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
        }}

        // ═══════════════════════════════════════════════════════
        //  NAV LINKS (Streamlit sidebar)
        // ═══════════════════════════════════════════════════════
        function getNavLinks() {{
            var items = doc.querySelectorAll('[data-testid="stSidebarNavLink"]');
            var result = [];
            items.forEach(function(el) {{
                var text = (el.textContent || el.innerText || '').trim();
                result.push({{ element: el, text: text, _isNav: true }});
            }});
            return result;
        }}

        // Znajdź element nawigacyjny dla danej strony
        function findNavElement(pagePath) {{
            var navLinks = getNavLinks();
            // Próba dopasowania po tytule lub końcówce ścieżki
            var pageName = pagePath.split('/').pop().replace('.py','').replace(/_/g,' ');
            for (var i = 0; i < navLinks.length; i++) {{
                var navText = navLinks[i].text.toLowerCase();
                if (navText.includes(pageName.toLowerCase()) ||
                    pageName.toLowerCase().includes(navText.replace(/[^a-z0-9]/g, ''))) {{
                    return navLinks[i].element;
                }}
            }}
            // Fallback: szukaj po słowach
            var words = pageName.toLowerCase().split(' ').filter(function(w) {{ return w.length > 2; }});
            for (var i = 0; i < navLinks.length; i++) {{
                var navText = navLinks[i].text.toLowerCase();
                var matches = words.filter(function(w) {{ return navText.includes(w); }});
                if (matches.length >= Math.max(1, Math.floor(words.length * 0.4))) {{
                    return navLinks[i].element;
                }}
            }}
            return null;
        }}

        // ═══════════════════════════════════════════════════════
        //  SEARCH & RENDER
        // ═══════════════════════════════════════════════════════
        function buildResults(query) {{
            var q = (query || '').trim();
            var mru = getMRU();

            if (q.length === 0) {{
                // Empty query: show MRU + all grouped
                var results = [];

                // MRU items first
                var mruItems = mru.map(function(id) {{
                    return SEARCH_INDEX.find(function(item) {{ return item.id === id; }});
                }}).filter(Boolean);

                if (mruItems.length > 0) {{
                    results.push({{ _isHeader: true, text: '🕒 Ostatnio używane' }});
                    mruItems.forEach(function(item) {{ results.push(item); }});
                }}

                // All by category
                var categories = {{}};
                SEARCH_INDEX.forEach(function(item) {{
                    var cat = item.category || '📁 Inne';
                    if (!categories[cat]) categories[cat] = [];
                    if (!mru.includes(item.id)) {{
                        categories[cat].push(item);
                    }}
                }});

                // Also include nav links not in index
                var navLinks = getNavLinks();
                var existingTitles = SEARCH_INDEX.map(function(i) {{ return i.title.toLowerCase(); }});
                navLinks.forEach(function(nl) {{
                    var t = nl.text.toLowerCase();
                    var inIndex = existingTitles.some(function(et) {{ return et.includes(t) || t.includes(et.substring(0,8)); }});
                    if (!inIndex && !mru.includes('nav_' + t)) {{
                        if (!categories['📄 Nawigacja']) categories['📄 Nawigacja'] = [];
                        categories['📄 Nawigacja'].push({{
                            id: 'nav_' + t, title: nl.text, subtitle: 'Link nawigacyjny',
                            category: '📄 Nawigacja', icon: '📄', tags: [t],
                            _navElement: nl.element
                        }});
                    }}
                }});

                var catOrder = ['⚡ Akcja', '📄 Nawigacja', '🛡️ Portfel i Ochrona', '📊 Analiza Rynkowa', '🧬 Osobiste', '⚙️ System', '📊 Wskaźnik Makro', '🔬 Metoda Quant', '📁 Inne'];
                catOrder.forEach(function(cat) {{
                    if (categories[cat] && categories[cat].length > 0) {{
                        results.push({{ _isHeader: true, text: cat }});
                        categories[cat].forEach(function(item) {{ results.push(item); }});
                    }}
                }});

                return results;
            }}

            // Search mode: score all items
            var scored = [];
            SEARCH_INDEX.forEach(function(item) {{
                var s = scoreItem(item, q);
                if (s > 0) scored.push({{ item: item, score: s }});
            }});

            // Also score nav links
            var navLinks = getNavLinks();
            navLinks.forEach(function(nl) {{
                var titleScore = fuzzyScore(q, nl.text);
                if (titleScore > 0) {{
                    scored.push({{
                        item: {{
                            id: 'nav_' + nl.text.toLowerCase(),
                            title: nl.text,
                            subtitle: 'Link nawigacyjny',
                            category: '📄 Nawigacja',
                            icon: '📄',
                            tags: [nl.text.toLowerCase()],
                            _navElement: nl.element
                        }},
                        score: titleScore * 0.8
                    }});
                }}
            }});

            // Sort by score desc
            scored.sort(function(a, b) {{ return b.score - a.score; }});

            // Group by category (preserving score order)
            var groups = {{}};
            scored.forEach(function(s) {{
                var cat = s.item.category || '📁 Inne';
                if (!groups[cat]) groups[cat] = [];
                groups[cat].push(s.item);
            }});

            var results = [];
            var catOrder = ['⚡ Akcja', '📄 Nawigacja', '🛡️ Portfel i Ochrona', '📊 Analiza Rynkowa', '🧬 Osobiste', '⚙️ System', '📊 Wskaźnik Makro', '🔬 Metoda Quant', '📁 Inne'];

            // First pass: categories in order
            catOrder.forEach(function(cat) {{
                if (groups[cat] && groups[cat].length > 0) {{
                    results.push({{ _isHeader: true, text: cat }});
                    groups[cat].forEach(function(item) {{ results.push(item); }});
                    delete groups[cat];
                }}
            }});
            // Remaining categories
            Object.keys(groups).forEach(function(cat) {{
                if (groups[cat].length > 0) {{
                    results.push({{ _isHeader: true, text: cat }});
                    groups[cat].forEach(function(item) {{ results.push(item); }});
                }}
            }});

            return results;
        }}

        function renderResults(query) {{
            resultsPane.innerHTML = '';
            flatList = [];
            activeIndex = 0;

            var results = buildResults(query);

            if (results.length === 0 || (results.length === 1 && results[0]._isHeader)) {{
                resultsPane.innerHTML = `<div id="spotlight-no-results">
                    <div style="font-size:28px;margin-bottom:8px">🔍</div>
                    Brak wyników dla "<strong style="color:#6b7591">${{escapeHtml(query)}}</strong>"<br>
                    <span style="font-size:11px;color:#2a2f45;margin-top:6px;display:block">Spróbuj innej frazy lub przeglądaj kategorie</span>
                </div>`;
                updatePreview(null, query);
                return;
            }}

            results.forEach(function(row, idx) {{
                if (row._isHeader) {{
                    var hdr = doc.createElement('div');
                    hdr.className = 'spotlight-group-header';
                    hdr.textContent = row.text;
                    resultsPane.appendChild(hdr);
                    return;
                }}

                var flatIdx = flatList.length;
                flatList.push(row);

                var el = doc.createElement('div');
                el.className = 'spotlight-item';
                if (flatIdx === 0) el.classList.add('active');
                el.setAttribute('data-flat-idx', flatIdx);

                var iconEl = doc.createElement('span');
                iconEl.className = 'spotlight-item-icon';
                iconEl.textContent = row.icon || '📄';

                var textEl = doc.createElement('div');
                textEl.className = 'spotlight-item-text';

                var titleEl = doc.createElement('div');
                titleEl.className = 'spotlight-item-title';
                titleEl.innerHTML = highlightMatches(row.title, query);

                var catEl = doc.createElement('div');
                catEl.className = 'spotlight-item-cat';
                catEl.textContent = row.category || '';

                textEl.appendChild(titleEl);
                textEl.appendChild(catEl);
                el.appendChild(iconEl);
                el.appendChild(textEl);

                el.addEventListener('mouseenter', function() {{
                    setActive(flatIdx);
                }});
                el.addEventListener('click', function() {{
                    activateItem(row);
                }});

                resultsPane.appendChild(el);
            }});

            if (flatList.length > 0) {{
                updatePreview(flatList[0], query);
            }}
        }}

        function setActive(idx) {{
            if (idx < 0) idx = 0;
            if (idx >= flatList.length) idx = flatList.length - 1;
            activeIndex = idx;

            var items = resultsPane.querySelectorAll('.spotlight-item');
            items.forEach(function(el) {{ el.classList.remove('active'); }});
            var activeEl = resultsPane.querySelector('[data-flat-idx="' + idx + '"]');
            if (activeEl) {{
                activeEl.classList.add('active');
                activeEl.scrollIntoView({{ block: 'nearest' }});
            }}

            updatePreview(flatList[idx], inputEl.value);
        }}

        function updatePreview(item, query) {{
            if (!item) {{
                doc.getElementById('spotlight-preview-icon').textContent = '🔦';
                doc.getElementById('spotlight-preview-title').textContent = 'Barbell Spotlight';
                doc.getElementById('spotlight-preview-subtitle').innerHTML =
                    'Zacznij pisać, aby wyszukać moduły, wskaźniki, metody i więcej.<br><br>' +
                    '<span style="color:#2a2f45">Indeks: <strong style="color:#00e676">' + SEARCH_INDEX.length + '</strong> elementów</span>';
                var oldBtn = doc.getElementById('spotlight-preview-open');
                if (oldBtn) oldBtn.remove();
                return;
            }}

            doc.getElementById('spotlight-preview-icon').textContent = item.icon || '📄';
            doc.getElementById('spotlight-preview-title').innerHTML = highlightMatches(item.title, query);
            doc.getElementById('spotlight-preview-subtitle').textContent = item.subtitle || item.category || '';

            // Remove old button
            var oldBtn = doc.getElementById('spotlight-preview-open');
            if (oldBtn) oldBtn.remove();

            // Add open button
            var btn = doc.createElement('button');
            btn.id = 'spotlight-preview-open';
            btn.innerHTML = '↗ Otwórz ' + escapeHtml(item.icon || '') + ' ' + escapeHtml(item.title.split('—')[0].split('–')[0].trim().substring(0,30));
            btn.addEventListener('click', function() {{ activateItem(item); }});
            previewPane.appendChild(btn);
        }}

        function activateItem(item) {{
            if (!item) return;
            saveMRU(item.id);
            closeModal();

            // 1. Bezposredni element nawigacyjny (linki z sidebara)
            if (item._navElement) {{
                item._navElement.click();
                return;
            }}

            // 2. Kliknij link w sidebarze po href = nav_url (bezpieczne w iframe)
            if (item.nav_url) {{
                var navLinks = getNavLinks();
                for (var i = 0; i < navLinks.length; i++) {{
                    var el = navLinks[i].element;
                    var elHref = el.getAttribute('href') || '';
                    // porownaj koniec href z nav_url
                    var navPath = item.nav_url.replace(/^\//, '');
                    if (elHref === item.nav_url ||
                        elHref.endsWith(item.nav_url) ||
                        elHref.endsWith('/' + navPath)) {{
                        el.click();
                        return;
                    }}
                }}
            }}

            // 3. Fallback: szukaj po tekscie w linkach sidebara
            var navLinks2 = getNavLinks();
            if (navLinks2.length === 0) return;
            var titleWords = (item.title || '').toLowerCase()
                .split(/[\s\-\/]+/).filter(function(w) {{ return w.length > 2; }});
            var bestMatch = null;
            var bestScore = 0;
            navLinks2.forEach(function(nl) {{
                var nlLower = nl.text.toLowerCase();
                var s = titleWords.filter(function(w) {{ return nlLower.includes(w); }}).length;
                if (s > bestScore) {{ bestScore = s; bestMatch = nl; }}
            }});
            if (bestMatch && bestScore > 0) {{
                bestMatch.element.click();
            }}
        }}

        // ═══════════════════════════════════════════════════════
        //  OPEN / CLOSE
        // ═══════════════════════════════════════════════════════
        function openModal() {{
            overlay.style.display = 'flex';
            inputEl.value = '';
            activeIndex = 0;
            renderResults('');
            setTimeout(function() {{ inputEl.focus(); }}, 60);
        }}

        function closeModal() {{
            overlay.style.display = 'none';
            inputEl.blur();
        }}

        // ═══════════════════════════════════════════════════════
        //  KEYBOARD EVENTS
        // ═══════════════════════════════════════════════════════
        doc._spotlightKeydownV2 = function(e) {{
            // Ctrl+K / Cmd+K — toggle
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {{
                e.preventDefault();
                e.stopPropagation();
                overlay.style.display === 'none' ? openModal() : closeModal();
                return;
            }}
            // Esc
            if (e.key === 'Escape' && overlay.style.display !== 'none') {{
                e.preventDefault();
                closeModal();
                return;
            }}
            // Arrow keys (only when modal open)
            if (overlay.style.display === 'none') return;
            if (e.key === 'ArrowDown') {{
                e.preventDefault();
                setActive(activeIndex + 1);
            }} else if (e.key === 'ArrowUp') {{
                e.preventDefault();
                setActive(activeIndex - 1);
            }} else if (e.key === 'Enter') {{
                e.preventDefault();
                if (flatList[activeIndex]) activateItem(flatList[activeIndex]);
            }}
        }};
        doc.addEventListener('keydown', doc._spotlightKeydownV2, true);

        inputEl.addEventListener('input', function() {{
            activeIndex = 0;
            renderResults(inputEl.value);
        }});

        overlay.addEventListener('click', function(e) {{
            if (e.target === overlay) closeModal();
        }});

        // ═══════════════════════════════════════════════════════
        //  EXPOSE for external triggering
        // ═══════════════════════════════════════════════════════
        doc.getElementById('spotlight-overlay-v2')._openSpotlight = openModal;
        doc.getElementById('spotlight-overlay-v2')._closeSpotlight = closeModal;

    }})();
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
        padding-top: 0rem !important; /* Maksymalnie do góry, brak wolnego miejsca */
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }
    h4 { margin-bottom: 4px !important; margin-top: 4px !important; }
    h1 { margin-top: -1.5rem !important; padding-top: 0 !important; }
    div[data-testid="stMarkdownContainer"] > h1 { margin-top: -1.5rem !important; }

    /* Ukrycie okruszków nawigacji i górnego paska Streamlit */
    header[data-testid="stHeader"], 
    [data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Ukrycie przycisku zwijania sidebara, jeśli wyświetla błędy/tekst */
    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    /* Jeśli Material Icons wyświetlają się jako tekst (brak fontu), ukryj je całkowicie */
    span[data-testid="stIconMaterial"],
    .st-emotion-cache-12bp31y, 
    [data-testid="stIconMaterial"] {
        display: none !important;
        visibility: hidden !important;
        font-size: 0 !important;
        color: transparent !important;
        width: 0 !important;
        height: 0 !important;
    }



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
    /* Padding top dla opcji sidebara */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem !important;
    }

    /* ── UJEDNOLICENIE CZCIONEK SIDEBARA (v10) ─────────────────────
       Bazowy rozmiar: 13.5px — identyczny z linkami nawigacyjnymi.
       Wszystkie widgety, nagłówki i etykiety w sidebarze.
    ────────────────────────────────────────────────────────────── */

    /* Globalna baza dla całego sidebara — kaskada dotrze do wszystkich elementów */
    [data-testid="stSidebar"] * {
        font-family: var(--font) !important;
    }

    /* === NAGŁÓWKI SIDEBARA (h1–h4, ### markdown) ================= */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        font-size: 13.5px !important;
        font-weight: 700 !important;
        color: #c8cde0 !important;
        letter-spacing: 0.3px !important;
        margin: 8px 0 4px 0 !important;
        text-transform: none !important;
        background: none !important;
        -webkit-text-fill-color: unset !important;
    }

    /* === ZWYKŁY TEKST / MARKDOWN W SIDEBARZE ===================== */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span {
        font-size: 13.5px !important;
        color: #9ca3af !important;
        line-height: 1.45 !important;
    }

    /* === ETYKIETY WIDGETÓW (slider, number_input, selectbox, itp.) */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSlider [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] .stNumberInput label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stToggle label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stTextArea label {
        font-size: 13.5px !important;
        color: #9ca3af !important;
        font-weight: 500 !important;
    }

    /* === SLIDER — wartości min/max i bieżąca ===================== */
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] span,
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] p,
    [data-testid="stSidebar"] .stSlider [data-testid="stSliderTickBarMin"],
    [data-testid="stSidebar"] .stSlider [data-testid="stSliderTickBarMax"],
    [data-testid="stSidebar"] .stSlider span {
        font-size: 11px !important;
        color: #6b7280 !important;
    }

    /* === SELECTBOX / DROPDOWN ===================================== */
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div,
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] span,
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] div {
        font-size: 13.5px !important;
        color: #c8cde0 !important;
    }

    /* === RADIO BUTTONS ============================================ */
    [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] .stRadio span {
        font-size: 13.5px !important;
        color: #9ca3af !important;
    }

    /* === CHECKBOXY I TOGGLE ======================================= */
    [data-testid="stSidebar"] .stCheckbox span,
    [data-testid="stSidebar"] .stToggle span,
    [data-testid="stSidebar"] .stCheckbox [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] .stToggle [data-testid="stMarkdownContainer"] p {
        font-size: 13.5px !important;
        color: #9ca3af !important;
    }

    /* === NUMBER INPUT ============================================= */
    [data-testid="stSidebar"] .stNumberInput input {
        font-size: 13.5px !important;
        color: #e2e4f0 !important;
    }

    /* === CAPTION (st.caption) ===================================== */
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] .stCaption {
        font-size: 11px !important;
        color: #6b7280 !important;
    }

    /* === INFO / WARNING / SUCCESS MESSAGES ======================== */
    [data-testid="stSidebar"] [data-testid="stNotification"] p,
    [data-testid="stSidebar"] .stAlert p,
    [data-testid="stSidebar"] [data-testid="stAlertContentInfo"] p,
    [data-testid="stSidebar"] [data-testid="stAlertContentWarning"] p,
    [data-testid="stSidebar"] [data-testid="stAlertContentSuccess"] p,
    [data-testid="stSidebar"] [data-testid="stAlertContentError"] p {
        font-size: 13.5px !important;
    }

    /* === BUTTONY W SIDEBARZE ====================================== */
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stButton > button span {
        font-size: 13.5px !important;
    }

    /* === EXPANDER W SIDEBARZE ==================================== */
    [data-testid="stSidebar"] .stExpander summary p,
    [data-testid="stSidebar"] .stExpander summary span,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
        font-size: 13.5px !important;
        color: #9ca3af !important;
    }

    /* === TYTUŁ SIDEBARA (st.sidebar.title) ======================== */
    [data-testid="stSidebar"] [data-testid="stHeading"] h1,
    [data-testid="stSidebar"] [data-testid="stHeading"] h2 {
        font-size: 15px !important;
        font-weight: 800 !important;
        color: var(--green) !important;
        background: none !important;
        -webkit-text-fill-color: unset !important;
        letter-spacing: 0.5px !important;
    }

    /* Header for options (custom class) */
    .sidebar-options-header {
        font-size: 10px !important;
        font-weight: 700 !important;
        color: var(--green) !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        margin-bottom: 8px !important;
        opacity: 0.8;
    }

    /* === LINKI NAWIGACYJNE ======================================== */
    [data-testid="stSidebarNavLink"] {
        border-radius: 8px !important;
        margin: 2px 8px !important;
        padding: 8px 14px !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        color: #e2e4f0 !important; /* Brighter color for better readability */
        transition: background 0.2s ease, color 0.2s ease !important;
    }
    [data-testid="stSidebarNavLink"]:hover {
        background: rgba(0,230,118,0.12) !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebarNavLink"][aria-selected="true"] {
        background: rgba(0,230,118,0.18) !important;
        color: #00e676 !important;
        border-left: 4px solid #00e676 !important;
        font-weight: 700 !important;
    }
    /* Separatory sekcji nawigacji (Nagłówki Grup) */
    [data-testid="stSidebarNavSeparator"],
    [data-testid="stSidebarNavItems"] div[data-testid="stCaptionContainer"] {
        color: var(--green) !important;
        font-size: 11px !important;
        font-weight: 800 !important;
        letter-spacing: 1.2px !important;
        text-transform: uppercase !important;
        padding: 16px 14px 6px 14px !important;
        opacity: 0.9 !important;
    }
    /* Tekst wewnątrz linków nawigacyjnych */
    [data-testid="stSidebarNavLink"] p,
    [data-testid="stSidebarNavLink"] span {
        font-size: 13.5px !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       PILLS STYLING — Premium look for st.pills
    ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stPills"] button {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #9ca3af !important;
        border-radius: 20px !important;
        padding: 4px 16px !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stPills"] button:hover {
        border-color: var(--green) !important;
        color: white !important;
        background-color: rgba(0, 230, 118, 0.05) !important;
    }
    div[data-testid="stPills"] button[aria-checked="true"] {
        background-color: var(--green) !important;
        color: #05060d !important;
        border-color: var(--green) !important;
        font-weight: 700 !important;
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.3) !important;
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

    /* ═══════════════════════════════════════════════════════════════
       PHASE 1 UX — P3-F: WCAG Kontrast (text-dim #9da8b8 → 5.1:1)
    ═══════════════════════════════════════════════════════════════ */
    :root {
        --text-dim: #9da8b8;  /* podwyższony kontrast z 8b95a5 → 5.1:1 WCAG AA */
    }

    /* ═══════════════════════════════════════════════════════════════
       PHASE 1 UX — P3-G: Focus visible (dostępność klawiatury)
    ═══════════════════════════════════════════════════════════════ */
    .stButton > button:focus-visible,
    [data-testid="stSidebarNavLink"]:focus-visible,
    div[data-testid="stPills"] button:focus-visible,
    div[data-testid="stExpander"] summary:focus-visible {
        outline: 2px solid var(--green) !important;
        outline-offset: 2px !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       PHASE 1 UX — P3-D: Button ripple / active press effect
    ═══════════════════════════════════════════════════════════════ */
    .stButton > button:active {
        transform: translateY(0px) scale(0.97) !important;
        box-shadow: 0 0 12px rgba(0, 230, 118, 0.18) !important;
        transition: all 0.07s ease !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       PHASE 1 UX — P2-F: Expander — szerszy obszar kliknięcia
    ═══════════════════════════════════════════════════════════════ */
    div[data-testid="stExpander"] summary {
        padding: 11px 16px !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        box-sizing: border-box !important;
        user-select: none !important;
        cursor: pointer !important;
    }
    div[data-testid="stExpander"] summary:hover {
        background: rgba(0, 204, 255, 0.05) !important;
        border-radius: 8px !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       PHASE 1 UX — P2-E: Zakładki st.tabs — animacja fade-in
    ═══════════════════════════════════════════════════════════════ */
    @keyframes tabFadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    div[data-testid="stTabContent"] {
        animation: tabFadeIn 0.22s ease-out !important;
    }
    /* Aktywna zakładka — podkreślenie neonowe */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--green) !important;
        border-bottom-color: var(--green) !important;
    }
    button[data-baseweb="tab"] {
        font-family: var(--font) !important;
        font-size: 13px !important;
        transition: color 0.15s ease, border-bottom-color 0.15s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #e2e4f0 !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       PHASE 1 UX — P2-D: st.dataframe — cyberpunk theme
    ═══════════════════════════════════════════════════════════════ */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    [data-testid="stDataFrame"] th {
        background: rgba(0, 230, 118, 0.07) !important;
        color: var(--green) !important;
        font-family: var(--font) !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        letter-spacing: 0.4px !important;
        border-bottom: 1px solid rgba(0, 230, 118, 0.15) !important;
        padding: 8px 12px !important;
    }
    [data-testid="stDataFrame"] td {
        font-family: var(--mono) !important;
        font-size: 12px !important;
        color: var(--text) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
        padding: 7px 12px !important;
        font-variant-numeric: tabular-nums !important;
    }
    [data-testid="stDataFrame"] tr:hover td {
        background: rgba(0, 230, 118, 0.04) !important;
    }
    [data-testid="stDataFrame"] tr:nth-child(even) td {
        background: rgba(255, 255, 255, 0.015) !important;
    }
    /* Scrollbar wewnątrz dataframe */
    [data-testid="stDataFrame"] ::-webkit-scrollbar { height: 4px; }
    [data-testid="stDataFrame"] ::-webkit-scrollbar-thumb {
        background: rgba(0, 230, 118, 0.2);
        border-radius: 4px;
    }

    /* ═══════════════════════════════════════════════════════════════
       PHASE 1 UX — P1-B: Sidebar max-width (13" laptopy)
    ═══════════════════════════════════════════════════════════════ */
    [data-testid="stSidebar"] > div:first-child {
        min-width: 220px !important;
        max-width: 268px !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       PHASE 1 UX — P1-A: Media Queries — Responsywność mobilna
    ═══════════════════════════════════════════════════════════════ */
    @media (max-width: 1280px) {
        div.block-container {
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }
        /* Metryki sparkline — mniejsza czcionka na ściśniętych kolumnach */
        .metric-value { font-size: 18px !important; }
    }

    @media (max-width: 1024px) {
        div.block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        /* 5-kolumnowy pillar grid → wrap do 2-3 kolumn */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.5rem !important;
        }
        [data-testid="stColumn"] {
            min-width: 30% !important;
            flex: 1 1 30% !important;
        }
        /* Ticker bar — mniejsza czcionka */
        .ticker-bar { font-size: 11px !important; }
        .ticker-item { margin: 0 14px !important; }
    }

    @media (max-width: 768px) {
        div.block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            padding-top: 1rem !important;
        }
        [data-testid="stColumn"] {
            min-width: 45% !important;
            flex: 1 1 45% !important;
        }
        /* Nagłówki modułów — mniejszy rozmiar */
        .module-title { font-size: 1.4rem !important; }
        h1 { font-size: 1.8rem !important; }
        /* Ticker bar — ukryj na bardzo małych ekranach */
        .ticker-bar { display: none !important; }
    }

    @media (max-width: 480px) {
        [data-testid="stColumn"] {
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .module-title { font-size: 1.2rem !important; }
        div.block-container {
            padding-left: 0.25rem !important;
            padding-right: 0.25rem !important;
        }
    }

    /* ═══════════════════════════════════════════════════════════════
       PHASE 1 UX — Skeleton Loader dla pustych wartości N/A
    ═══════════════════════════════════════════════════════════════ */
    @keyframes shimmer {
        0%   { background-position: -200% 0; }
        100% { background-position:  200% 0; }
    }
    .skeleton {
        background: linear-gradient(
            90deg,
            rgba(255,255,255,0.03) 25%,
            rgba(255,255,255,0.08) 50%,
            rgba(255,255,255,0.03) 75%
        ) !important;
        background-size: 200% 100% !important;
        animation: shimmer 1.8s infinite !important;
        border-radius: 4px !important;
        color: transparent !important;
        user-select: none !important;
    }

    /* ═══════════════════════════════════════════════════════════════
       PHASE 1 UX — Stale Data Badge
    ═══════════════════════════════════════════════════════════════ */
    .stale-badge {
        display: inline-block;
        font-size: 9px;
        color: #f39c12;
        border: 1px solid rgba(243, 156, 18, 0.3);
        border-radius: 4px;
        padding: 1px 6px;
        margin-left: 6px;
        vertical-align: middle;
        font-family: var(--mono);
        letter-spacing: 0.5px;
        animation: blinkBg 3s infinite;
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
        <div class='module-breadcrumb'>
            <a href='/' target='_self'
               style='color:var(--green);text-decoration:none;opacity:0.85;
                      transition:opacity 0.15s ease;'
               onmouseover="this.style.opacity='1'"
               onmouseout="this.style.opacity='0.85'">🏠 Dashboard</a>
            <span style='margin:0 6px;opacity:0.4;'>›</span>
            <span style='opacity:0.7;'>{title}</span>
        </div>
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

    # Czy wartość jest niedostępna — zastosuj skeleton
    is_na = str(value).strip().upper() in ("N/A", "NONE", "", "--", "—")
    val_style = (
        f"font-family:'JetBrains Mono',monospace;font-size:20px;"
        f"font-weight:700;color:{accent_color};font-variant-numeric:tabular-nums;"
        f"line-height:1.1;"
    )
    val_html = (
        f"<div class='skeleton' style='height:24px;width:60%;margin:4px 0;border-radius:4px;'>&nbsp;</div>"
        if is_na else
        f"<div style='{val_style}'>{value}"
        f"<span style='font-size:11px;color:#9ca3af;font-weight:400;margin-left:2px;'>"
        f"{suffix}</span></div>"
        f"{d_html}{spark}"
    )

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
        f'{val_html}'
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

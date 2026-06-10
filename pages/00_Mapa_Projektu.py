import streamlit as st

st.markdown("""
    <style>
        /* Zerowe paddingi pionowe — graf wypelnia cala dostepna wysokosc */
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0 !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        /* Ukryj toolbary i stopki Streamlit */
        footer, .stDeployButton { display: none !important; }
        /* Graf (iframe agraph) wypelnia reszte wysokosci */
        iframe[title="streamlit_agraph.agraph"] {
            width: 100% !important;
            display: block !important;
            border: none !important;
        }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 1. STRUKTURA MENU — ZGODNA Z module_xxx.py
# ==========================================
# Każda sekcja ma:
#   "hub": plik module_xxx.py (Level 2) — lub None dla Centrum Dowodzenia
#   "subpages": lista podstron widocznych w danym module (Level 3)

from modules.module_registry import build_menu_structure

# ── AUTO-GENERATED: module_registry — DO NOT EDIT BELOW ──────────────
MENU_STRUCTURE = build_menu_structure()
# ── END AUTO-GENERATED ───────────────────────────────────────────────

# ==========================================
# 2. BUDOWA GRAFU
# ==========================================
from streamlit_agraph import agraph, Node, Edge, Config
import colorsys


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def lighten_color(hex_color, factor=0.65):
    try:
        r, g, b = hex_to_rgb(hex_color)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        s = max(0, s * (1 - factor))
        v = min(1.0, v + (1.0 - v) * factor)
        return rgb_to_hex(*colorsys.hsv_to_rgb(h, s, v))
    except Exception:
        return "#e8eaf6"


def dark_text_for(hex_color):
    try:
        r, g, b = hex_to_rgb(hex_color)
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "#111111" if lum > 0.4 else "#ffffff"
    except Exception:
        return "#111111"


nodes = []
edges = []
added_ids = set()

# ROOT (Level 0)
nodes.append(Node(
    id="root",
    label="System Barbell",
    size=60,
    color="#0d0f1e",
    font={"color": "#ffffff", "size": 28, "face": "Inter", "bold": True},
    shape="box",
    borderWidth=3,
))
added_ids.add("root")

for section_name, section_data in MENU_STRUCTURE.items():
    section_id = section_name
    clean_label = section_name.split(" ", 1)[-1].strip() if " " in section_name else section_name

    bg_color, _ = section_data["color"]
    text_color = dark_text_for(bg_color)

    hub_bg   = lighten_color(bg_color, factor=0.40)
    hub_text = dark_text_for(hub_bg)
    sub_bg   = lighten_color(bg_color, factor=0.70)
    sub_text = dark_text_for(sub_bg)
    ec_sec   = "#555555"
    ec_hub   = lighten_color(bg_color, factor=0.20)
    ec_sub   = lighten_color(bg_color, factor=0.40)

    # SEKCJA (Level 1)
    nodes.append(Node(
        id=section_id,
        label=clean_label,
        size=45,
        color=bg_color,
        font={"color": text_color, "size": 20, "face": "Inter", "bold": True},
        shape="box",
        borderWidth=2,
    ))
    added_ids.add(section_id)
    edges.append(Edge(source="root", target=section_id, color=ec_sec, physics=False))

    hub = section_data.get("hub")

    if hub:
        # HUB (Level 2) — plik module_xxx.py
        hub_id = hub["page_file"]
        if hub_id not in added_ids:
            nodes.append(Node(
                id=hub_id,
                label=hub["title"],
                size=38,
                color=hub_bg,
                font={"color": hub_text, "size": 17, "face": "Inter", "bold": True},
                shape="box",
                borderWidth=2,
            ))
            added_ids.add(hub_id)
        edges.append(Edge(source=section_id, target=hub_id, color=ec_hub, physics=False))
        parent_id = hub_id
    else:
        parent_id = section_id

    # PODSTRONY (Level 3 lub Level 2 dla sekcji bez huba)
    seen_sub = set()
    for sp in section_data.get("subpages", []):
        sp_id = sp["page_file"]
        if sp_id in seen_sub:
            continue
        seen_sub.add(sp_id)

        if sp_id not in added_ids:
            nodes.append(Node(
                id=sp_id,
                label=sp["title"],
                size=28,
                color=sub_bg,
                font={"color": sub_text, "size": 14, "face": "Inter"},
                shape="box",
                borderWidth=1,
            ))
            added_ids.add(sp_id)
        edges.append(Edge(source=parent_id, target=sp_id, color=ec_sub, physics=False))

# ==========================================
# 3. KONFIGURACJA I RENDER
# ==========================================
config = Config(
    width="100%",
    height=1200,   # Wewnetrzna wielkosc canvas dla vis.js (musi byc int)
    directed=True,
    physics=False,
    hierarchical=True,
    layout={
        "hierarchical": {
            "enabled": True,
            "direction": "LR",
            "sortMethod": "directed",
            "levelSeparation": 270,
            "nodeSpacing": 42,
            "treeSpacing": 60,
        }
    },
    interaction={
        "hover": True,
        "zoomView": True,
        "dragView": True,
        "navigationButtons": False,  # wlasne przyciski ukryte bo jestesmy fullscreen
    }
)

clicked_node = agraph(nodes=nodes, edges=edges, config=config)

# ==========================================
# 4. NAWIGACJA PO KLIKNIĘCIU
# ==========================================
if clicked_node:
    if clicked_node == "root" or clicked_node in MENU_STRUCTURE:
        pass
    elif clicked_node and clicked_node.endswith(".py"):
        if clicked_node == "app.py":
            st.switch_page("app.py")
        else:
            st.switch_page(clicked_node)

# ==========================================
# 5. JS: DOSTOSUJ WYSOKOSC GRAFU DO EKRANU
# ==========================================
import streamlit.components.v1 as components
components.html("""
<script>
(function() {
    var doc = window.parent.document;

    function resizeGraph() {
        var iframes = doc.querySelectorAll('iframe[title="streamlit_agraph.agraph"]');
        if (!iframes.length) return;
        
        // Czyste obliczenie dostepnego miejsca bez paska nawigacji glownej
        var header = doc.querySelector('header[data-testid="stHeader"]');
        var headerH = header ? header.offsetHeight : 0;
        var availH = window.parent.innerHeight - headerH - 8; // mały zapas
        
        iframes.forEach(function(f) {
            f.style.height = availH + 'px';
            f.style.minHeight = availH + 'px';
            f.style.border = 'none';
            f.style.overflow = 'hidden';
            
            // Opcjonalnie wejdz wgłąb (jakby streamlit wrapper przeszkadzał)
            var parentDiv = f.closest('.element-container');
            if(parentDiv) {
                parentDiv.style.height = availH + 'px';
                parentDiv.style.overflow = 'hidden';
            }
        });
    }

    resizeGraph();
    setTimeout(resizeGraph, 300);
    setTimeout(resizeGraph, 800);
    window.parent.addEventListener('resize', resizeGraph);
}());
</script>
""", height=0)

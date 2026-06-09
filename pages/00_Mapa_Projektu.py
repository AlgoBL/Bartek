import streamlit as st

st.markdown("""
    <style>
        .block-container {
            padding-top: 0rem !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 100% !important;
        }
        h1 {
            margin-top: -1.5rem !important;
            padding-top: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🗺️ Mapa Projektu (Drzewo Architektury)")
st.markdown("Graficzna wizualizacja całej architektury projektu Barbell. **Kliknij dowolny kafelek, aby do niego przejść.**")

# ==========================================
# 1. STRUKTURA MENU — ZGODNA Z module_xxx.py
# ==========================================
# Każda sekcja ma:
#   "hub": plik module_xxx.py (Level 2) — lub None dla Centrum Dowodzenia
#   "subpages": lista podstron widocznych w danym module (Level 3)

MENU_STRUCTURE = {
    "🌐 Centrum Dowodzenia": {
        "color": ("#2979ff", "#fff"),
        "hub": None,
        "subpages": [
            {"title": "Control Center",     "page_file": "app.py"},
            {"title": "Globalne Ustawienia","page_file": "pages/0_Globalne_Ustawienia.py"},
            {"title": "Doradca AI",         "page_file": "pages/28_Doradca.py"},
            {"title": "Mapa Projektu",      "page_file": "pages/00_Mapa_Projektu.py"},
        ]
    },
    "📉 Środowisko Makro i Reżimy": {
        "color": ("#ff9100", "#000"),
        "hub": {"title": "Makro i Reżimy", "page_file": "pages/module_makro.py"},
        "subpages": [
            {"title": "Skaner Rynku",        "page_file": "pages/2_Skaner.py"},
            {"title": "Zegar Inwestycyjny",  "page_file": "pages/11_Regime_Clock.py"},
            {"title": "Alokacja Reżimowa",   "page_file": "pages/12_Regime_Allocation.py"},
            {"title": "Recession Nowcasting","page_file": "pages/26_Recession_Nowcasting.py"},
        ]
    },
    "⚖️ Zarządzanie Portfelem": {
        "color": ("#ffea00", "#000"),
        "hub": {"title": "Zarządzanie Portfelem", "page_file": "pages/module_portfel.py"},
        "subpages": [
            {"title": "Portfolio Health Monitor","page_file": "pages/8_Health_Monitor.py"},
            {"title": "Smart Rebalancing",       "page_file": "pages/16_Rebalancing.py"},
            {"title": "Tax Optimizer PL",        "page_file": "pages/15_Tax_Optimizer.py"},
            {"title": "Wealth Optimizer",        "page_file": "pages/19_Wealth_Optimizer.py"},
        ]
    },
    "🛡️ Centrum Ryzyka": {
        "color": ("#00e676", "#000"),
        "hub": {"title": "Centrum Ryzyka", "page_file": "pages/module_ryzyko.py"},
        "subpages": [
            {"title": "Stress Test",         "page_file": "pages/3_Stress_Test.py"},
            {"title": "Concentration Risk",  "page_file": "pages/9_Concentration_Risk.py"},
            {"title": "Liquidity Risk",      "page_file": "pages/13_Liquidity_Risk.py"},
            {"title": "EVT — Tail Risk",     "page_file": "pages/5_EVT_Analysis.py"},
            {"title": "Systemic Risk",       "page_file": "pages/27_Systemic_Risk.py"},
            {"title": "Drawdown Recovery",   "page_file": "pages/10_Drawdown_Recovery.py"},
            {"title": "Tail Risk Hedging",   "page_file": "pages/14_Tail_Hedging.py"},
            {"title": "Inzynieria Opcji",    "page_file": "pages/32_Inzynieria_Opcji.py"},
        ]
    },
    "🧬 Laboratorium Quant i AI": {
        "color": ("#d500f9", "#fff"),
        "hub": {"title": "Laboratorium Quant", "page_file": "pages/module_quant.py"},
        "subpages": [
            {"title": "Black-Litterman AI",  "page_file": "pages/6_BL_Dashboard.py"},
            {"title": "HERC Portfolio",      "page_file": "pages/24_HERC_Portfolio.py"},
            {"title": "DCC — Korelacje",     "page_file": "pages/7_DCC_Dashboard.py"},
            {"title": "Factor Zoo & PCA",    "page_file": "pages/22_Factor_Analysis.py"},
            {"title": "Sieci Przyczynowe",   "page_file": "pages/33_Sieci_Przyczynowe.py"},
            {"title": "Bledy Stochastyczne", "page_file": "pages/48_Stochastic_Errors.py"},
        ]
    },
    "💰 Planowanie Majatku (FIRE)": {
        "color": ("#ff1744", "#fff"),
        "hub": {"title": "Planowanie Majatku", "page_file": "pages/module_majatku.py"},
        "subpages": [
            {"title": "Emerytura / FIRE",    "page_file": "pages/4_Emerytura.py"},
            {"title": "Decumulation / SWR",  "page_file": "pages/4_Emerytura.py"},
        ]
    },
    "♟️ Meta-Decyzje i Teoria": {
        "color": ("#00e5ff", "#000"),
        "hub": {"title": "Meta-Decyzje", "page_file": "pages/module_meta.py"},
        "subpages": [
            {"title": "Przewaga Informacyjna","page_file": "pages/34_Przewaga_Informacyjna.py"},
            {"title": "Kalkulator Bayesa",    "page_file": "pages/29_Kalkulator_Bayesa.py"},
            {"title": "Asymetria Informacji", "page_file": "pages/31_Asymetria_Informacji.py"},
            {"title": "Teoria Gier",          "page_file": "pages/30_Teoria_Gier.py"},
            {"title": "Chaos Deterministyczny","page_file": "pages/49_Chaos_Deterministyczny.py"},
        ]
    },
    "🎯 Moduly Aktywne i Trening": {
        "color": ("#ff6d00", "#fff"),
        "hub": {"title": "Moduly Aktywne", "page_file": "pages/module_aktywne.py"},
        "subpages": [
            {"title": "Symulator Barbell",   "page_file": "pages/1_Symulator.py"},
            {"title": "Day Trading",         "page_file": "pages/21_Day_Trading.py"},
            {"title": "Sentiment & Flow",    "page_file": "pages/17_Sentiment_Flow.py"},
            {"title": "Walk-Forward CPCV",   "page_file": "pages/23_Walk_Forward.py"},
            {"title": "Life OS",             "page_file": "pages/20_Life_OS.py"},
        ]
    },
}

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
    height=860,
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
        "navigationButtons": True,
    }
)

clicked_node = agraph(nodes=nodes, edges=edges, config=config)

# ==========================================
# 4. NAWIGACJA PO KLIKNIĘCIU
# ==========================================
if clicked_node:
    if clicked_node == "root" or clicked_node in MENU_STRUCTURE:
        st.info("To jest kategoria menu. Kliknij na węzeł podrzędny, aby tam przejść.")
    elif clicked_node and clicked_node.endswith(".py"):
        if clicked_node == "app.py":
            st.switch_page("app.py")
        else:
            st.switch_page(clicked_node)

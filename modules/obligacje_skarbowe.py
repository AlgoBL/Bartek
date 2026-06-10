"""
obligacje_skarbowe.py — Kalkulator Obligacji Skarbowych
Barbell Strategy Dashboard

Obsługiwane typy (bez ROS/ROD):
  OTS  3 mies.  stałe 2,00%
  ROR  1 rok    zmienne (NBP ref + 0%)
  DOR  2 lata   zmienne (NBP ref + 0,15%)
  TOS  3 lata   stałe 4,40%
  COI  4 lata   inflacja + 1,50%
  EDO  10 lat   inflacja + 2,00%

Logika:
  - Podatek Belki 19% od zysku brutto (BEZ IKE)
  - Rozliczenie rok po roku (kupon na koniec każdego okresu)
  - Sekcja wcześniejszego wykupu z opłatami
  - Suwak inflacji dla obligacji indeksowanych
  - fetch_bond_rates(): live scraping z obligacjeskarbowe.pl, cache 1h, fallback na dane wbudowane
"""

import streamlit as st
import re
import datetime
from config import TAX_BELKA, INFLATION_RATE_PL

_BASE_URL = "https://www.obligacjeskarbowe.pl"
_OFFERS_URL = f"{_BASE_URL}/oferta-obligacji/"
_FETCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "pl-PL,pl;q=0.9",
}
_CACHE_TTL_SECONDS = 3600  # 1 godzina


# ─────────────────────────────────────────────────────────────────────────────
#  DEFINICJA OBLIGACJI
# ─────────────────────────────────────────────────────────────────────────────

BONDS = {
    "OTS": {
        "name": "OTS",
        "label": "3 mies.",
        "rate_first": 0.0200,
        "years": 0.25,
        "periods": 1,          # jedna kapitalizacja po 3 mies.
        "type": "fixed",
        "inflation_margin": None,
        "early_fee_per_unit": 0.50,  # opłata za wcześniejszy wykup (zł/obligację 100 zł)
        "early_fee_note": "0,50 zł / obligację (100 zł)",
        "unit_price": 100,
        "description": "Trzymiesięczne obligacje o stałym oprocentowaniu. Kapitalizacja jednorazowa po 3 miesiącach.",
    },
    "ROR": {
        "name": "ROR",
        "label": "1 rok",
        "rate_first": 0.0400,
        "years": 1,
        "periods": 1,
        "type": "variable",
        "inflation_margin": None,
        "nbp_margin": 0.00,
        "early_fee_per_unit": 0.50,
        "early_fee_note": "0,50 zł / obligację (100 zł) — min. 30 dni od zakupu",
        "unit_price": 100,
        "description": "Roczne obligacje o zmiennym oprocentowaniu. Rok 1 = 4,00% (stałe). Kolejne lata: stopa ref. NBP + 0 pp.",
    },
    "DOR": {
        "name": "DOR",
        "label": "2 lata",
        "rate_first": 0.0415,
        "years": 2,
        "periods": 2,
        "type": "variable",
        "inflation_margin": None,
        "nbp_margin": 0.0015,
        "early_fee_per_unit": 0.70,
        "early_fee_note": "0,70 zł / obligację (100 zł) — po min. 7 dniach od zakupu",
        "unit_price": 100,
        "description": "Dwuletnie obligacje o zmiennym oprocentowaniu. Rok 1 = 4,15%. Rok 2: stopa ref. NBP + 0,15 pp.",
    },
    "TOS": {
        "name": "TOS",
        "label": "3 lata",
        "rate_first": 0.0440,
        "years": 3,
        "periods": 3,
        "type": "fixed",
        "inflation_margin": None,
        "early_fee_per_unit": 0.70,
        "early_fee_note": "0,70 zł / rok posiadania / obligację (100 zł)",
        "unit_price": 100,
        "description": "Trzyletnie obligacje o stałym oprocentowaniu 4,40% p.a. Kupon roczny, reinwestowany.",
    },
    "COI": {
        "name": "COI",
        "label": "4 lata",
        "rate_first": 0.0475,
        "years": 4,
        "periods": 4,
        "type": "inflation",
        "inflation_margin": 0.0150,
        "early_fee_per_unit": 0.70,
        "early_fee_note": "0,70 zł / rok posiadania / obligację (100 zł)",
        "unit_price": 100,
        "description": "Czteroletnie obligacje indeksowane inflacją. Rok 1 = 4,75%. Lata 2-4: CPI + 1,50 pp. marży.",
    },
    "EDO": {
        "name": "EDO",
        "label": "10 lat",
        "rate_first": 0.0535,
        "years": 10,
        "periods": 10,
        "type": "inflation",
        "inflation_margin": 0.0200,
        "early_fee_per_unit": 2.00,
        "early_fee_note": "2,00 zł / rok posiadania / obligację (100 zł)",
        "unit_price": 100,
        "description": "Dziesięcioletnie obligacje indeksowane inflacją. Rok 1 = 5,35%. Lata 2-10: CPI + 2,00 pp. marży. Odsetki kapitalizowane rocznie.",
    },
}

# Aktualna stopa referencyjna NBP (czerwiec 2026)
NBP_REF_RATE = 0.0375

# ─────────────────────────────────────────────────────────────────────────────
#  LIVE FETCH — obligacjeskarbowe.pl
# ─────────────────────────────────────────────────────────────────────────────

def fetch_bond_rates() -> dict:
    """
    Pobiera aktualne oprocentowanie (rok 1) z obligacjeskarbowe.pl.

    Algorytm:
      1. Strona /oferta-obligacji/ → wykrywa aktualne URL-e serii
         (np. /oferta-obligacji/obligacje-3-letnie-tos/tos0629/)
         dzięki czemu automatycznie dostosuje się gdy zmieni się seria.
      2. Z karty na stronie głównej odczytuje stawkę z .product-card__promo-value
      3. Wynik cache'uje w st.session_state na _CACHE_TTL_SECONDS.
      4. Fallback: jeśli scraping się nie powiedzie, zwraca pusty dict
         (BONDS zostają z wartościami domyślnymi).

    Returns:
        dict symbol → rate_first (float, np. 0.044)
        dict może być pusty gdy sieć niedostępna.
    """
    import requests

    # ── Cache check ────────────────────────────────────────────────────────
    now = datetime.datetime.now()
    cached = st.session_state.get("_obl_rates_cache")
    if cached and (now - cached["ts"]).total_seconds() < _CACHE_TTL_SECONDS:
        return cached["rates"]

    rates: dict = {}
    fetch_time = now.strftime("%H:%M")

    try:
        # ── Krok 1: strona główna oferty → linki do serii ──────────────────
        r = requests.get(_OFFERS_URL, timeout=8, headers=_FETCH_HEADERS)
        r.raise_for_status()
        html = r.text

        # Wyciągamy linki np. /oferta-obligacji/obligacje-3-letnie-tos/tos0629/
        link_pattern = re.compile(
            r'href="(/oferta-obligacji/[^"]+/((?:ots|ror|dor|tos|coi|edo)\d+)/)",?',
            re.IGNORECASE,
        )
        series_urls = {}  # symbol → pełny URL
        for path, series_code in link_pattern.findall(html):
            symbol = series_code[:3].upper()
            if symbol in BONDS and symbol not in series_urls:
                series_urls[symbol] = f"{_BASE_URL}{path}"

        # ── Krok 2: stawka z HTML karty na stronie głównej ────────────────
        # Wzorzec: <span class="product-card__promo-value">...<span ...>4,75<sub>%</sub>...
        # Lepszy regex: wartość tuż przed <sub>%</sub>
        promo_pattern = re.compile(
            r'product-card__promo-value.*?(\d+),(\d+)\s*<sub>%</sub>',
            re.DOTALL,
        )
        # Parsujemy cały HTML strony głównej — każda karta ma jedną stawkę
        # Kolejność kart NIE jest gwarantowana, więc matchujemy po tytule obligacji
        card_pattern = re.compile(
            r'product-card[^>]*>.*?product-card__promo-value[^>]*>.*?(\d+),(\d+)\s*<sub>%</sub>'
            r'.*?product-card__title[^>]*>([^<]+)<',
            re.DOTALL,
        )
        for m in card_pattern.finditer(html):
            int_part, dec_part, title = m.group(1), m.group(2), m.group(3).strip()
            rate = float(f"{int_part}.{dec_part}") / 100
            # Mapuj tytuł → symbol
            for sym in BONDS:
                if sym.lower() in title.lower():
                    rates[sym] = rate
                    break

        # ── Krok 3 (backup): jeśli karta nie dała wyniku, pobierz osobną stronę ─
        missing = [s for s in BONDS if s not in rates]
        for symbol in missing:
            url = series_urls.get(symbol)
            if not url:
                continue
            try:
                rp = requests.get(url, timeout=6, headers=_FETCH_HEADERS)
                m2 = re.search(r'(\d+),(\d+)\s*%', rp.text)
                if m2:
                    rates[symbol] = float(f"{m2.group(1)}.{m2.group(2)}") / 100
            except Exception:
                pass

    except Exception:
        pass  # Sieć niedostępna — rates zostaje pusty, użyjemy fallbacku

    # ── Zapisz cache ───────────────────────────────────────────────────────
    st.session_state["_obl_rates_cache"] = {
        "rates": rates,
        "ts":    now,
        "fetch_time": fetch_time,
        "source": "live" if rates else "fallback",
    }
    return rates


def _apply_live_rates(rates: dict):
    """Nakłada pobrane stawki na globalny słownik BONDS (in-place)."""
    for symbol, rate in rates.items():
        if symbol in BONDS:
            BONDS[symbol]["rate_first"] = rate



def _get_annual_rate(bond: dict, year: int, inflation: float) -> float:
    """Zwraca oprocentowanie dla danego roku (1-indexed)."""
    if year == 1:
        return bond["rate_first"]
    btype = bond["type"]
    if btype == "fixed":
        return bond["rate_first"]
    elif btype == "variable":
        return NBP_REF_RATE + bond.get("nbp_margin", 0.0)
    elif btype == "inflation":
        return inflation + bond["inflation_margin"]
    return bond["rate_first"]


def calculate_bond_returns(kwota: float, bond: dict, inflation: float) -> dict:
    """
    Oblicza pełne wyniki dla wybranej obligacji.

    Returns:
      dict z kluczami:
        brutto, belka, netto, lacznie,
        lacznie_pct, cagr_netto,
        yearly_breakdown (list of dicts),
        early_redemption (list of dicts),
    """
    years    = bond["years"]
    btype    = bond["type"]
    periods  = bond["periods"]

    # ── OTS — specjalny przypadek (3 mies.) ──────────────────────────────────
    if bond["name"] == "OTS":
        brutto_jednostkowy = kwota * bond["rate_first"] * 0.25
        belka   = brutto_jednostkowy * TAX_BELKA
        netto   = brutto_jednostkowy - belka
        lacznie = kwota + netto
        return {
            "brutto":       round(brutto_jednostkowy, 2),
            "belka":        round(-belka, 2),
            "netto":        round(netto, 2),
            "lacznie":      round(lacznie, 2),
            "lacznie_pct":  round(netto / kwota * 100, 4),
            "cagr_netto":   round(((lacznie / kwota) ** (1 / 0.25) - 1) * 100, 4),
            "years":        0.25,
            "yearly_breakdown": [
                {
                    "okres":       "3 miesiące",
                    "rate_pct":    round(bond["rate_first"] * 100, 2),
                    "wartosc":     round(lacznie, 2),
                    "kupon_brutto": round(brutto_jednostkowy, 2),
                    "kupon_netto": round(netto, 2),
                    "narost":      round(netto, 2),
                }
            ],
            "early_redemption": _calc_early(kwota, bond, inflation),
        }

    # ── Obligacje wieloletnie — procent składany (kapitalizacja roczna) ──────────
    # Wszystkie obligacje wieloletnie kapitalizują odsetki rocznie do kapitału.
    # To odpowiada reinwestycji kuponu — identyczne z kalkulatorem na screenshocie.
    # Przykład TOS: 10000 × 1.044^3 − 10000 = 1381.13 zł brutto
    balance      = kwota
    total_brutto = 0.0
    yearly       = []

    for y in range(1, periods + 1):
        rate_y       = _get_annual_rate(bond, y, inflation)
        kupon_brutto = balance * rate_y       # kupon od bieżącego salda (procent składany)
        balance     += kupon_brutto           # reinwestycja — odsetki wchodzą do kapitału
        total_brutto += kupon_brutto

        yearly.append({
            "rok":            y,
            "rate_pct":       round(rate_y * 100, 2),
            "kupon_brutto":   round(kupon_brutto, 2),
            "narost":         round(total_brutto, 2),
            "wartosc_brutto": round(balance, 2),
        })

    brutto  = total_brutto
    belka   = brutto * TAX_BELKA
    netto   = brutto - belka
    lacznie = kwota + netto

    # CAGR
    cagr = ((lacznie / kwota) ** (1.0 / float(years)) - 1) * 100

    # Wzbogacamy yearly o wartości netto (podatek pobierany dopiero na końcu)
    for row in yearly:
        row["narost_netto"]  = round(row["narost"] * (1 - TAX_BELKA), 2)
        row["wartosc_netto"] = round(kwota + row["narost_netto"], 2)

    return {
        "brutto":       round(brutto, 2),
        "belka":        round(-belka, 2),
        "netto":        round(netto, 2),
        "lacznie":      round(lacznie, 2),
        "lacznie_pct":  round(netto / kwota * 100, 4),
        "cagr_netto":   round(cagr, 4),
        "years":        years,
        "yearly_breakdown": yearly,
        "early_redemption": _calc_early(kwota, bond, inflation, yearly),
    }


def _calc_early(kwota: float, bond: dict, inflation: float, yearly_breakdown: list = None) -> list:
    """
    Oblicza kwotę zwrotu przy wcześniejszym wykupie po każdym pełnym roku.
    Używa rzeczywistych skumulowanych odsetek (brutto) z yearly_breakdown.
    Opłata za wcześniejszy wykup odliczana od narosłych odsetek.
    """
    n_units = kwota / bond["unit_price"]
    years   = bond["years"]
    result  = []

    if bond["name"] == "OTS":
        # OTS: stały koszt 0,50 zł / obligację (niezależnie od terminu)
        fee        = n_units * bond["early_fee_per_unit"]
        kupon_3m   = kwota * bond["rate_first"] * 0.25
        narost     = max(0, kupon_3m - fee)
        belka_e    = narost * TAX_BELKA
        netto_e    = narost - belka_e
        result.append({
            "moment":   "Po 6 tygodniach",
            "fee":      round(fee, 2),
            "netto":    round(netto_e, 2),
            "lacznie":  round(kwota + netto_e, 2),
        })
        return result

    # Pobieramy skumulowane odsetki brutto po każdym roku z breakdown
    narost_per_year = {}
    if yearly_breakdown:
        for row in yearly_breakdown:
            narost_per_year[row["rok"]] = row["narost"]

    for y in range(1, int(years)):
        # Rzeczywisty narost brutto po y latach (z compound interest)
        narost_y = narost_per_year.get(y, kwota * _get_annual_rate(bond, 1, inflation) * y)

        # Opłata: fee_per_unit × n_units × lata posiadania (dla TOS/COI/EDO)
        if bond["name"] in ("TOS", "COI", "EDO"):
            fee = n_units * bond["early_fee_per_unit"] * y
        else:  # ROR, DOR: stały ryczałt za wykup
            fee = n_units * bond["early_fee_per_unit"]

        narost_after_fee = max(0, narost_y - fee)
        belka_e = narost_after_fee * TAX_BELKA
        netto_e = narost_after_fee - belka_e

        result.append({
            "moment":   f"Po {y} {'roku' if y == 1 else 'latach'}",
            "fee":      round(fee, 2),
            "netto":    round(netto_e, 2),
            "lacznie":  round(kwota + netto_e, 2),
        })

    return result



# ─────────────────────────────────────────────────────────────────────────────
#  STYLE CSS — wstrzykiwane lokalnie dla modułu
# ─────────────────────────────────────────────────────────────────────────────

def _inject_obligacje_css():
    st.markdown("""
    <style>
    /* ── SELEKTOR OBLIGACJI ─────────────────────────────────── */
    .obl-selector-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 18px;
    }
    .obl-card {
        background: #111827;
        border: 1.5px solid #1f2937;
        border-radius: 10px;
        padding: 12px 18px;
        cursor: pointer;
        min-width: 90px;
        text-align: center;
        transition: all 0.18s ease;
        position: relative;
    }
    .obl-card:hover {
        border-color: #374151;
        background: #1a2233;
    }
    .obl-card.active {
        border-color: #00e676;
        background: rgba(0, 230, 118, 0.07);
        box-shadow: 0 0 18px rgba(0,230,118,0.15);
    }
    .obl-card .obl-symbol {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #9ca3af;
        text-transform: uppercase;
    }
    .obl-card.active .obl-symbol { color: #6ee7b7; }
    .obl-card .obl-rate {
        font-size: 22px;
        font-weight: 800;
        color: #e2e8f0;
        line-height: 1.2;
        margin: 4px 0;
        font-family: 'Inter', monospace;
    }
    .obl-card.active .obl-rate { color: #00e676; }
    .obl-card .obl-period {
        font-size: 10px;
        color: #6b7280;
    }

    /* ── PARAMETRY PANEL ────────────────────────────────────── */
    .param-panel {
        background: linear-gradient(145deg, #0f1422, #111827);
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 20px 18px;
        height: 100%;
    }
    .param-label {
        font-size: 10px;
        letter-spacing: 1.5px;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .param-value-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }
    .param-value {
        font-size: 13px;
        font-weight: 600;
        color: #00e676;
        font-family: monospace;
    }
    .quick-btns {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 10px 0;
    }

    /* ── WYNIK PANEL ─────────────────────────────────────────── */
    .wynik-panel {
        background: linear-gradient(145deg, #0f1422, #111827);
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 24px 22px;
        height: 100%;
    }
    .wynik-header {
        font-size: 10px;
        letter-spacing: 2px;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .wynik-label-small {
        font-size: 11px;
        color: #6b7280;
        margin-bottom: 6px;
    }
    .wynik-big-number {
        font-size: 56px;
        font-weight: 900;
        color: #00e676;
        line-height: 1;
        font-family: 'Inter', monospace;
        text-shadow: 0 0 30px rgba(0,230,118,0.35);
        margin-bottom: 4px;
        letter-spacing: -2px;
    }
    .wynik-subtitle {
        font-size: 12px;
        color: #6b7280;
        margin-bottom: 18px;
    }
    .wynik-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 14px;
        margin: 14px 0;
        padding-top: 14px;
        border-top: 1px solid #1f2937;
    }
    .wynik-cell {
        text-align: center;
    }
    .wynik-cell-label {
        font-size: 9px;
        letter-spacing: 1.2px;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .wynik-cell-value {
        font-size: 20px;
        font-weight: 700;
        font-family: monospace;
    }
    .val-brutto   { color: #60a5fa; }
    .val-belka    { color: #f87171; }
    .val-netto    { color: #00e676; }
    .wynik-bar {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid #1f2937;
    }
    .wynik-bar-cell-label {
        font-size: 9px;
        letter-spacing: 1.2px;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .wynik-bar-cell-value {
        font-size: 16px;
        font-weight: 800;
        color: #facc15;
        font-family: monospace;
    }

    /* ── ACCORDION ───────────────────────────────────────────── */
    .obl-accordion-header {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 14px 18px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 13px;
        color: #d1d5db;
        font-weight: 600;
        margin-top: 10px;
        transition: background 0.15s;
    }
    .obl-accordion-header:hover { background: #1a2233; }

    /* ── BOND TAG ─────────────────────────────────────────────── */
    .bond-tag {
        display: inline-block;
        background: rgba(0,230,118,0.12);
        border: 1px solid rgba(0,230,118,0.3);
        color: #00e676;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }
    /* ── TABELA BREAKDOWN ─────────────────────────────────────── */
    .obl-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 12px;
    }
    .obl-table th {
        text-align: left;
        color: #6b7280;
        font-size: 10px;
        letter-spacing: 1px;
        text-transform: uppercase;
        padding: 6px 10px;
        border-bottom: 1px solid #1f2937;
    }
    .obl-table td {
        padding: 8px 10px;
        border-bottom: 1px solid #111827;
        color: #e2e8f0;
    }
    .obl-table tr:last-child td { border-bottom: none; font-weight: 700; }
    .obl-table .td-green { color: #00e676; }
    .obl-table .td-red   { color: #f87171; }
    .obl-table .td-blue  { color: #60a5fa; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  KOMPONENTY UI
# ─────────────────────────────────────────────────────────────────────────────

def _bond_selector(selected_key: str) -> str:
    """Renderuje selektor obligacji jako kwadratowe karty (bez st.button — przez radio hidden)."""
    cols = st.columns(len(BONDS))
    new_key = selected_key

    for i, (key, bond) in enumerate(BONDS.items()):
        is_active = (key == selected_key)
        rate_str = f"{bond['rate_first']*100:.2f}%".replace(".", ",")

        with cols[i]:
            clicked = st.button(
                f"**{bond['name']}**\n{rate_str}\n{bond['label']}",
                key=f"bond_btn_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            )
            if clicked and key != selected_key:
                st.session_state["obl_selected"] = key
                st.rerun()

    return selected_key


def _param_panel(kwota: float, bond: dict, inflation: float):
    """Renderuje lewy panel parametrów i zwraca (kwota, inflation)."""
    st.markdown('<div class="param-label">PARAMETRY</div>', unsafe_allow_html=True)

    # ── KWOTA ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="param-value-row">'
        f'<div class="param-label" style="margin:0">Kwota inwestycji</div>'
        f'<div class="param-value">{int(kwota):,} zł</div>'
        f'</div>'.replace(",", " "),
        unsafe_allow_html=True,
    )

    kwota_new = st.slider(
        "Kwota",
        min_value=500,
        max_value=500_000,
        value=int(kwota),
        step=500,
        label_visibility="collapsed",
    )

    # Szybkie przyciski kwoty
    quick_cols = st.columns(5)
    quick_vals = [1_000, 5_000, 10_000, 50_000, 100_000]
    for ci, qv in enumerate(quick_vals):
        with quick_cols[ci]:
            lbl = f"{qv//1000}k" if qv < 100_000 else "100k"
            if st.button(lbl, key=f"obl_quick_{qv}", use_container_width=True):
                st.session_state["obl_kwota"] = qv
                st.rerun()

    st.markdown("---")

    # ── INFLACJA (tylko dla COI/EDO) ─────────────────────────────────────────
    inf_new = inflation
    if bond["type"] == "inflation":
        st.markdown(
            f'<div class="param-value-row">'
            f'<div class="param-label" style="margin:0">Prognoza inflacji CPI</div>'
            f'<div class="param-value">{inflation*100:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        inf_new = st.slider(
            "Inflacja",
            min_value=0.0,
            max_value=15.0,
            value=round(inflation * 100, 1),
            step=0.1,
            format="%.1f%%",
            label_visibility="collapsed",
        ) / 100.0
        st.caption(f"💡 Rok 1 zawsze {bond['rate_first']*100:.2f}% (stałe). Kolejne lata: CPI + {bond['inflation_margin']*100:.2f} pp.")
    elif bond["type"] == "variable":
        st.info(f"⚡ Rok 1: {bond['rate_first']*100:.2f}% (stałe). Kolejne lata: stopa ref. NBP ({NBP_REF_RATE*100:.2f}%) + {bond.get('nbp_margin',0)*100:.2f} pp.")

    return kwota_new, inf_new


def _wynik_panel(kwota: float, bond: dict, result: dict):
    """Renderuje prawy panel z wynikiem kalkulacji."""
    netto    = result["netto"]
    brutto   = result["brutto"]
    belka    = result["belka"]
    lacznie  = result["lacznie"]
    lp       = result["lacznie_pct"]
    cagr     = result["cagr_netto"]
    years    = result["years"]
    y_label  = "3 mies." if years < 1 else (f"{int(years)} {'rok' if years==1 else 'lata' if years<5 else 'lat'}")

    netto_str  = f"{int(round(netto)):,}".replace(",", " ")
    brutto_str = f"{int(round(brutto)):,}".replace(",", " ")
    belka_str  = f"{int(round(abs(belka))):,}".replace(",", " ")
    lacznie_str = f"{int(round(lacznie)):,}".replace(",", " ")
    kwota_str  = f"{int(round(kwota)):,}".replace(",", " ")

    lp_str   = f"{lp:.2f}".replace(".", ",")
    cagr_str = f"{cagr:.2f}".replace(".", ",")

    st.markdown(
        f"""
        <div class="bond-tag">{bond['name']} — {y_label}</div>
        <div class="wynik-header">ZYSK NETTO PO BELCE 19%</div>
        <div class="wynik-big-number">{netto_str} zł</div>
        <div class="wynik-subtitle">
            Łącznie po {y_label}: <b>{lacznie_str} zł</b> · Wpłata: {kwota_str} zł
        </div>
        <div class="wynik-grid">
            <div class="wynik-cell">
                <div class="wynik-cell-label">BRUTTO</div>
                <div class="wynik-cell-value val-brutto">{brutto_str} zł</div>
            </div>
            <div class="wynik-cell">
                <div class="wynik-cell-label">BELKA 19%</div>
                <div class="wynik-cell-value val-belka">−{belka_str} zł</div>
            </div>
            <div class="wynik-cell">
                <div class="wynik-cell-label">NETTO</div>
                <div class="wynik-cell-value val-netto">{netto_str} zł</div>
            </div>
        </div>
        <div class="wynik-bar">
            <div>
                <div class="wynik-bar-cell-label">ŁĄCZNIE NETTO</div>
                <div class="wynik-bar-cell-value">{lp_str} % / {y_label}</div>
            </div>
            <div>
                <div class="wynik-bar-cell-label">ROCZNIE NETTO (CAGR)</div>
                <div class="wynik-bar-cell-value">{cagr_str} % / rok</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _yearly_breakdown(result: dict):
    """Sekcja akordeon — Rozliczenie rok po roku."""
    with st.expander("📋 Rozliczenie rok po roku", expanded=False):
        rows = result["yearly_breakdown"]
        if not rows:
            st.info("Brak danych dla tego typu obligacji.")
            return

        # Budujemy tabelę HTML
        header = """
        <table class="obl-table">
        <thead><tr>
            <th>Okres</th>
            <th>Oprocentowanie</th>
            <th>Kupon brutto</th>
            <th>Kupon netto</th>
            <th>Wartość (netto)</th>
        </tr></thead><tbody>
        """
        tbody = ""
        kwota = result["lacznie"] - result["netto"]
        for row in rows:
            if "rok" in row:
                okres = f"Rok {row['rok']}"
            else:
                okres = row.get("okres", "—")
            rate_val   = row.get("rate_pct", "—")
            k_brutto   = row.get("kupon_brutto", 0)
            k_netto    = row.get("kupon_netto", k_brutto * (1 - TAX_BELKA))
            if "wartosc_netto" in row:
                wart = row["wartosc_netto"]
            else:
                wart = row.get("wartosc", kwota + k_netto)

            is_last = row == rows[-1]
            bold = " style='font-weight:700'" if is_last else ""
            tbody += (
                f"<tr{bold}>"
                f"<td>{okres}</td>"
                f"<td class='td-blue'>{rate_val}%</td>"
                f"<td class='td-blue'>{k_brutto:,.2f} zł</td>"
                f"<td class='td-netto td-green'>{k_netto:,.2f} zł</td>"
                f"<td class='td-green'>{wart:,.2f} zł</td>"
                f"</tr>"
            )

        st.markdown(header + tbody + "</tbody></table>", unsafe_allow_html=True)


def _early_redemption(result: dict, bond: dict):
    """Sekcja akordeon — Wcześniejszy wykup."""
    with st.expander("⚡ Wcześniejszy wykup", expanded=False):
        rows = result["early_redemption"]
        kwota = result["lacznie"] - result["netto"]

        st.markdown(
            f"<div style='font-size:12px;color:#9ca3af;margin-bottom:10px;'>"
            f"💸 Opłata: <b style='color:#f87171'>{bond['early_fee_note']}</b> — "
            f"odliczana od narosłych odsetek (nie od kapitału)"
            f"</div>",
            unsafe_allow_html=True,
        )

        if not rows:
            st.info("Nie dotyczy — obligacja krótkoterminowa (nie ma wcześniejszego wykupu między latami).")
            return

        header = """
        <table class="obl-table">
        <thead><tr>
            <th>Moment wykupu</th>
            <th>Opłata</th>
            <th>Zysk netto</th>
            <th>Łącznie z kapitałem</th>
        </tr></thead><tbody>
        """
        tbody = ""
        for row in rows:
            color_fee  = "td-red"
            color_net  = "td-green" if row["netto"] > 0 else "td-red"
            tbody += (
                f"<tr>"
                f"<td>{row['moment']}</td>"
                f"<td class='{color_fee}'>−{row['fee']:,.2f} zł</td>"
                f"<td class='{color_net}'>{row['netto']:,.2f} zł</td>"
                f"<td class='td-blue'>{row['lacznie']:,.2f} zł</td>"
                f"</tr>"
            )

        # Dodaj wiersz "trzymaj do końca" jako porównanie
        tbody += (
            f"<tr style='font-weight:700;background:rgba(0,230,118,0.05)'>"
            f"<td>✅ Do terminu (bez opłaty)</td>"
            f"<td style='color:#6b7280'>0,00 zł</td>"
            f"<td class='td-green'>{result['netto']:,.2f} zł</td>"
            f"<td class='td-green'>{result['lacznie']:,.2f} zł</td>"
            f"</tr>"
        )

        st.markdown(header + tbody + "</tbody></table>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  GŁÓWNA FUNKCJA MODUŁU
# ─────────────────────────────────────────────────────────────────────────────

def render_obligacje_module():
    """Renderuje pełny moduł Obligacji Skarbowych."""
    _inject_obligacje_css()

    # ── Pobierz aktualne oprocentowanie z obligacjeskarbowe.pl ────────────────
    live_rates = fetch_bond_rates()
    _apply_live_rates(live_rates)

    # Status fetcha — czy dane są live czy fallback
    cache = st.session_state.get("_obl_rates_cache", {})
    _src = cache.get("source", "fallback")
    _ts  = cache.get("fetch_time", "—")
    _n_live = len(live_rates)

    if _src == "live" and _n_live > 0:
        _status_badge = (
            f"<span style='background:rgba(0,230,118,0.12);border:1px solid rgba(0,230,118,0.35);"
            f"color:#00e676;border-radius:5px;padding:2px 8px;font-size:10px;font-weight:700;"
            f"letter-spacing:1px;'>● LIVE</span>"
            f"<span style='color:#4b5563;font-size:10px;margin-left:6px;'>"
            f"obligacjeskarbowe.pl · pobrano o {_ts} · {_n_live}/6 obligacji</span>"
        )
    else:
        _status_badge = (
            f"<span style='background:rgba(251,191,36,0.12);border:1px solid rgba(251,191,36,0.35);"
            f"color:#fbbf24;border-radius:5px;padding:2px 8px;font-size:10px;font-weight:700;"
            f"letter-spacing:1px;'>⚠ OFFLINE</span>"
            f"<span style='color:#4b5563;font-size:10px;margin-left:6px;'>"
            f"Dane wbudowane (czerwiec 2026) · brak połączenia z obligacjeskarbowe.pl</span>"
        )

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='margin-bottom: 18px;'>
        <h2 style='color:#e2e8f0;font-size:24px;font-weight:800;margin:0 0 6px 0;'>
            🏦 Kalkulator Obligacji Skarbowych
        </h2>
        <div style='display:flex;align-items:center;gap:10px;flex-wrap:wrap;'>
            {_status_badge}
            <span style='color:#374151;font-size:10px;'>·</span>
            <span style='color:#6b7280;font-size:11px;'>Podatek Belki 19% · Bez IKE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state ─────────────────────────────────────────────────────────
    if "obl_selected" not in st.session_state:
        st.session_state["obl_selected"] = "TOS"
    if "obl_kwota" not in st.session_state:
        st.session_state["obl_kwota"] = 10_000
    if "obl_inflation" not in st.session_state:
        st.session_state["obl_inflation"] = INFLATION_RATE_PL

    # ── TYP OBLIGACJI — selektor ─────────────────────────────────────────────
    st.markdown('<div style="font-size:10px;letter-spacing:2px;color:#6b7280;text-transform:uppercase;font-weight:700;margin-bottom:8px;">TYP OBLIGACJI</div>', unsafe_allow_html=True)

    selected_key = _bond_selector(st.session_state["obl_selected"])
    if selected_key != st.session_state["obl_selected"]:
        st.session_state["obl_selected"] = selected_key

    bond = BONDS[st.session_state["obl_selected"]]


    # ── LAYOUT: dwie kolumny ─────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1.8], gap="large")

    with col_left:
        st.markdown('<div class="param-panel">', unsafe_allow_html=True)
        new_kwota, new_inf = _param_panel(
            st.session_state["obl_kwota"],
            bond,
            st.session_state["obl_inflation"],
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Zaktualizuj session_state po zmianie suwaków
    st.session_state["obl_kwota"]    = new_kwota
    st.session_state["obl_inflation"] = new_inf

    # ── KALKULACJA ────────────────────────────────────────────────────────────
    result = calculate_bond_returns(
        kwota     = float(st.session_state["obl_kwota"]),
        bond      = bond,
        inflation = st.session_state["obl_inflation"],
    )

    with col_right:
        st.markdown('<div class="wynik-panel">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:10px;letter-spacing:2px;color:#6b7280;'
            'text-transform:uppercase;font-weight:700;margin-bottom:12px;">'
            'WYNIK KALKULACJI</div>',
            unsafe_allow_html=True,
        )
        _wynik_panel(float(st.session_state["obl_kwota"]), bond, result)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── AKORDEONY ────────────────────────────────────────────────────────────
    _yearly_breakdown(result)
    _early_redemption(result, bond)

    # ── STOPKA / INFO ─────────────────────────────────────────────────────────
    st.markdown("""
    <div style='margin-top:20px;padding:12px 16px;background:#0f1422;border:1px solid #1f2937;
         border-radius:10px;font-size:11px;color:#4b5563;'>
        📌 Dane oprocentowania: <a href="https://www.obligacjeskarbowe.pl" target="_blank"
        style="color:#6b7280">obligacjeskarbowe.pl</a> (czerwiec 2026) ·
        Stopa ref. NBP: {nbp}% ·
        Podatek Belki: 19% od zysku ·
        Obliczenia mają charakter poglądowy i nie stanowią doradztwa finansowego.
    </div>
    """.format(nbp=f"{NBP_REF_RATE*100:.2f}"), unsafe_allow_html=True)

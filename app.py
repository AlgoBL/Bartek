import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.styling import apply_styling
import datetime

st.set_page_config(
    page_title="Barbell Strategy Dashboard",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=900, show_spinner=False)
def fetch_control_center_data():
    from modules.ai.oracle import TheOracle
    from modules.ai.agents import LocalGeopolitics
    oracle = TheOracle()
    macro = oracle.get_macro_snapshot()
    geo = LocalGeopolitics()
    news = oracle.get_latest_news_headlines(30)
    geo_report = geo.analyze_news(news)
    return macro, geo_report

def calculate_regime_score(macro, geo_report):
    score = 50.0
    vix_ts = macro.get("VIX_TS_Ratio", 1.0)
    if vix_ts > 1.05: score += 15.0
    gex = macro.get("total_gex_billions", 0)
    if gex < 0: score += 10.0
    ted = macro.get("FRED_TED_Spread")
    if ted and ted > 0.5: score += 10.0
    fci = macro.get("FRED_Financial_Stress_Index")
    if fci and fci > 0: score += 15.0
    if macro.get("Yield_Curve_Inverted", False): score += 10.0
    ry = macro.get("FRED_Real_Yield_10Y")
    if ry and ry > 2.0: score += 5.0
    hy = macro.get("FRED_HY_Spread")
    if hy and hy > 600: score += 10.0
    cs = macro.get("FRED_Credit_Spread_BAA_AAA")
    if cs and cs > 3.0: score += 5.0
    sent = geo_report.get("compound_sentiment", 0.0)
    score -= sent * 15.0
    breadth = macro.get("Breadth_Momentum")
    if breadth and breadth < -0.02: score += 10.0
    return max(1.0, min(100.0, score))

def determine_business_cycle(macro):
    yc = macro.get("Yield_Curve_Spread", 0)
    claims = macro.get("FRED_Initial_Jobless_Claims", 250000)
    pmi = macro.get("FRED_ISM_Manufacturing_PMI", 50.0)
    if yc < 0:
        return "Spowolnienie (Slowdown)", "Zacieśnianie polityki przez bank centralny. Inwersja krzywej.", "📉", "#f39c12"
    elif claims > 300000 and yc >= 0:
        return "Recesja (Recession)", "Kryzys gospodarczy. Rosnące bezrobocie, dno rynkowe.", "💀", "#e74c3c"
    elif pmi < 50 and yc > 0.5:
        return "Odrodzenie (Recovery)", "Dno za nami. Stymulacja systemowa dyskontuje poprawę.", "🌱", "#3498db"
    else:
        return "Ekspansja (Expansion)", "Silny wzrost gospodarczy. Zyski rosną, optymizm na rynkach.", "🚀", "#2ecc71"

# ─────────────────────────────────────────────────────────────────────────────
#  KONTEKSTOWE OPISY WSKAŹNIKÓW (DYMKI)
# ─────────────────────────────────────────────────────────────────────────────

def get_help_bond_vol(v):
    base = "📘 BOND VOL (MOVE Proxy) — Zmienność rynku obligacji skarbowych USA mierzona jako roczna vol ETF TLT (20+Y). Wzrost poprzedza panikę na rynkach akcji (korelacja ~0.72 z VIX).\n\n"
    if v is None: return base + "Brak danych."
    if v < 8:   return base + f"✅ Wartość {v:.1f}% — Rynek obligacji spokojny. Normalne warunki ryzyka."
    if v < 15:  return base + f"🟡 Wartość {v:.1f}% — Umiarkowana zmienność. Wzmożona czujność."
    if v < 25:  return base + f"🔴 Wartość {v:.1f}% — Wysoka zmienność obligacji! Historycznie sygnał zbliżającego się stresu."
    return base + f"🚨 Wartość {v:.1f}% — Ekstremalny stres na rynku długu. Risk-Off natychmiast."

def get_help_vix(v):
    base = "📘 VIX 1M — CBOE Volatility Index. Implikowana zmienność 30-dniowa opcji na S&P 500. Znany jako 'Indeks Strachu'. Bazuje na modelu Black-Scholes i odzwierciedla oczekiwania rynku co do przyszłej zmienności.\n\n"
    if v is None: return base + "Brak danych."
    if v < 15:  return base + f"✅ VIX = {v:.1f} — Rynek spokojny. Niska premia za ryzyko."
    if v < 25:  return base + f"🟡 VIX = {v:.1f} — Umiarkowana niepewność. Normalny zakres."
    if v < 35:  return base + f"🔴 VIX = {v:.1f} — Wysoka zmienność! Wall Street w strachu."
    return base + f"🚨 VIX = {v:.1f} — PANIKA. Poziom krachu (2008: 80, 2020: 85). Szukaj dna."

def get_help_ted(v):
    base = "📘 TED SPREAD — Różnica między LIBOR 3M (ryzyko bankowe) a T-Bill 3M (ryzyko zerowe). Mierzy zaufanie na rynku międzybankowym. Wzrost > 0.5% sugeruje kryzys płynności dolara.\n\n"
    if v is None: return base + "Brak danych."
    if v < 0.2:  return base + f"✅ TED = {v:.2f}% — Pełne zaufanie bancowe. Rynek pieniężny spokojny."
    if v < 0.5:  return base + f"🟡 TED = {v:.2f}% — Wzmożona ostrożność. Obserwuj trend."
    return base + f"🔴 TED = {v:.2f}% — UWAGA! Napięcia na rynku pieniężnym. Historyczny prog alarmu: 0.5%."

def get_help_fci(v):
    base = "📘 STLFSI — St. Louis Fed Financial Stress Index. Agreguje 18 wskaźników rynkowych (stopy, spready, VIX). Wartość 0 = norma historyczna. Powyżej 0 = restrykcyjne warunki finansowe.\n\n"
    if v is None: return base + "Brak danych."
    if v < -1.0: return base + f"✅ STLFSI = {v:.2f} — Ultra-luźne warunki. Środowisko risk-on."
    if v < 0.0:  return base + f"✅ STLFSI = {v:.2f} — Normalne, łagodne warunki finansowe."
    if v < 2.0:  return base + f"🟡 STLFSI = {v:.2f} — Warunki restrykcyjne powyżej normy. Wzmożona ostrożność."
    return base + f"🔴 STLFSI = {v:.2f} — ALARM! Ekstremalne napięcia. Poziom kryzysu finansowego."

def get_help_yield_curve(v):
    base = "📘 KRZYWA DOCHODOWOŚCI (10Y minus 3M) — Różnica rentowności 10-letnich i 3-miesięcznych obligacji USA. Jeden z najlepszych predyktorów recesji (trafność ~77% wg Fed NY). Inwersja poprzedza recesję o 12-18 miesięcy.\n\n"
    if v is None: return base + "Brak danych."
    if v < -0.5: return base + f"🚨 INWERSJA silna ({v:+.2f}%). Sygnał nadchodzącej recesji. Strzeż kapitału."
    if v < 0.0:  return base + f"🔴 INWERSJA ({v:+.2f}%). Historyczny sygnał spowolnienia za 12-18 mies."
    if v < 0.5:  return base + f"🟡 Krzywa płaska ({v:+.2f}%). Ryzyko spowolnienia. Obserwuj."
    return base + f"✅ Krzywa normalna ({v:+.2f}%). Środowisko ekspansji gospodarczej."

def get_help_real_yield(v):
    base = "📘 REAL 10Y YIELD (TIPS) — Rentowność 10-letnich obligacji USA skorygowana o inflację (indeksowane TIPS). Realny koszt pieniądza w gospodarce. Wysoki real yield ogranicza wyceny akcji i nieruchomości.\n\n"
    if v is None: return base + "Brak danych."
    if v < 0.0:  return base + f"✅ Real Yield = {v:.2f}% — Ujemny: pieniądz tani. Sprzyja ryzykownym aktywom."
    if v < 1.5:  return base + f"🟡 Real Yield = {v:.2f}% — Umiarkowany. Neutralny wpływ na rynki."
    if v < 2.5:  return base + f"🔴 Real Yield = {v:.2f}% — Wysoki! Ogranicza P/E akcji i ceny RE."
    return base + f"🚨 Real Yield = {v:.2f}% — Bardzo wysoki koszt pieniądza. Presja na wszystkie aktywa."

def get_help_baltic(v):
    base = "📘 BALTIC DRY INDEX (BDRY ETF) — Indeks kosztu frachtu morskiego suchego ładunku. Puls globalnego handlu surowcami (ruda żelaza, węgiel, zboże). Barometr realnej aktywności ekonomicznej.\n\n"
    if v is None: return base + "Brak danych."
    if v < 1000:  return base + f"🔴 BDI = {v:.0f} — Słaby globalny handel. Sugestia spowolnienia ekonomicznego."
    if v < 2000:  return base + f"🟡 BDI = {v:.0f} — Umiarkowany poziom frachtu."
    if v < 3500:  return base + f"✅ BDI = {v:.0f} — Silny handel globalny. Risk-On sygnał."
    return base + f"✅ BDI = {v:.0f} — Bardzo silny handel. Super-cykl surowcowy możliwy."

def get_help_copper(v):
    base = "📘 DR. COPPER — Miedź ($/lb) jako barometr globalnego wzrostu przemysłowego. Rynki analizują miedź jako leading indicator PKB, bo jej cena dyskontuje przyszły popyt. Niska cena = oczekiwanie recesji.\n\n"
    if v is None: return base + "Brak danych."
    if v < 3.0:  return base + f"🔴 Miedź = ${v:.2f}/lb — Słaba. Rynki dyskontują spowolnienie."
    if v < 4.0:  return base + f"🟡 Miedź = ${v:.2f}/lb — Neutralna. Stabilne oczekiwania wzrostu."
    if v < 5.0:  return base + f"✅ Miedź = ${v:.2f}/lb — Silna. Rynki dyskontują ekspansję."
    return base + f"✅ Miedź = ${v:.2f}/lb — Bardzo wysoka. Silny cykl przemysłowy globalnie."

def get_help_cuau(v):
    base = "📘 COPPER/GOLD RATIO (×10⁴) — Stosunek ceny miedzi do złota. Miedź = wzrost przemysłowy, Złoto = safe-haven. Rosnący ratio = rynki preferują wzrost (risk-on). Koreluje z rentownościami 10Y (r~0.8). Popularyzowany przez Jeff'a Gundlacha.\n\n"
    if v is None: return base + "Brak danych."
    rv = v * 10000
    if rv < 2.0:  return base + f"🔴 Ratio = {rv:.2f} ×10⁻⁴ — Niski. Dominuje safe-haven. Risk-Off."
    if rv < 3.5:  return base + f"🟡 Ratio = {rv:.2f} ×10⁻⁴ — Neutralny. Brak wyraźnego sygnału."
    return base + f"✅ Ratio = {rv:.2f} ×10⁻⁴ — Wysoki. Risk-On: cykliczne aktywa faworyzowane."

def get_help_sentiment(v):
    base = "📘 NEWS NLP SENTIMENT — Analiza sentymentu globalnych nagłówków finansowych przez model VADER (Valence Aware Dictionary). Wartości: -1.0 (skrajny strach) do +1.0 (skrajny optymizm). AI agreguje 30+ źródeł.\n\n"
    if v is None: return base + "Brak danych."
    if v < -0.3:  return base + f"🔴 Sentyment = {v:.2f} — Globalne media w trybie strachu. Risk-Off narracja dominuje."
    if v < -0.1:  return base + f"🟡 Sentyment = {v:.2f} — Lekko negatywny. Ostrożność mediów finansowych."
    if v < 0.1:   return base + f"🟡 Sentyment = {v:.2f} — Neutralny. Brak wyraźnego kierunku narracji."
    if v < 0.3:   return base + f"✅ Sentyment = {v:.2f} — Lekko pozytywny. Optymizm rynkowy."
    return base + f"✅ Sentyment = {v:.2f} — Silnie pozytywny. Globalne media w trybie risk-on."

def get_help_breadth(v):
    base = "📘 MARKET BREADTH (RSP vs SPY) — Momentum szerokości rynku: zwrot RSP (Equal-Weight S&P500) minus zwrot SPY (Cap-Weight) za ostatni miesiąc, w punktach bazowych. Wysoki = wiele spółek uczestniczy w hossie (zdrowe). Niski = wąska hossa (niebezpieczne).\n\n"
    if v is None: return base + "Brak danych."
    bps = v * 10000
    if bps < -150: return base + f"🔴 Breadth = {bps:.0f}bp — Wąska hossa. Tylko megacap rośnie. Groźne rozdźwięki."
    if bps < -50:  return base + f"🟡 Breadth = {bps:.0f}bp — Poniżej średniej. Uczestnictwo rynku słabe."
    if bps < 50:   return base + f"🟡 Breadth = {bps:.0f}bp — Neutralny. Mieszane uczestnictwo."
    return base + f"✅ Breadth = {bps:.0f}bp — Szeroka hossa. Większość spółek uczestniczy. Zdrowe środowisko."

def get_help_fng(v):
    base = "📘 CRYPTO FEAR & GREED — Indeks sentymentu rynku kryptowalut (alternative.me). Agreguje: volatility, market momentum, social media, dominance, trends. Zakres 0-100. Używany jako proxy dla spekulacyjnego risk-on.\n\n"
    if v is None: return base + "Brak danych."
    if v < 20:  return base + f"🔴 F&G = {v:.0f} — Ekstremalny strach. Historycznie dobra strefa akumulacji."
    if v < 40:  return base + f"🟡 F&G = {v:.0f} — Strach. Spekulanci ostrożni."
    if v < 60:  return base + f"🟡 F&G = {v:.0f} — Neutralny. Rynek krypto bez wyraźnego kierunku."
    if v < 80:  return base + f"✅ F&G = {v:.0f} — Chciwość. Spekulacyjne środowisko risk-on."
    return base + f"🚨 F&G = {v:.0f} — Ekstremalny entuzjazm. Ryzyko bańki spekulacyjnej."

def get_help_gex(v):
    base = "📘 DARK POOL GEX — Gamma Exposure dealers opcyjnych na rynek SPY/SPX (mld USD). Dodatni GEX: dealerzy muszą kupować przy spadkach i sprzedawać przy wzrostach → tłumi zmienność. Ujemny GEX: odwrotnie → amplifikuje ruchy (short gamma).\n\n"
    if v is None: return base + "Brak danych."
    if v < -5:   return base + f"🚨 GEX = {v:.1f}B — Silny Short Gamma! Rynek może gwałtownie się poruszać w obu kierunkach."
    if v < 0:    return base + f"🔴 GEX = {v:.1f}B — Ujemny GEX. Zmienność niezabezpieczona. Ostrożność."
    if v < 3:    return base + f"🟡 GEX = {v:.1f}B — Neutralny. Umiarkowana stabilizacja rynku."
    return base + f"✅ GEX = {v:.1f}B — Wysoki Long Gamma. Dealerzy stabilizują rynek. Niska zmienność."

def get_help_hy(v):
    base = "📘 HY SPREAD (OAS) — High Yield Option-Adjusted Spread: różnica rentowności obligacji śmieciowych vs Treasuries. Kluczowy leading indicator kryzysu kredytowego. Wzrost > 600bps historycznie zapowiada recesję lub kryzys.\n\n"
    if v is None: return base + "Brak danych."
    if v < 300:  return base + f"✅ HY Spread = {v:.0f}bps — Bardzo niski. Rynek kredytowy w euforii. Risk-On."
    if v < 450:  return base + f"✅ HY Spread = {v:.0f}bps — Normalny. Spokojne warunki kredytowe."
    if v < 600:  return base + f"🟡 HY Spread = {v:.0f}bps — Podwyższony. Inwestorzy żądają wyższej premii."
    return base + f"🔴 HY Spread = {v:.0f}bps — ALARM! Powyżej 600bps = sygnał kryzysu kredytowego."

def get_help_credit_spread(v):
    base = "📘 CREDIT SPREAD (BAA-AAA) — Różnica rentowności obligacji korporacyjnych BAA (średnie ryzyko) vs AAA (najwyższe bezpieczeństwo). Mierzy premię za ryzyko upadłości w sektorze korporacyjnym.\n\n"
    if v is None: return base + "Brak danych."
    if v < 2.5:  return base + f"✅ Credit Spread = {v:.2f}% — Normalny poziom. Zdrowe warunki inwestycyjne."
    if v < 3.5:  return base + f"🟡 Credit Spread = {v:.2f}% — Podwyższony. Inwestorzy zaczynają się niepokoić, rynek domaga się premii."
    return base + f"🔴 Credit Spread = {v:.2f}% — ALARM! Wysokie ryzyko sekularne dla rynków kredytowych."

def get_help_m2(v):
    base = "📘 M2 MONEY SUPPLY YoY — Roczna zmiana agregatu M2 (gotówka + depozyty + fundusze pieniężne). Silny wzrost M2 napędza ceny aktywów (Milton Friedman: 'inflacja jest zawsze zjawiskiem monetarnym'). Kurczący M2 sugeruje deflację aktywów.\n\n"
    if v is None: return base + "Brak danych."
    if v < -2:  return base + f"🔴 M2 = {v:.1f}% YoY — Kurczenie płynności! Ryzyko deflacji aktywów i spowolnienia."
    if v < 0:   return base + f"🟡 M2 = {v:.1f}% YoY — Lekko ujemny. Ograniczone rezerwy płynności."
    if v < 6:   return base + f"✅ M2 = {v:.1f}% YoY — Normalny wzrost. Zdrowe środowisko płynności."
    if v < 12:  return base + f"🟡 M2 = {v:.1f}% YoY — Szybki wzrost M2. Możliwe presje inflacyjne."
    return base + f"🔴 M2 = {v:.1f}% YoY — Bardzo szybki wzrost! Wysokie ryzyko inflacji aktywów."

def get_help_vts(vix, vxmt):
    ratio = (vix / vxmt) if vxmt and vxmt > 0 else 1.0
    base = "📘 VIX TERM STRUCTURE — Porównanie implikowanej zmienności krótkoterminowej (VIX 1M) z średnioterminową (VXMT 3M). Normalna krzywa: VIX < VXMT (Contango). Odwrócenie (VIX > VXMT) = Backwardation = sygnał ostrego stresu.\n\n"
    if vix is None: return base + "Brak danych."
    status = "BACKWARDATION" if ratio > 1.02 else "CONTANGO"
    emoji = "🔴" if ratio > 1.02 else "✅"
    vxmt_disp = vxmt if vxmt else 0.0
    return base + f"{emoji} Status: {status} | Ratio VIX/VXMT = {ratio:.3f} | VIX={vix:.1f}, VXMT={vxmt_disp:.1f}"

def get_help_usd(v):
    base = "📘 USD INDEX (DXY) — Siła dolara względem koszyka 6 walut (EUR, JPY, GBP, CAD, SEK, CHF). Silny dolar = negatywny dla surowców, rynków wschodzących i aktywów ryzykownych globalnie.\n\n"
    if v is None: return base + "Brak danych."
    if v < 95:  return base + f"✅ DXY = {v:.1f} — Słaby dolar. Korzystny dla surowców i risk-on globalnie."
    if v < 103: return base + f"🟡 DXY = {v:.1f} — Neutralny poziom. Brak presji kursowej."
    return base + f"🔴 DXY = {v:.1f} — Silny dolar. Presja na EM i surowce. Risk-Off."

def get_help_gold(v):
    base = "📘 ZŁOTO (Gold Spot $/oz) — Klasyczne aktywo safe-haven. Rośnie przy strachu, inflacji i słabym dolarze. Spada przy wysokich realnych yieldsach. Barometr systemowego ryzyka i zaufania do walut fiat.\n\n"
    if v is None: return base + "Brak danych."
    if v < 1800: return base + f"🟡 Gold = ${v:.0f} — Poniżej historycznych szczytów. Umiarkowane bezpieczne poszukiwanie."
    if v < 2200: return base + f"✅ Gold = ${v:.0f} — Wysoki poziom. Rynki szukają bezpiecznej przystani."
    return base + f"🔴 Gold = ${v:.0f} — Ekstremalnie wysoki! Silna ucieczka do safe-haven. Risk-Off."

def get_help_oil(v):
    base = "📘 ROPA NAFTOWA (Crude Oil $/bbl) — Cena ropy wpływa bezpośrednio na inflację CPI i koszty produkcji. Wysoka ropa = presja inflacyjna = wyższe stopy procentowe = gorsza wycena aktywów. Niska ropa = deflacyjna siła.\n\n"
    if v is None: return base + "Brak danych."
    if v < 50:  return base + f"✅ Ropa = ${v:.0f} — Niska. Brak presji inflacyjnej z energii."
    if v < 80:  return base + f"🟡 Ropa = ${v:.0f} — Umiarkowana. Kontrolowane koszty energii."
    if v < 100: return base + f"🟡 Ropa = ${v:.0f} — Podwyższona. Presja inflacyjna przez energię."
    return base + f"🔴 Ropa = ${v:.0f} — Bardzo wysoka! Silna presja inflacyjna. Stagflacja możliwa."

# ─────────────────────────────────────────────────────────────────────────────
#  WIZUALIZACJE
# ─────────────────────────────────────────────────────────────────────────────

def draw_regime_radar(score):
    if score <= 30:   needle_color = "#00e676"
    elif score <= 65: needle_color = "#ffea00"
    else:             needle_color = "#ff1744"

    # Określa strefę słownie
    zone_label = "🟢 HOSSA / RISK-ON" if score <= 30 else ("🔴 PANIKA / ALARM" if score > 65 else "🟡 NEUTRAL / OSTROŻNIE")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            'font': {'size': 58, 'color': needle_color, 'family': 'Inter, monospace'},
            'suffix': "",
        },
        title={
            # Tytuł nad wykresem, pod nim etykieta strefy — bez delta żeby nie nakładać
            'text': (
                "<span style='font-size:15px;color:#aaa;letter-spacing:2px;'>REGIME SCORE</span>"
                f"<br><span style='font-size:11px;color:{needle_color};'>{zone_label}</span>"
            ),
            'font': {'size': 15, 'color': '#aaa', 'family': 'Inter'}
        },
        gauge={
            'axis': {
                'range': [1, 100],
                'tickwidth': 1,
                'tickcolor': "#555",
                # Tylko czyste liczby — eliminacja duplikatów z annotations
                'tickvals': [1, 25, 50, 75, 100],
                'ticktext':  ['1', '25', '50', '75', '100'],
                'tickfont': {'size': 11, 'color': '#888', 'family': 'Inter'}
            },
            'bar': {'color': needle_color, 'thickness': 0.10},
            'bgcolor': "#0a0b0e",
            'borderwidth': 1, 'bordercolor': "#2a2a3a",
            'steps': [
                {'range': [1,  30],  'color': "rgba(0, 230, 118, 0.20)"},
                {'range': [30, 65],  'color': "rgba(255, 234, 0, 0.13)"},
                {'range': [65, 100], 'color': "rgba(255, 23, 68, 0.23)"},
            ],
            'threshold': {'line': {'color': needle_color, 'width': 5}, 'thickness': 0.92, 'value': score}
        }
    ))
    fig.update_layout(
        height=330,
        # t=90 dla tytułu dwulinijkowego, b=20 dla czytelności
        margin=dict(l=40, r=40, t=90, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': 'Inter'},
        # Czytelne etykiety stref POZA łukiem (poza kolizją z tickami)
        annotations=[
            dict(text="HOSSA",   x=0.12, y=0.08, xref="paper", yref="paper",
                 font=dict(size=9, color="#00e676", family='Inter'), showarrow=False, opacity=0.7),
            dict(text="PANIKA",  x=0.88, y=0.08, xref="paper", yref="paper",
                 font=dict(size=9, color="#ff1744", family='Inter'), showarrow=False, opacity=0.7),
        ]
    )
    return fig


def draw_advanced_gauge(title, value, min_val, max_val, invert=False, suffix="", prefix=""):
    """
    Arc-gauge z czytelną czcionką Inter, 3 strefami i tytułem bez kolizji.
    height=190, t=44 (miejsce na tytuł), b=28 (miejsce pod łukiem).
    """
    if value is None:
        return go.Figure()

    P_GREEN  = "#00e676"
    P_YELLOW = "#ffea00"
    P_RED    = "#ff1744"

    norm_val = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    norm_val = max(0.0, min(1.0, norm_val))

    # ── Strefa kolorów (styl identyczny ze Skanerem) ──────────────────────────
    # Skan używa: #2ecc71 (zielony) / #f39c12 (pomarańczowy) / #e74c3c (czerwony)
    C_GREEN  = "#2ecc71"
    C_YELLOW = "#f39c12"
    C_RED    = "#e74c3c"

    z1 = (max_val - min_val) * 0.35 + min_val
    z2 = (max_val - min_val) * 0.65 + min_val

    if not invert:
        steps = [
            {'range': [min_val, z1], 'color': C_GREEN},
            {'range': [z1, z2],      'color': C_YELLOW},
            {'range': [z2, max_val], 'color': C_RED},
        ]
    else:
        steps = [
            {'range': [min_val, z1], 'color': C_RED},
            {'range': [z1, z2],      'color': C_YELLOW},
            {'range': [z2, max_val], 'color': C_GREEN},
        ]

    if abs(value) < 10:     fmt = '.2f'
    elif abs(value) < 1000: fmt = '.1f'
    else:                   fmt = '.0f'

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={
            'prefix': prefix,
            'suffix': suffix,
            # Styl liczby jak w Skanerze: font size=32, biały
            'font': {'size': 32, 'color': 'white', 'family': 'Inter, monospace'},
            'valueformat': fmt
        },
        title={'text': '', 'font': {'size': 1}},
        gauge={
            'axis': {
                'range': [min_val, max_val],
                'tickwidth': 1,
                'tickcolor': "#555",
                'nticks': 4,
                'tickfont': {'size': 9, 'color': '#aaa'}
            },
            # Styl Skanera: przezroczysty pasek → widoczne tylko strefy + linia
            'bar': {'color': 'rgba(0,0,0,0)'},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': steps,
            # Biała linia threshold jak w Skanerze
            'threshold': {
                'line': {'color': 'white', 'width': 5},
                'thickness': 0.85,
                'value': value
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=5, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': 'Inter'}
    )
    return fig


def draw_vix_term_structure(vix_1m, vxmt):
    if vix_1m is None:
        return go.Figure()
    vxmt = vxmt or vix_1m
    ratio = vix_1m / vxmt if vxmt else 1.0
    is_back = ratio > 1.02
    c1 = "#ff1744" if is_back else "#00e676"
    c2 = "#ff5252" if is_back else "#00bcd4"
    status_text  = "BACKWARDATION ⚠️" if is_back else "CONTANGO ✅"
    status_color = "#ff1744" if is_back else "#00e676"

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[vix_1m], y=["VIX 1M"],  orientation='h', marker_color=c1,
                         text=f"{vix_1m:.1f}", textposition='inside',
                         textfont=dict(size=14, color='white'), name="VIX 1M"))
    fig.add_trace(go.Bar(x=[vxmt],   y=["VXMT 3M"], orientation='h', marker_color=c2,
                         text=f"{vxmt:.1f}",   textposition='inside',
                         textfont=dict(size=14, color='white'), name="VXMT 3M"))
    fig.update_layout(
        height=120, margin=dict(l=10, r=10, t=25, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white", 'family': 'Inter'}, barmode='group', showlegend=False,
        xaxis=dict(range=[0, max(vix_1m, vxmt) * 1.3], gridcolor="#1c1c2e", tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=11))
    )
    return fig


def draw_credit_spread_chart(hy, cs_baa_aaa):
    if hy is None and cs_baa_aaa is None:
        return go.Figure()
    labels, values, colors = [], [], []
    if hy is not None:
        labels.append("HY Spread (bps)")
        values.append(hy)
        colors.append("#ff1744" if hy > 600 else "#ffea00" if hy > 400 else "#00e676")
    if cs_baa_aaa is not None:
        labels.append("Credit Spread %")
        values.append(cs_baa_aaa)
        colors.append("#ff1744" if cs_baa_aaa > 3.5 else "#ffea00" if cs_baa_aaa > 2.5 else "#00e676")

    fig = go.Figure()
    for lbl, val, col in zip(labels, values, colors):
        fig.add_trace(go.Bar(x=[val], y=[lbl], orientation='h', marker_color=col,
                             text=f"{val:.1f}", textposition='inside',
                             textfont=dict(size=13, color='white'), name=lbl))
    fig.update_layout(
        height=120 if len(labels) > 1 else 90,
        margin=dict(l=2, r=2, t=25, b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={'color': 'white', 'family': 'Inter'}, showlegend=False,
        xaxis=dict(tickfont=dict(size=8), gridcolor="#1c1c2e"),
        yaxis=dict(tickfont=dict(size=10))
    )
    return fig


def show_gauge(label, fig, help_text, overlap_margin="-20px"):
    """
    Renderuje gauge z tytułem w stałej wysokości poza Plotly.
    Jeden element HTML (tytuł + ℹ tooltip) gwarantuje równe wyrównanie kolumn.
    """
    safe = help_text.replace('"', "'").replace('\n', ' ').replace('<', '&lt;').replace('>', '&gt;')
    st.markdown(
        f"""
        <div style='
            position: relative;
            z-index: 10;
            height: 22px;
            line-height: 22px;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            font-family: Inter, sans-serif;
            font-size: 12px;
            color: #aaa;
            padding: 0 12px;
            cursor: help;
            margin-bottom: {overlap_margin};
        ' title="{safe}">
            <span style='color:#555;margin-right:4px;'>&#9432;</span>{label}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.plotly_chart(fig, use_container_width=True)


def get_active_alerts(score, macro, geo_report):
    alerts = []
    vix = macro.get("VIX_1M")
    if vix and vix > 25: alerts.append(f"🔴 VIX={vix:.1f} — Wysoka zmienność (>25)")
    if macro.get("Yield_Curve_Inverted"):
        yc = macro.get("Yield_Curve_Spread", 0)
        alerts.append(f"🔴 Krzywa odwrócona ({yc:.2f}%)")
    vix_ts = macro.get("VIX_TS_Ratio", 1.0)
    if vix_ts > 1.05: alerts.append(f"🔴 VIX Backwardation ({vix_ts:.2f})")
    gex = macro.get("total_gex_billions")
    if gex is not None and gex < -2: alerts.append(f"🟡 GEX={gex:.1f}B — Short Gamma")
    hy = macro.get("FRED_HY_Spread")
    if hy and hy > 600: alerts.append(f"🔴 HY Spread={hy:.0f}bps — Kryzys kredytowy")
    elif hy and hy > 400: alerts.append(f"🟡 HY Spread={hy:.0f}bps — Podwyższone ryzyko")
    ted = macro.get("FRED_TED_Spread")
    if ted and ted > 0.5: alerts.append(f"🔴 TED={ted:.2f} — Napięcie bankowe")
    fci = macro.get("FRED_Financial_Stress_Index")
    if fci and fci > 0: alerts.append(f"🟡 STLFSI={fci:.2f} — Powyżej normy")
    breadth = macro.get("Breadth_Momentum")
    if breadth and breadth < -0.02: alerts.append(f"🟡 Breadth={breadth*10000:.0f}bp — Wąska hossa")
    sent = geo_report.get("compound_sentiment", 0)
    if sent < -0.15: alerts.append(f"🟡 Sentyment NLP={sent:.2f} — Negatywny")
    m2 = macro.get("FRED_M2_YoY_Growth")
    if m2 is not None and m2 < 0: alerts.append(f"🟡 M2={m2:.1f}% — Kurczenie płynności")
    if not alerts: alerts.append("✅ Brak aktywnych alertów — środowisko Risk-On")
    return alerts

def get_vanguard_report(score, macro, geo_report):
    sent = geo_report.get("compound_sentiment", 0)
    cycle, _, _, _ = determine_business_cycle(macro)
    if score > 70:
        return "ALARM: Wysokie ryzyko systemowe. Dark Pools i VIX wskazują na kaskadową zmienność. Rekomendacja: Obrona kapitału.", "#e74c3c"
    elif score < 35 and sent > 0.1:
        return "STATUS: Rynek w silnym reżimie Risk-On. Płynność wspiera wzrosty. Rekomendacja: Ekspansja w Risky Sleeve.", "#2ecc71"
    else:
        return f"STATUS: Reżim mieszany. Faza {cycle}. Rynek szuka kierunku przy stabilnych warunkach finansowych.", "#3498db"

# ─────────────────────────────────────────────────────────────────────────────
#  STRONA GŁÓWNA
# ─────────────────────────────────────────────────────────────────────────────

def home():
    st.markdown(apply_styling(), unsafe_allow_html=True)
    st.markdown("""
    <style>
        div.block-container { padding-top: 0.6rem !important; padding-bottom: 0.5rem !important; }
        div[data-testid="stVerticalBlock"] > div { gap: 0.2rem; }
        .stPlotlyChart { margin-bottom: -8px; margin-top: -14px; }
        h4 { margin-bottom: 4px !important; margin-top: 4px !important; }
    </style>
    """, unsafe_allow_html=True)

    if "force_navigate" in st.session_state:
        target = st.session_state.pop("force_navigate")
        if target == "📉 Symulator": st.switch_page("pages/1_Symulator.py")
        elif target == "⚡ Stress Test": st.switch_page("pages/3_Stress_Test.py")

    with st.spinner("Synchronizacja terminala V9.5..."):
        try:
            macro, geo_report = fetch_control_center_data()
        except Exception as e:
            st.error(f"Błąd synchronizacji terminala: {e}")
            macro, geo_report = {}, {}

    if not macro:
        st.warning("Brak połączenia z siecią sensorów.")
        return

    score = calculate_regime_score(macro, geo_report)
    report_text, report_color = get_vanguard_report(score, macro, geo_report)

    # ─── ROW 1: MAIN GAUGE | BUSINESS CYCLE | VIX TS + SAFE HAVEN ────────────
    col_main, col_cycle, col_vts = st.columns([2.2, 1.0, 1.8])

    with col_main:
        st.plotly_chart(draw_regime_radar(score), use_container_width=True)

    with col_cycle:
        phase, desc, icon, color = determine_business_cycle(macro)
        yc = macro.get("Yield_Curve_Spread", 0)
        claims = macro.get("FRED_Initial_Jobless_Claims")
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0f111a,#1a1c28);padding:18px 14px;
                    border-radius:12px;text-align:center;border:1px solid #2a2a3a;
                    height:310px;display:flex;flex-direction:column;justify-content:center;'>
            <div style='font-size:52px;line-height:1;'>{icon}</div>
            <div style='color:{color};margin-top:8px;font-size:18px;font-weight:700;'>{phase}</div>
            <div style='color:#888;font-size:11px;margin-top:6px;line-height:1.35;'>{desc}</div>
            <div style='margin-top:14px;border-top:1px solid #2a2a3a;padding-top:10px;'>
                <span style='color:#aaa;font-size:10px;'>10Y-3M: <b style='color:{color}'>{yc:+.2f}%</b></span>
                {"&nbsp;&nbsp;|&nbsp;&nbsp;<span style='color:#aaa;font-size:10px;'>Claims: <b style='color:#f39c12'>" + f"{claims/1000:.0f}k</b></span>" if claims else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_vts:
        vix_1m = macro.get("VIX_1M")
        vxmt   = macro.get("VXMT_MidTerm")
        usd    = macro.get("US_Dollar_Index")
        gold   = macro.get("Gold")
        oil    = macro.get("Crude_Oil")

        rA, rB, rC = st.columns(3)
        for col_ref, lbl, val, suf, help_fn in [
            (rA, "🇺🇸 USD",  usd,  "",  get_help_usd),
            (rB, "🥇 Gold",  gold, "$", get_help_gold),
            (rC, "🛢️ Oil",   oil,  "$", get_help_oil),
        ]:
            with col_ref:
                if val:
                    v_color = "#f1c40f" if "Gold" in lbl else "#00ccff"
                    safe_tip = help_fn(val).replace('"', "'").replace('\n', ' ').replace('<','&lt;').replace('>','&gt;')
                    st.markdown(f"""
                    <div title="{safe_tip}"
                         style='background:#0f111a;border:1px solid #2a2a3a;border-radius:8px;
                                padding:8px 4px;text-align:center;margin-top:45px;cursor:default;margin-bottom:8px;'>
                        <div style='font-size:10px;color:#777;'>&#9432; {lbl}</div>
                        <div style='font-size:18px;font-weight:700;color:{v_color};'>{suf}{val:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        # Render VIX Term Structure Below USD/Gold/Oil
        if vix_1m is not None:
            vxmt_val = vxmt or vix_1m
            ratio = vix_1m / vxmt_val if vxmt_val else 1.0
            is_back = ratio > 1.02
            status_text  = "BACKWARDATION ⚠️" if is_back else "CONTANGO ✅"
            status_color = "#ff1744" if is_back else "#00e676"
            label = f"VIX TS — <span style='color:{status_color}'>{status_text}</span>"
            show_gauge(label, draw_vix_term_structure(vix_1m, vxmt), get_help_vts(vix_1m, vxmt), overlap_margin="0px")
        else:
            st.plotly_chart(draw_vix_term_structure(vix_1m, vxmt), use_container_width=True)

    st.divider()

    # ─── ROW 2: 5-PILLAR GRID ─────────────────────────────────────────────────
    PILLAR_STYLE = "text-align:center;font-size:13px;font-weight:700;letter-spacing:1px;margin-bottom:4px;padding:4px 0;border-radius:6px;"
    p1, p2, p3, p4, p5 = st.columns(5)

    # ── PILLAR 1: STRESS & VOL ──
    with p1:
        st.markdown(f"<h4 style='{PILLAR_STYLE}color:#ff1744;background:rgba(255,23,68,0.08);'>📡 STRESS &amp; VOL</h4>", unsafe_allow_html=True)

        bv = macro.get("Bond_Vol_Proxy")
        if bv is not None:
            show_gauge("Bond Vol (MOVE Proxy)", draw_advanced_gauge("Bond Vol", bv, 3, 35, invert=False, suffix="%"), get_help_bond_vol(bv))

        vix1 = macro.get("VIX_1M")
        if vix1 is not None:
            show_gauge("VIX 1M (Implied Vol)", draw_advanced_gauge("VIX 1M", vix1, 10, 50, invert=False), get_help_vix(vix1))

        ted = macro.get("FRED_TED_Spread")
        if ted is not None:
            show_gauge("TED Spread", draw_advanced_gauge("TED Spread", ted, 0, 1.5, invert=False), get_help_ted(ted))

    # ── PILLAR 2: MACRO & POLICY ──
    with p2:
        st.markdown(f"<h4 style='{PILLAR_STYLE}color:#3498db;background:rgba(52,152,219,0.08);'>🏛️ MACRO &amp; POLICY</h4>", unsafe_allow_html=True)

        fci = macro.get("FRED_Financial_Stress_Index")
        if fci is not None:
            show_gauge("Financial Stress (STLFSI)", draw_advanced_gauge("STLFSI", fci, -2.5, 6, invert=False), get_help_fci(fci))

        yc = macro.get("Yield_Curve_Spread", 0)
        show_gauge("Yield Curve 10Y - 3M", draw_advanced_gauge("Yield Curve", yc, -1.5, 3.5, invert=True, suffix="%"), get_help_yield_curve(yc))

        ry = macro.get("FRED_Real_Yield_10Y")
        if ry is not None:
            show_gauge("Real 10Y Yield (TIPS)", draw_advanced_gauge("Real Yield", ry, -1.0, 4.0, invert=False, suffix="%"), get_help_real_yield(ry))

    # ── PILLAR 3: REAL ECONOMY ──
    with p3:
        st.markdown(f"<h4 style='{PILLAR_STYLE}color:#2ecc71;background:rgba(46,204,113,0.08);'>🏭 REAL ECONOMY</h4>", unsafe_allow_html=True)

        bdry = macro.get("Baltic_Dry")
        if bdry is not None:
            show_gauge("Baltic Dry Index", draw_advanced_gauge("Baltic Dry", bdry, 500, 5000, invert=True), get_help_baltic(bdry))

        cu = macro.get("Copper")
        if cu is not None:
            show_gauge("Dr. Copper ($/lb)", draw_advanced_gauge("Dr. Copper", cu, 2.0, 6.5, invert=True, prefix="$"), get_help_copper(cu))

        cu_au = macro.get("CuAu_Ratio")
        if cu_au is not None:
            show_gauge("Cu/Au Ratio ×10⁴", draw_advanced_gauge("Cu/Au Ratio", cu_au * 10000, 1.0, 5.0, invert=True), get_help_cuau(cu_au))

    # ── PILLAR 4: SENTIMENT & BREADTH ──
    with p4:
        st.markdown(f"<h4 style='{PILLAR_STYLE}color:#f1c40f;background:rgba(241,196,15,0.08);'>🧠 SENTIMENT</h4>", unsafe_allow_html=True)

        sent = geo_report.get("compound_sentiment", 0)
        show_gauge("News NLP Sentiment", draw_advanced_gauge("NLP Sentiment", sent, -1.0, 1.0, invert=True), get_help_sentiment(sent))

        breadth = macro.get("Breadth_Momentum")
        if breadth is not None:
            show_gauge("Market Breadth (bp)", draw_advanced_gauge("Breadth", breadth * 10000, -300, 300, invert=True), get_help_breadth(breadth))

        fng = macro.get("Crypto_FearGreed")
        if fng is not None:
            show_gauge("Crypto Fear & Greed", draw_advanced_gauge("Fear & Greed", fng, 0, 100, invert=True), get_help_fng(fng))

    # ── PILLAR 5: CREDIT & LIQUIDITY ──
    with p5:
        st.markdown(f"<h4 style='{PILLAR_STYLE}color:#a855f7;background:rgba(168,85,247,0.08);'>💳 CREDIT &amp; LIQ</h4>", unsafe_allow_html=True)

        gex = macro.get("total_gex_billions")
        if gex is not None:
            show_gauge("Dark Pool GEX", draw_advanced_gauge("GEX", gex, -15, 15, invert=True, suffix="B"), get_help_gex(gex))

        hy = macro.get("FRED_HY_Spread")
        if hy is not None:
            show_gauge("HY Spread (OAS)", draw_advanced_gauge("HY Spread", hy, 200, 1200, invert=False, suffix=" bps"), get_help_hy(hy))

        m2 = macro.get("FRED_M2_YoY_Growth")
        if m2 is not None:
            show_gauge("M2 Money Supply YoY", draw_advanced_gauge("M2 YoY", m2, -5, 15, invert=True, suffix="%"), get_help_m2(m2))

        cs_baa = macro.get("FRED_Credit_Spread_BAA_AAA")
        if cs_baa is not None or hy is not None:
            cs_fig = draw_credit_spread_chart(hy, cs_baa)
            if cs_fig.data:
                help_text = ""
                if hy is not None: help_text += get_help_hy(hy) + "\n\n"
                if cs_baa is not None: help_text += get_help_credit_spread(cs_baa)
                show_gauge("Credit & HY Spread", cs_fig, help_text.strip(), overlap_margin="0px")

    st.divider()

    # ─── ROW 3: INTELLIGENCE REPORT ───────────────────────────────────────────
    alerts = get_active_alerts(score, macro, geo_report)
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    n_red    = sum(1 for a in alerts if "🔴" in a)
    n_yellow = sum(1 for a in alerts if "🟡" in a)
    score_color = "#e74c3c" if n_red > 1 else "#f39c12" if n_yellow > 1 else "#2ecc71"
    alerts_html = "".join([f"<span style='margin-right:18px;font-size:12px;color:#ccc;'>{a}</span>" for a in alerts])

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0a0b0e,#0f111a);
                padding:16px 20px;border-radius:12px;
                border-left:5px solid {report_color};border:1px solid #1e1e2e;margin-top:4px;'>
        <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;'>
            <span style='font-size:11px;color:#555;'>⏱ {now_str} &nbsp;|&nbsp; V9.5 Terminal &nbsp;|&nbsp;
                🎯 Score: <b style='color:{score_color}'>{score:.0f}/100</b> &nbsp;|&nbsp;
                🔴 Alerty: <b style='color:#e74c3c'>{n_red}</b> &nbsp;
                🟡 Ostrzeżenia: <b style='color:#f39c12'>{n_yellow}</b>
            </span>
        </div>
        <p style='margin:0 0 10px 0;color:white;font-size:16px;line-height:1.5;'>
            <b>📋 Raport:</b> {report_text}
        </p>
        <div style='border-top:1px solid #1e1e2e;padding-top:8px;flex-wrap:wrap;display:flex;'>
            {alerts_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


pages = {
    "Start": [
        st.Page(home, title="Strona główna", icon="🏠", default=True),
    ],
    "Narzędzia Analityczne": [
        st.Page("pages/1_Symulator.py", title="Symulator", icon="📉"),
        st.Page("pages/2_Skaner.py",    title="Skaner",    icon="🔍"),
        st.Page("pages/3_Stress_Test.py", title="Stress Test", icon="⚡"),
    ],
    "Planowanie": [
        st.Page("pages/4_Emerytura.py", title="Emerytura", icon="🏖️"),
    ]
}

pg = st.navigation(pages)
pg.run()

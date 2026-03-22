import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from modules.styling import apply_styling
from modules.i18n import t

st.set_page_config(page_title="Day Trading — Matematyka", page_icon="📈", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;margin-bottom:4px'>📈 Day Trading — Niewidzialna Matematyka</h1>
<p style='text-align:center;color:#6b7280;font-size:14px;margin-bottom:24px'>
System Wartości Oczekiwanej · Ryzyko Ruiny · Law of Large Numbers · Mikrostruktura kosztów
</p>
""", unsafe_allow_html=True)

CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px 20px;margin-bottom:8px"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px"
NOTE = "color:#6b7280;font-size:12px;line-height:1.6"

# KAPITAŁ POCZĄTKOWY
st.sidebar.markdown("### 💰 Ustawienia Główne")
starting_capital = st.sidebar.number_input("Kapitał Początkowy (PLN)", min_value=1000, max_value=1000000, value=10000, step=1000)

st.markdown(f"<div style='text-align:center;margin-bottom:20px;font-size:18px;'><b>Obecny budżet symulacji: <span style='color:#00e676'>{starting_capital:,.0f} PLN</span></b></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SEKCJA 1 — WARTOŚĆ OCZEKIWANA I HEATMAPA R:R
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## ⚖️ Sekcja 1 — Oczekiwana Wartość (EV) i Skuteczność (Win Rate)")

col_ev1, col_ev2 = st.columns([3, 2])

with col_ev1:
    wr_range = np.linspace(0.1, 0.9, 81) # Win rate od 10% do 90%
    rr_range = np.linspace(0.5, 5.0, 46) # Risk:Reward od 1:0.5 do 1:5.0
    
    Z_ev = np.zeros((len(rr_range), len(wr_range)))
    
    for i, rr in enumerate(rr_range):
        for j, wr in enumerate(wr_range):
            # EV = (WR * RR) - (LossRate * 1)
            # Przykład: ryzyko to przeważnie 1 jednostka (np. $10), nagroda to RR jednostek (np. $30)
            Z_ev[i, j] = (wr * rr) - ((1 - wr) * 1)

    colorscale_ev = [
        [0.0, "#ff1744"],   # Ekstremalna strata
        [0.45, "#f39c12"],  # Lekka strata
        [0.5, "#2a2a3a"],   # Break-even
        [0.6, "#00e676"],   # Zysk
        [1.0, "#3498db"]    # Super zysk
    ]
    
    fig_ev = go.Figure(data=go.Heatmap(
        z=Z_ev,
        x=wr_range * 100,
        y=rr_range,
        colorscale=colorscale_ev,
        zmin=-1,
        zmax=2,
        colorbar=dict(title=dict(text="EV (Zysk/Ryzyko)", font=dict(color="white")), tickfont=dict(color="white")),
        hovertemplate="Win Rate: %{x:.0f}%<br>Risk:Reward: 1:%{y:.1f}<br>EV: %{z:.2f}R<extra></extra>"
    ))
    
    user_wr = st.slider("Twoja zakładywana Skuteczność (Win Rate %)", 10, 90, 40, key="ev_wr")
    user_rr = st.slider("Twój średni Risk:Reward (1:X)", 0.5, 5.0, 2.0, 0.1, key="ev_rr")
    
    user_ev = (user_wr/100 * user_rr) - ((1 - user_wr/100) * 1)
    ev_color = "#00e676" if user_ev > 0 else "#ff1744"
    
    fig_ev.add_trace(go.Scatter(
        x=[user_wr], y=[user_rr], mode="markers", 
        marker=dict(size=14, color=ev_color, line=dict(color="white", width=2)),
        name="Twoja Strategia"
    ))

    fig_ev.update_layout(
        title="Mapa Rentowności (Złudzenie 90% Win-Rate)", height=400,
        xaxis=dict(title="Skuteczność (Win Rate %)", gridcolor="#1c1c2e"), 
        yaxis=dict(title="Risk:Reward Ratio (1:X)", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=40, r=20, t=50, b=40)
    )
    # Linia Break-Even EV=0: RR = (1-WR)/WR
    be_wr = np.linspace(0.16, 0.9, 100)
    be_rr = (1 - be_wr) / be_wr
    # Filter valid points within plot area
    valid_be = be_rr <= 5.0
    fig_ev.add_trace(go.Scatter(x=be_wr[valid_be]*100, y=be_rr[valid_be], mode="lines", line=dict(color="white", dash="dash", width=1), name="Punkt Wyjścia na Zero"))
    
    st.plotly_chart(fig_ev, use_container_width=True)

with col_ev2:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>📐 EV: Expected Value</div>
    <p style='{NOTE}'>
    Prawdziwy trader nie przejmuje się tym, <b>jak często</b> ma rację, ale <b>ile zarabia</b>, gdy ma rację, i ile traci, gdy się myli.<br><br>
    Jeśli ryzykujesz <b>1 R</b>, by zyskać <b>{user_rr} R</b>, mając skuteczność <b>{user_wr}%</b>:<br><br>
    Wartość Oczekiwana (EV) 1 transakcji = <b style='color:{ev_color};font-size:16px'>{user_ev:.2f} R</b>
    </p></div>""", unsafe_allow_html=True)
    
    if user_ev > 0:
        st.success(f"System ZYSKOWNY! Na każdą zaryzykowaną złotówkę w długim terminie wyciągniesz statystycznie średnio +{user_ev:.2f} PLN netto nad ten zaryzykowany kapitał.")
    else:
        st.error(f"System STRATNY! Niezależnie od analizy technicznej, ten system to ujemna wartość oczekiwana. Statystycznie tracisz {abs(user_ev):.2f} R na każdą transakcję.")


# ═══════════════════════════════════════════════════════════
# SEKCJA 2 — POZYCJONOWANIE I RYZYKO RUINY
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 💣 Sekcja 2 — Wielkość Pozycji i Asymetria Odrabiania Strat")

col_po1, col_po2 = st.columns(2)

with col_po1:
    risk_per_trade_pct = st.slider("Ryzyko na jeden trejd (% kapitału)", 0.1, 15.0, 1.0, 0.1, key="risk_pct")
    risk_pln = starting_capital * (risk_per_trade_pct / 100)
    
    st.markdown(f"""<div style='background:#1a1c28;border-left:4px solid #f39c12;padding:12px;margin-bottom:12px'>
    <b>Złota Zasada Ochrony Kapitału:</b><br>
    Twój twardy Stop Loss powinien zawsze ucinać stratę na poziomie dokładnie <b>{risk_pln:,.0f} PLN</b> (czyli {risk_per_trade_pct}% Twojego budżetu {starting_capital:,.0f} PLN).
    </div>""", unsafe_allow_html=True)
    
    # Kalkulacja Drawdown -> Recovery
    drawdowns = np.linspace(0.01, 0.95, 95)
    recovery_needed = (1 / (1 - drawdowns)) - 1
    
    fig_rec = go.Figure()
    fig_rec.add_trace(go.Scatter(x=drawdowns*100, y=recovery_needed*100, mode="lines", line=dict(color="#ff1744", width=3), name="Wymagany Zysk (%)"))
    # Highlight 50%
    fig_rec.add_trace(go.Scatter(x=[50], y=[100], mode="markers", marker=dict(size=12, color="#f39c12"), name="Mroczna połówka"))
    
    fig_rec.update_layout(
        title="Matematyka Bankructwa (Jak odrobić stratę?)", height=300,
        xaxis=dict(title="Twoja Strata (Drawdown) % kapitału", gridcolor="#1c1c2e"), 
        yaxis=dict(title="Wymagany zysk do wyjścia na 0 (%)", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=40, r=20, t=40, b=40), showlegend=False
    )
    fig_rec.add_annotation(x=50, y=100, text="Strata -50% wymaga zysku +100%", showarrow=True, arrowhead=1, ax=-60, ay=-40)
    fig_rec.add_annotation(x=80, y=400, text="Strata -80% wymaga zysku +400%", showarrow=True, arrowhead=1, ax=-60, ay=40)
    st.plotly_chart(fig_rec, use_container_width=True)

with col_po2:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>Ryzyko Ruiny (Risk of Ruin)</div>
    <p style='{NOTE}'>
    Jeśli ryzykujesz <b>{risk_per_trade_pct}%</b> kapitału na trejd, ile rzędowych strat dzieli Cię od utraty 50% kapitału (poziomu, z którego bardzo ciężko psychicznie wrócić)?
    </p></div>""", unsafe_allow_html=True)
    
    # Obliczenie strat rzędu (compound losses) by stracić 50% kapitału
    # (1 - risk)^n = 0.5  => n = ln(0.5) / ln(1 - risk)
    if risk_per_trade_pct < 100:
        streaks_to_halve = np.log(0.5) / np.log(1 - (risk_per_trade_pct/100))
        streaks_to_ruin_90 = np.log(0.1) / np.log(1 - (risk_per_trade_pct/100))
        
        c1, c2 = st.columns(2)
        c1.metric("Trejdów na startę -50%", f"{int(streaks_to_halve)} z rzędu", delta="Psychological break", delta_color="off")
        c2.metric("Trejdów na startę -90%", f"{int(streaks_to_ruin_90)} z rzędu", delta="Konto spalone", delta_color="inverse")
        
        if risk_per_trade_pct > 3.0:
            st.error(f"Ryzykując {risk_per_trade_pct}% na transakcję grasz w rosyjską ruletkę. Wystarczy {int(streaks_to_halve)} stratowanych trejdów podrząd, abyś stracił połowę z {starting_capital} PLN.")
        else:
            st.success(f"Dobre zarządzanie kasą! Amortyzujesz serie strat. Możesz pomylić się {int(streaks_to_halve)} razy i nadal będziesz miał połowę kapitału do odrobienia.")
            
    # Zależność kapitału a margin
    st.info("Kiedy tracisz 10% kapitału, do odrobienia straty potrzebujesz wypracować nie 10%, ale **11.1%**. Kiedy tracisz 50%, odrabiać musisz **100%**. Straty kompensują się znacznie szybciej niż zyski tną twój kapitał. To dlatego grubasy zawsze ucinają straty natychmiast.")

# ═══════════════════════════════════════════════════════════
# SEKCJA 3 — PRAWO WIELKICH LICZB I MONTE CARLO
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🎲 Sekcja 3 — Law of Large Numbers (Symulacja Monte Carlo)")
st.markdown("<p style='color:#bbb;font-size:14px'>Odtworzymy tu 100 alternatywnych wszechświatów, gdzie w każdym z nich masz identyczny system o tych samych parametrach skuteczności.</p>", unsafe_allow_html=True)

mc_num_trades = st.slider("Liczba transakcji w teście", 100, 1000, 250, key="mc_trades")

# Używamy parametrów z sekcji 1 i 2
win_prob = user_wr / 100
rr = user_rr
risk_fraction = risk_per_trade_pct / 100

st.markdown(f"**Używane parametry systemu:** Win Rate: **{user_wr}%** | Risk:Reward: **1:{user_rr}** | Ryzyko per trade: **{risk_per_trade_pct}%**")

if st.button("Uruchom Symulację Monte Carlo (100 Ścieżek) 🚀", type="primary"):
    with st.spinner("Generowanie równoległych rzeczywistości..."):
        num_simulations = 100
        
        # Inicjalizacja wyników
        equity_curves = np.zeros((num_simulations, mc_num_trades + 1))
        equity_curves[:, 0] = starting_capital
        
        max_drawdowns = []
        consecutive_losses_max = []
        
        np.random.seed(42) # Dla względnej spójności, ale można zdjąć
        for i in range(num_simulations):
            capital = starting_capital
            peak_capital = starting_capital
            max_dd = 0
            
            # Wektor losowych wyników, True = Win, False = Loss
            # Prawdopodobieństwo wygranej = win_prob
            results = np.random.random(mc_num_trades) < win_prob
            
            # Liczymy rzędy porażek
            losses = (~results).astype(int)
            # Find max consecutive 1s in losses (1 = pomylka)
            count = 0
            max_streak = 0
            for l in losses:
                if l == 1:
                    count += 1
                    if count > max_streak: max_streak = count
                else:
                    count = 0
            consecutive_losses_max.append(max_streak)
            
            # Obliczamy ścieżkę kapitału (Compound logic)
            # Wygrana dodaje % kapitału pomnożony przez RR
            # Przegrana odejmuje % kapitału
            for t in range(mc_num_trades):
                risk_amount = capital * risk_fraction
                if results[t]:
                    capital += risk_amount * rr
                else:
                    capital -= risk_amount
                
                equity_curves[i, t+1] = capital
                
                if capital > peak_capital: peak_capital = capital
                dd = (peak_capital - capital) / peak_capital
                if dd > max_dd: max_dd = dd
            
            max_drawdowns.append(max_dd)

        fig_mc = go.Figure()
        
        final_capitals = equity_curves[:, -1]
        best_idx = np.argmax(final_capitals)
        worst_idx = np.argmin(final_capitals)
        
        # Rysowanie tła (szare cienkie)
        for i in range(num_simulations):
            if i != best_idx and i != worst_idx:
                fig_mc.add_trace(go.Scatter(y=equity_curves[i, :], mode="lines", line=dict(color="rgba(100,100,120,0.15)", width=1), showlegend=False, hoverinfo="skip"))
        
        # Rysowanie najlepszego/najgorszego
        fig_mc.add_trace(go.Scatter(y=equity_curves[best_idx, :], mode="lines", line=dict(color="#00e676", width=3), name=f"Najlepszy scenariusz (+{((final_capitals[best_idx]/starting_capital)-1)*100:.0f}%)"))
        fig_mc.add_trace(go.Scatter(y=equity_curves[worst_idx, :], mode="lines", line=dict(color="#ff1744", width=3), name=f"Najgorszy scenariusz ({((final_capitals[worst_idx]/starting_capital)-1)*100:.0f}%)"))
        fig_mc.add_hline(y=starting_capital, line_color="#ffffff", line_dash="dash", annotation_text="KAPITAŁ STARTOWY", annotation_position="top left")
        
        fig_mc.update_layout(
            title=f"Krzywe Kapitału (Equity Curves) dla {num_simulations} traderów", height=450,
            xaxis=dict(title="Liczba Trejdów", gridcolor="#1c1c2e"), 
            yaxis=dict(title="Kapitał (PLN)", gridcolor="#1c1c2e"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=40, r=20, t=50, b=40),
            legend=dict(orientation="h", x=0, y=1.1)
        )
        st.plotly_chart(fig_mc, use_container_width=True)
        
        c3, c4, c5 = st.columns(3)
        c3.metric("Najgorszy napotkany Drawdown", f"{np.max(max_drawdowns)*100:.1f}%")
        c4.metric("Konto Zyskowne", f"{len([c for c in final_capitals if c > starting_capital])} na 100 traderów")
        c5.metric("MAX seria strat pod rząd!", f"{np.max(consecutive_losses_max)} strat", delta="Sila uderzyć w psychikę!", delta_color="inverse")
        
        st.markdown(f"""<div style='{CARD}'>
        Haczyk psychologii: Każda z tych osób miała <b>dokładnie TEN SAM matematyczny edge</b> rynkowy. Ale losowy rozkład zwycięstw i porażek sprawia, że jedni zarabiają potężne pieniądze już od początku (utwierdzając się w błędzie pychy - overconfidence bias), podczas gdy inni doświadczają wielkiej bolesnej serii strat przed pierwszym zarobkiem (rezygnując z działającego systemu po drodze). Odporność na serię strat to Twój główny cel bycia traderem.
        </div>""", unsafe_allow_html=True)

else:
    st.info("⬆️ Skonfiguruj parametry wyżej i kliknij przycisk, by wymodelować symulację Monte Carlo dla tysięcy transakcji.")

# ═══════════════════════════════════════════════════════════
# SEKCJA 4 — KOSZTY TRANSAKCYJNE (THE FRICTION THIEF)
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 💸 Sekcja 4 — Złodziej Niewidzialny: Friction (Spread + Prowizje)")

col_f1, col_f2 = st.columns([1, 1])

with col_f1:
    trades_per_day = st.slider("Średnia ilość transakcji Dziennie (Overtrading/Scalping?)", 1, 50, 5)
    cost_per_trade = st.slider("Średni średni koszt otwarcia/zamknięcia w PLN (Spready, Giełda)", 1.0, 50.0, 10.0, step=1.0)
    
    daily_cost = trades_per_day * cost_per_trade
    monthly_cost = daily_cost * 20 # ok 20 dni giełdowych w miesiącu
    yearly_cost = monthly_cost * 12
    
    pct_of_capital = (yearly_cost / starting_capital) * 100
    
    st.markdown(f"""<div style='background:#111;border:1px solid #ff1744;border-radius:10px;padding:16px;'>
    Rocznie same koszty giełdowe obciążają Twoje konto kwotą: <b style='color:#ff1744;font-size:22px'>{yearly_cost:,.0f} PLN</b><br><br>
    To stanowi <b style='color:#f39c12'>{pct_of_capital:.1f}%</b> Twojego kapitału początkowego (<b>{starting_capital} PLN</b>) oddane brokerowi całkowicie ZA NIC, niezależnie od faktu czy zgadujesz kierunek dobrze, czy źle. To potężne opóźnienie krzywej Wartości Oczekiwanej (EV). 
    </div>""", unsafe_allow_html=True)

with col_f2:
    years = ['Miesiąc 1', 'Miesiąc 3', 'Miesiąc 6', 'Miesiąc 12']
    costs = [monthly_cost, monthly_cost*3, monthly_cost*6, yearly_cost]
    remaining = [starting_capital - c for c in costs]
    
    fig_fric = go.Figure(data=[
        go.Bar(name='Kapitał Własny Mimo Kosztów', x=years, y=remaining, marker_color='#3498db'),
        go.Bar(name='Oddane jako Prowizja/Spread', x=years, y=costs, marker_color='#ff1744')
    ])
    fig_fric.update_layout(
        title="Drenaż Kapitału przez Koszty Transakcyjne (Gdybyś nie miał EV)", height=280,
        barmode='stack', paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=40, r=20, t=40, b=40),
        yaxis=dict(title="PLN", gridcolor="#1c1c2e")
    )
    if remaining[3] < 0:
         fig_fric.add_annotation(x='Miesiąc 12', y=starting_capital/2, text="BANKRUTUJESZ OD SAMEGO KLIKANIA", showarrow=False, font=dict(color="white", size=16, weight="bold"))
    
    st.plotly_chart(fig_fric, use_container_width=True)

st.markdown("---")
st.markdown("## 🛑 Sekcja 5 — Minimalny Cel Dzienny vs Danina Maklerska (Breakeven Analysis)")

col_b1, col_b2 = st.columns([2, 3])

with col_b1:
    st.markdown(f"""<div style='{CARD}'>
    <div style='{H3}'>Ile musisz wyrwać rynkowi, żeby on nie wyrwał Ciebie?</div>
    <p style='{NOTE}'>
    Założenie zysku dziennego to pułapka psychologiczna początkujących, ale spójrzmy prawdzie w oczy: jeśli stawiasz cel zysku, musisz doliczyć spread jako <b>ukryty podatek obrotowy</b>.
    </p></div>""", unsafe_allow_html=True)
    
    daily_target_pln = st.slider("Cel: Wymagany Czysty Zysk Dzienny Netto (PLN)", 10, 500, 50, step=10, key="dt_target")
    avg_trade_profit = (starting_capital * risk_fraction) * rr
    avg_trade_loss = starting_capital * risk_fraction
    
    # Koszty
    st.markdown("<b>Parametry Tradingowe:</b>", unsafe_allow_html=True)
    dt_trades_count = st.slider("Ile transakcji obstawiasz dziś zagrać?", 1, 50, 10, key="dt_count")
    # Łączny podatek maklerski dzisiaj
    dt_total_cost = dt_trades_count * cost_per_trade
    
    # Target brutto by wyjść na cel
    gross_target = daily_target_pln + dt_total_cost
    
    st.markdown(f"""
    <div style='border-left: 3px solid #ffea00; padding-left: 10px; margin-top: 15px;'>
        <div style='color:#ccc; font-size: 13px;'>Dzisiejszy haracz maklerski (Prowizje/Spread):</div>
        <div style='color:#ff1744; font-size: 20px; font-weight: 700;'>- {dt_total_cost:,.0f} PLN</div>
    </div>
    <div style='border-left: 3px solid #3498db; padding-left: 10px; margin-top: 10px;'>
        <div style='color:#ccc; font-size: 13px;'>Ryneczku, muszę dziś zarobić BRUTTO aż:</div>
        <div style='color:#00e676; font-size: 20px; font-weight: 700;'>{gross_target:,.0f} PLN</div>
    </div>
    """, unsafe_allow_html=True)

with col_b2:
    # Symulacja p-stwa osiągnięcia celu vs ilość trades
    # Pytanie: przy zadanym RR i risk per trade, i przy założonej prowizji, jaki Win Rate POTRZEBUJESZ żeby ugrać gross_target w N transakcjach?
    # W * avg_profit - L * avg_loss = gross_target
    # W + L = N -> L = N - W
    # W * P - (N-W) * L = T
    # W * P - N*L + W*L = T
    # W*(P+L) = T + N*L
    # W_req = (T + N*L) / (P+L)
    
    val_P = avg_trade_profit
    val_L = avg_trade_loss
    val_N = dt_trades_count
    val_T = gross_target
    
    req_wins = (val_T + val_N * val_L) / (val_P + val_L)
    req_wr_pct = (req_wins / val_N) * 100
    
    # Wykres wymagany Win Rate w funkcji ilości zagrań aby OBRONIĆ target netto
    t_range = np.arange(1, 51)
    req_wr_array = []
    
    for nn in t_range:
        tt = daily_target_pln + (nn * cost_per_trade)
        ww = (tt + nn * val_L) / (val_P + val_L)
        wr = (ww / nn) * 100
        req_wr_array.append(wr)
        
    fig_req = go.Figure()
    
    # Obszar "niemożliwego"
    fig_req.add_hrect(y0=100, y1=max(120, max(req_wr_array)+10), fillcolor="rgba(255,23,68,0.1)", line_width=0)
    fig_req.add_hline(y=100, line_dash="dash", line_color="#ff1744", annotation_text="Kłamstwo 100% skuteczności")
    
    # Linia wymaganej skuteczności
    fig_req.add_trace(go.Scatter(x=t_range, y=req_wr_array, mode="lines", 
        line=dict(color="#3498db", width=3), name="Wymagany Win Rate (%)",
        fill="tozeroy", fillcolor="rgba(52,152,219,0.1)"))
        
    # Punkt obecny
    fig_req.add_trace(go.Scatter(x=[val_N], y=[req_wr_pct], mode="markers", 
        marker=dict(size=14, color="#ffea00", line=dict(color="white",width=2)), name="Twój Plan Dnia"))
        
    fig_req.update_layout(
         title=f"Wymagany Win Rate żeby zarobić {daily_target_pln} PLN netto", height=350,
         xaxis=dict(title="Ilość zagrań dzisiaj (Overtrading)", gridcolor="#1c1c2e"), 
         yaxis=dict(title="Wymagana Skuteczność (Win Rate %)", gridcolor="#1c1c2e", range=[0, min(120, max(req_wr_array)+10)]),
         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=40, r=20, t=40, b=40)
    )
    
    if req_wr_pct > 100:
        fig_req.add_annotation(x=val_N, y=105, text="MATEMATYCZNIE NIEMOŻLIWE", showarrow=False, font=dict(color="#ff1744", size=14, weight="bold"))
    
    st.plotly_chart(fig_req, use_container_width=True)
    
    if req_wr_pct > 100:
        st.error(f"❌ Twój system oparty na zysku 1:{rr} ryzykując {risk_per_trade_pct}% kapitału w połączeniu z prowizjami {cost_per_trade} PLN gwarantuje, że nie możesz zrealizować tego celu w {val_N} transakcjach. Musiałbyś mieć ponad 100% skuteczności, co matematycznie odrzuca setup. Zwiększ zyskowność trejdu (Risk/Reward) albo ogranicz docelowy Target na dzień.")
    elif req_wr_pct > user_wr:
         st.warning(f"⚠️ Uwaga! Aby to dowieźć, musisz zagrać dziś ze skutecznością {req_wr_pct:.1f}%. W Twoim długoterminowym planie wpisałeś, że potrafisz utrzymać {user_wr}%. To sygnał do zaprzestania overtradingu. Spready Cię miażdżą.")
    else:
         st.success(f"✅ Realistyczne założenie! Wystarczy Ci {req_wr_pct:.1f}% skutecznych zagrań dzisiaj, aby uwzględniając prowizje wyjść na docelowe {daily_target_pln} PLN netto.")

# ═══════════════════════════════════════════════════════════
# SEKCJA 6 — EKONOFIZYKA: FRAKTALE I ROUGH VOLATILITY
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🌀 Sekcja 6 — Ekonofizyka: Rough Volatility & Fractional Brownian Motion")

col_rv1, col_rv2 = st.columns([1, 1])

with col_rv1:
    h_exponent = st.slider("Wykładnik Hursta (Hurst Exponent - H)", 0.1, 0.9, 0.5, 0.05, key="hurst_h")
    
    st.markdown(f"""
    <div style='{CARD}'>
    <b>Krótka pamięć rynku:</b> Mniej znaczy "bardziej szorstko". <br>
    <ul>
    <li><b>H = 0.5:</b> Klasyczny Random Walk (brak pamięci, Gaus). Szum Billa Browna.</li>
    <li><b>H < 0.5 (np. 0.2):</b> Mean-Reverting. Szorstka zmienność (Rough Volatility, powrót do średniej - idealne do grid tradingu/scalpingu).</li>
    <li><b>H > 0.5 (np. 0.8):</b> Persistent. Silne treny, gładkie momentum (podążanie za rynkiem, breakout).</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple FBM simulation via Cholesky decomposition of covariance matrix
    # for rendering speed we limit to N=250
    np.random.seed(42)
    N = 250
    t = np.linspace(0, 1, N)
    
    # Covariance matrix for FBM: C(t,s) = 0.5 * (t^(2H) + s^(2H) - |t-s|^(2H))
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            C[i, j] = 0.5 * (t[i]**(2*h_exponent) + t[j]**(2*h_exponent) - abs(t[i] - t[j])**(2*h_exponent))
    
    # Small nugget for numerical stability
    C += np.eye(N) * 1e-6
    try:
        L = np.linalg.cholesky(C)
        Z = np.random.randn(N)
        fbm_path = L @ Z
    except np.linalg.LinAlgError:
        # Fallback if matrix is not positive definite
        fbm_path = np.cumsum(np.random.randn(N))
    
    fig_fbm = go.Figure()
    fig_fbm.add_trace(go.Scatter(y=fbm_path, mode="lines", line=dict(color="#00e676" if h_exponent > 0.5 else ("#a855f7" if h_exponent < 0.5 else "#3498db"), width=2)))
    
    fig_fbm.update_layout(
        title=f"Symulacja Ceny (Fractional Brownian Motion, H={h_exponent})",
        height=350,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(title="Czas", gridcolor="#1c1c2e", showticklabels=False),
        yaxis=dict(title="Cena", gridcolor="#1c1c2e", showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_fbm, use_container_width=True)

with col_rv2:
    # Fractal Dimension vs Hurst
    # D = 2 - H for 1D paths
    fractal_dim = 2.0 - h_exponent
    
    # Let's plot Hurst vs Fractal Dimension mapping
    h_arr = np.linspace(0.1, 0.9, 100)
    d_arr = 2.0 - h_arr
    
    fig_fd = go.Figure()
    fig_fd.add_trace(go.Scatter(x=h_arr, y=d_arr, mode="lines", line=dict(color="#3498db", width=3, dash="dash"), showlegend=False))
    fig_fd.add_trace(go.Scatter(x=[h_exponent], y=[fractal_dim], mode="markers", marker=dict(color="#ff1744", size=16, line=dict(width=2, color="white")), name="Obecny stan"))
    
    fig_fd.update_layout(
         title=f"Wymiar Fraktalny D = {fractal_dim:.2f}",
         height=350,
         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
         font=dict(color="white"),
         xaxis=dict(title="Wykładnik Hursta (H)", gridcolor="#1c1c2e"),
         yaxis=dict(title="Wymiar Fraktalny (D)", gridcolor="#1c1c2e"),
         margin=dict(l=40, r=20, t=40, b=40),
         showlegend=False
    )
    fig_fd.add_vline(x=0.5, line_dash="solid", line_color="#555", annotation_text="Random Walk")
    st.plotly_chart(fig_fd, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# SEKCJA 7 — MIKROSTRUKTURA: ORDER BOOK HEATMAP
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🧊 Sekcja 7 — Mikrostruktura: Order Book Heatmap / Liquidity Map")
st.markdown("<p style='color:#bbb;font-size:14px'>Wizualizacja płynności na różnych poziomach cenowych w czasie (Limit Order Book). Cena przyciągana jest jak magnes przez strefy o wysokiej płynności.</p>", unsafe_allow_html=True)

# Generate synthetic heatmap data
np.random.seed(1)
time_steps = 100
price_levels = 50
liquidity = np.zeros((price_levels, time_steps))

# Add some dynamic price path
curr_p = 25
path = [curr_p]
for i in range(1, time_steps):
    step = np.random.choice([-1, 0, 1], p=[0.4, 0.2, 0.4])
    curr_p = max(5, min(45, curr_p + step))
    path.append(curr_p)

# Generate liquidity pools
for p_idx in range(price_levels):
    # Base noise
    liquidity[p_idx, :] = np.random.uniform(0, 10, time_steps)
    # Add heavy liquidity at round numbers
    if p_idx in [10, 20, 30, 40]:
        liquidity[p_idx, :] += np.random.uniform(50, 80, time_steps)
        # Maybe liquidity gets eaten
        for t in range(time_steps):
            if abs(path[t] - p_idx) <= 1:
                liquidity[p_idx, t] = np.random.uniform(0, 10)  # Eaten liquidity

fig_ob = go.Figure()

# Heatmap of liquidity
fig_ob.add_trace(go.Heatmap(
    z=liquidity,
    colorscale="Plasma",
    zmin=0, zmax=80,
    showscale=True,
    colorbar=dict(title="Volume", tickfont=dict(color="white"))
))

# Overlay price path
fig_ob.add_trace(go.Scatter(
    x=list(range(time_steps)),
    y=path,
    mode="lines+markers",
    line=dict(color="#00ff88", width=3),
    marker=dict(size=4, color="#ffffff"),
    name="Mid Price"
))

fig_ob.update_layout(
    title="Heatmapa Płynności Księgi Zleceń (Synthetic LOB)",
    height=450,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white"),
    xaxis=dict(title="Czas (Ticks)", gridcolor="#1c1c2e"),
    yaxis=dict(title="Poziom Cenowy", gridcolor="#1c1c2e"),
    margin=dict(l=40, r=20, t=40, b=20)
)
st.plotly_chart(fig_ob, use_container_width=True)


st.markdown("---")
st.markdown(f"""<div style='text-align:center;color:#2a2a3a;font-size:11px;padding:12px'>
Day Trading OS v2.0 · Quant Platform Ecosystem · Algorytmy Oparte Na:<br>
Statystyka Rynków Losowych, Twierdzenie Bernoulliego, Modele Monte Carlo, <br>
Risk-of-Ruin Math, Friction Analysis, Ekonofizyka (Rough Volatility / FBM), <br>
Liquidity Mapping & Limit Order Book Heatmaps.
</div>""", unsafe_allow_html=True)

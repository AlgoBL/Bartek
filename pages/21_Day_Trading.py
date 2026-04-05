import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from modules.styling import apply_styling, scicard
from modules.i18n import t

st.markdown(apply_styling(), unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align:center;margin-bottom:4px'>📈 Day Trading — Niewidzialna Matematyka</h1>
<p style='text-align:center;color:#6b7280;font-size:14px;margin-bottom:24px'>
System Wartości Oczekiwanej · Ryzyko Ruiny · Law of Large Numbers · Mikrostruktura kosztów
</p>
""", unsafe_allow_html=True)

# UI-3 FIX: Dodano font-family: 'Inter', sans-serif do wszystkich inline styli
# by zapewnić spójność typograficzną z resztą dashboardu (zdefiniowanego w apply_styling())
CARD = "background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px 20px;margin-bottom:8px;font-family:'Inter',sans-serif"
H3 = "color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px;font-family:'Inter',sans-serif"
NOTE = "color:#6b7280;font-size:12px;line-height:1.6;font-family:'Inter',sans-serif"

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
st.markdown("## ⚖️ Sekcja 8 — Kelly Criterion: Optymalny Rozmiar Pozycji")
st.markdown("<p style='color:#bbb;font-size:14px'>Ile dokładnie kapitału powinieneś zaryzykować mając określoną przewagę (edge)? Kelly Criterion oblicza optymalną frakcję kapitału, która maksymalizuje długoterminowy wzrost, chroniąc przed ryzykiem ruiny.</p>", unsafe_allow_html=True)

col_k1, col_k2 = st.columns([1, 1])

with col_k1:
    st.markdown(f"""
    <div style='{CARD}'>
    <div style='{H3}'>Zmienne Kelly'ego</div>
    <ul style='color:#bbb;font-size:14px'>
    <li><b>W (Win Rate):</b> Prawdopodobieństwo wygranej</li>
    <li><b>R (Risk:Reward):</b> Ile zarabiasz względem ryzyka</li>
    <li><b>f* (Kelly %):</b> Optymalny % kapitału do zaryzykowania: <br><br>
    <b style="color:#00e676;font-size:18px;">f* = W - ((1 - W) / R)</b></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    k_wr = st.slider("Kelly Win Rate (%)", 10, 90, 40, key="kelly_wr") / 100
    k_rr = st.slider("Kelly Risk:Reward (1:X)", 0.5, 5.0, 2.0, 0.1, key="kelly_rr")
    
    kelly_f = k_wr - ((1 - k_wr) / k_rr)
    kelly_pct = kelly_f * 100
    
    # Fractional Kelly
    frac_kelly_slider = st.slider("Mnożnik Kelly'ego (Fractional Kelly)", 0.1, 2.0, 0.5, 0.1, help="Wielu profesjonalistów używa Half-Kelly (0.5), aby drastycznie zmniejszyć zmienność kapitału przy zachowaniu dużej części zysku.")
    applied_kelly = kelly_pct * frac_kelly_slider
    
    if kelly_pct <= 0:
        st.error("System nie ma przewagi! Według kryterium Kelly'ego optymalna wielkość transakcji wynosi 0 lub mniej. Nie graj tego.")
    else:
        st.markdown(f"""
        <div style='border-left: 3px solid #00e676; padding-left: 10px; margin-top: 15px;'>
            <div style='color:#ccc; font-size: 13px;'>Full Kelly (Maksymalny agresywny wzrost):</div>
            <div style='color:#00e676; font-size: 20px; font-weight: 700;'>{kelly_pct:.2f}% kapitału</div>
        </div>
        <div style='border-left: 3px solid #3498db; padding-left: 10px; margin-top: 10px;'>
            <div style='color:#ccc; font-size: 13px;'>Aplikowany Kelly ({frac_kelly_slider}x):</div>
            <div style='color:#3498db; font-size: 20px; font-weight: 700;'>{applied_kelly:.2f}% kapitału</div>
        </div>
        """, unsafe_allow_html=True)

with col_k2:
    # Symulacja Kelly vs Overbetting
    bet_mults = np.linspace(0.1, 2.5, 50)
    growth_rates = []
    
    for mult in bet_mults:
        f = kelly_f * mult
        if f >= 1.0 or f <= 0:
            growth_rates.append(np.nan)
        else:
            g = k_wr * np.log(1 + f * k_rr) + (1 - k_wr) * np.log(1 - f)
            growth_rates.append(g)
            
    fig_kelly = go.Figure()
    fig_kelly.add_trace(go.Scatter(x=bet_mults, y=growth_rates, mode="lines", line=dict(color="#3498db", width=3), name="Krzywa Wzrostu"))
    
    if kelly_pct > 0 and kelly_f < 1.0:
        fig_kelly.add_trace(go.Scatter(x=[1], y=[k_wr * np.log(1 + kelly_f * k_rr) + (1 - k_wr) * np.log(1 - kelly_f)],
                         mode="markers", marker=dict(size=12, color="#00e676"), name="Full Kelly (Max Wzrost)"))
        
    if frac_kelly_slider > 0 and (kelly_f * frac_kelly_slider) < 1.0 and applied_kelly > 0:
         fig_kelly.add_trace(go.Scatter(x=[frac_kelly_slider], y=[k_wr * np.log(1 + (kelly_f*frac_kelly_slider) * k_rr) + (1 - k_wr) * np.log(1 - (kelly_f*frac_kelly_slider))],
                         mode="markers", marker=dict(size=12, color="#f39c12"), name=f"{frac_kelly_slider}x Kelly"))
    
    fig_kelly.add_vrect(x0=2.0, x1=2.5, fillcolor="rgba(255,23,68,0.2)", layer="below", line_width=0, annotation_text="KAPITAŁ PŁONIE (Ryzyko Ruiny)", annotation_position="top left")
    
    fig_kelly.update_layout(
        title="Spadek zysków z chciwości (Over-betting)", height=350,
        xaxis=dict(title="Mnożnik Kelly'ego (1.0 = Max, >2.0 = Gwarantowana Strata)", gridcolor="#1c1c2e"),
        yaxis=dict(title="Geometryczna Stopa Wzrostu", gridcolor="#1c1c2e", showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_kelly, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# SEKCJA 9 — PROSPECT THEORY
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🧠 Sekcja 9 — Prospect Theory: Psychologia Strat (Kahneman & Tversky)")
st.markdown("<p style='color:#bbb;font-size:14px'>Asymetria odczuwania bólu straty względem radości zysku powoduje Disposition Effect (awersja do zamykania strat i zbyt szybkie realizowanie zysków).</p>", unsafe_allow_html=True)

col_pt1, col_pt2 = st.columns([1, 1])

with col_pt1:
    lambda_loss = st.slider("Współczynnik awersji do straty (λ)", 1.0, 5.0, 2.25, 0.05, help="Daniel Kahneman odkrył, że średnio ludzie odczuwają startę 2.25 raza mocniej niż równo warty zysk.")
    alpha_gain = st.slider("Krzywizna użyteczności zysków (α)", 0.2, 1.0, 0.88, 0.02)
    
    x_pt = np.linspace(-100, 100, 1000)
    y_pt = np.where(x_pt >= 0, x_pt**alpha_gain, -lambda_loss * (-x_pt)**alpha_gain)
    
    fig_pt = go.Figure()
    fig_pt.add_trace(go.Scatter(x=x_pt, y=y_pt, mode="lines", line=dict(color="#a855f7", width=3)))
    fig_pt.add_trace(go.Scatter(x=[50, -50], y=[50**alpha_gain, -lambda_loss * (50)**alpha_gain], mode="markers+text", 
                     marker=dict(size=10, color=["#00e676", "#ff1744"]),
                     text=["Radość z +50 PLN", "Ból z -50 PLN"], textposition="top center"))
    
    fig_pt.update_layout(
        title="Krzywa Wartości (Prospect Theory)", height=350,
        xaxis=dict(title="Rzeczywisty Wynik Transakcji (Zysk/Strata)", gridcolor="#1c1c2e", zerolinecolor="#ffffff"),
        yaxis=dict(title="Odczuwana 'Wartość' (Emocje)", gridcolor="#1c1c2e", zerolinecolor="#ffffff", showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_pt, use_container_width=True)

with col_pt2:
    st.markdown(f"""
    <div style='{CARD}'>
    <div style='{H3}'>Disposition Effect w liczbach</div>
    <p style='{NOTE}'>
    Ponieważ ból ze straty skaluje się przez wielokrotność <b>{lambda_loss}x</b>, mózg oszukuje Cię: <br>
    <ul>
    <li>Kiedy masz stratę: "jeszcze odbije" (szukasz ryzyka by uniknąć pewnej straty) -> <b>Risk-seeking in losses</b></li>
    <li>Kiedy masz zysk: "lepiej zrealizować, bo zabiorą" (boisz się stracić to co masz) -> <b>Risk-averse in gains</b></li>
    </ul>
    To niszczy każdy matematyczny edge, zmieniając Twój system w <i>High Win Rate / Negative RR</i>.
    </p></div>
    """, unsafe_allow_html=True)
    st.info("Zarządzanie ryzykiem to tak naprawdę zarządzanie emocjami. Trading mechaniczny (algorytmy) wyłącza Prospect Theory.")

# ═══════════════════════════════════════════════════════════
# SEKCJA 10 — REGIME DETECTION
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🌤️ Sekcja 10 — Regime Detection: Modele Markowa (Markov Switching)")
st.markdown("<p style='color:#bbb;font-size:14px'>Rynek nie ma jednego stanu. Przełącza się (często gwałtownie) między różnymi reżimami (Trend a Szum/Konsolidacja).</p>", unsafe_allow_html=True)

col_rd1, col_rd2 = st.columns([1, 1])

with col_rd1:
    p_trend_to_trend = st.slider("Prawdopodobieństwo pozostania w Trendzie", 0.1, 0.99, 0.95, 0.01)
    p_chop_to_chop = st.slider("Prawdopodobieństwo pozostania w Konsolidacji", 0.1, 0.99, 0.90, 0.01)
    
    st.markdown(f"""
    <div style='{CARD}'>
    <div style='{H3}'>Macierz Przejścia (Transition Matrix)</div>
    <ul>
    <li>P(Trend | Trend) = {p_trend_to_trend:.2f}</li>
    <li>P(Chop | Trend) = {1 - p_trend_to_trend:.2f}</li>
    <li>P(Chop | Chop) = {p_chop_to_chop:.2f}</li>
    <li>P(Trend | Chop) = {1 - p_chop_to_chop:.2f}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

with col_rd2:
    np.random.seed(42)
    n_days = 200
    states = np.zeros(n_days, dtype=int)
    states[0] = 0
    for i in range(1, n_days):
        if states[i-1] == 0:
            states[i] = 0 if np.random.rand() < p_trend_to_trend else 1
        else:
            states[i] = 1 if np.random.rand() < p_chop_to_chop else 0
            
    returns = np.zeros(n_days)
    for i in range(n_days):
        if states[i] == 0:
            returns[i] = np.random.normal(loc=0.002, scale=0.01) # Trend
        else:
            returns[i] = np.random.normal(loc=-0.001, scale=0.03) # Chop/Crash
            
    price = np.cumsum(returns)
    
    fig_hmm = go.Figure()
    
    change_idx = np.where(states[:-1] != states[1:])[0]
    start_idx = 0
    for idx in list(change_idx) + [n_days-1]:
        s = states[start_idx]
        color = "rgba(0, 230, 118, 0.2)" if s == 0 else "rgba(255, 23, 68, 0.2)"
        fig_hmm.add_vrect(x0=start_idx, x1=idx, fillcolor=color, layer="below", line_width=0, opacity=0.5)
        start_idx = idx + 1
        
    fig_hmm.add_trace(go.Scatter(y=price, mode="lines", line=dict(color="#ffffff", width=2), name="Cena Skumulowana"))
    
    fig_hmm.update_layout(
        title="Symulacja Przełączeń Reżimowych (Hidden Markov Model)", height=350,
        xaxis=dict(title="Czas", gridcolor="#1c1c2e"),
        yaxis=dict(title="Cena", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    st.plotly_chart(fig_hmm, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# SEKCJA 11 — STATYSTYCZNA ISTOTNOŚĆ (MARCOS LOPEZ DE PRADO)
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 📊 Sekcja 11 — Statystyczna Istotność: Czy Twój Edge jest prawdziwy?")
st.markdown("<p style='color:#bbb;font-size:14px'>Większość traderów przerywa testowanie strategii gdy tylko zobaczą zysk na małej próbce. Statystyka mówi: potrzebujesz dowodu. Według Lopeza de Prado, Sharpe ratio wymaga weryfikacji przez t-statistic przy użyciu odpowiedniej liczby prób.</p>", unsafe_allow_html=True)

col_ss1, col_ss2 = st.columns([1, 1])

with col_ss1:
    st.markdown(f"""
    <div style='{CARD}'>
    <div style='{H3}'>Test Hipotezy Edge'u</div>
    <p style='{NOTE}'>Wpisz wyniki swojego historycznego backtestu lub journala, żeby sprawdzić, czy zysk mógł być czystym przypadkiem (p-value).</p>
    </div>
    """, unsafe_allow_html=True)
    
    n_trades = st.number_input("Ilość zagrań w historii (N)", min_value=10, max_value=5000, value=50, step=10)
    sharpe_est = st.slider("Szacowane Sharpe Ratio Strategii (na pojedynczy trejd)", 0.0, 3.0, 0.5, 0.1)
    
    # Simple t-stat for positive mean return (assuming SR = mean / std)
    # t = SR * sqrt(N)
    t_stat = sharpe_est * np.sqrt(n_trades)
    
    # p-value na 1 tail
    from scipy.stats import t
    p_val = 1 - t.cdf(t_stat, df=n_trades-1)
    
    # Required N for significance (np. t=3.0) -> N_req = (3.0 / SR)^2
    if sharpe_est > 0:
        req_n = (3.0 / sharpe_est)**2
    else:
        req_n = float('inf')

    if t_stat > 3.0:
        st.success(f"**T-Statistic: {t_stat:.2f}** | P-Value: {p_val:.5f} \n\n✅ MOCNY DOWÓD! Wyniki są wysoce statystycznie istotne. Bardzo mała szansa, że ten wynik to dzieło przypadku.")
    elif t_stat > 2.0:
        st.info(f"**T-Statistic: {t_stat:.2f}** | P-Value: {p_val:.4f} \n\n⚠️ OBIECUJĄCE. Przekroczono próg ufności 95%, ale dla tradingu sugerowany jest próg (t > 3) ze względu na grube ogony (fat tails). Zrób więcej trejdów.")
    else:
        st.error(f"**T-Statistic: {t_stat:.2f}** | P-Value: {p_val:.3f} \n\n❌ Za mało danych! Wynik nie jest istotny statystycznie. Równie dobrze mógł to być rzut monetą. Osiągnięcie ufności wymagałoby historii {int(req_n) if req_n != float('inf') else '∞'} zagrań przy tym SR.")

with col_ss2:
    # Bootstrap Simulator representation
    st.markdown(f"""
    <div style='{CARD}'>
    <div style='{H3}'>Krzywa Istotności (Significance t-stat > 3)</div>
    <p style='{NOTE}'>Zielona strefa to strefa 'Prawdziwego Edge'u'. Wszystko poniżej mogło być wygenerowane przez małpę rzucającą lotkami.</p>
    </div>
    """, unsafe_allow_html=True)
    
    n_range = np.linspace(10, max(200, n_trades*2), 100)
    t_curve = sharpe_est * np.sqrt(n_range)
    
    fig_ss = go.Figure()
    fig_ss.add_trace(go.Scatter(x=n_range, y=t_curve, mode="lines", line=dict(color="#3498db", width=3), name="Twój t-statistic"))
    
    # Point
    fig_ss.add_trace(go.Scatter(x=[n_trades], y=[t_stat], mode="markers", marker=dict(size=14, color="#ff1744" if t_stat < 3 else "#00e676", line=dict(color="white",width=2)), name="Twój wynik"))
    
    # Threshold line
    fig_ss.add_hrect(y0=3.0, y1=max(5.0, t_stat+1), fillcolor="rgba(0,230,118,0.1)", line_width=0)
    fig_ss.add_hline(y=3.0, line_dash="dash", line_color="#00e676", annotation_text="t=3.0 (Złoty Standard)")
    
    fig_ss.update_layout(
        title="Przyrastanie Statystycznej Pewności Czasem", height=300,
        xaxis=dict(title="Ilość Zagrań w Historii", gridcolor="#1c1c2e"),
        yaxis=dict(title="Statystyka t (t-stat)", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_ss, use_container_width=True)

# ═══════════════════════════════════════════════════════════
# SEKCJA 12 — ZMIENNOŚĆ (GARCH & OPCJE)
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🌪️ Sekcja 12 — Zmienność jako Zasób: Modele GARCH i Klastering")
st.markdown("<p style='color:#bbb;font-size:14px'>Wielkie ruchy podążają za wielkimi ruchami, a cisza przepowiada ciszę. Zrozumienie Modelu GARCH (Robert Engle, Nobel 2003) i grupowania zmienności jest kluczem do przetrwania na rynku.</p>", unsafe_allow_html=True)

col_g1, col_g2 = st.columns([1, 1])

with col_g1:
    alpha_g = st.slider("Szok rynkowy - α (Reakcja na nowe wydarzenia)", 0.01, 0.40, 0.15, 0.01, help="Jak mocno 'news' wpływa na dzisiejszą zmienność")
    beta_g = st.slider("Pamięć rynkowa - β (Długotrwałość zmienności)", 0.50, 0.95, 0.80, 0.01, help="Jak długo rynek 'pamięta' wczorajszą zmienność (wysokie β oznacza długie okresy burz lub spokoju)")
    
    if alpha_g + beta_g >= 1.0:
        st.error("Model jest niestabilny (α + β >= 1). Zmienność eksploduje w nieskończoność matematyczną!")
    else:
        # GARCH(1,1) Simulation
        np.random.seed(42)
        n_sim = 250
        omega = 0.0001
        
        returns_g = np.zeros(n_sim)
        sigma2 = np.zeros(n_sim)
        
        sigma2[0] = omega / (1 - alpha_g - beta_g)
        returns_g[0] = np.random.normal(0, np.sqrt(sigma2[0]))
        
        for t_step in range(1, n_sim):
            sigma2[t_step] = omega + alpha_g * (returns_g[t_step-1]**2) + beta_g * sigma2[t_step-1]
            returns_g[t_step] = np.random.normal(0, np.sqrt(sigma2[t_step]))
            
        fig_garch1 = go.Figure()
        fig_garch1.add_trace(go.Bar(x=np.arange(n_sim), y=returns_g, marker_color="#3498db", name="Zwroty Cena"))
        fig_garch1.add_trace(go.Scatter(x=np.arange(n_sim), y=1.96*np.sqrt(sigma2), mode="lines", line=dict(color="#ff1744", width=2), name="Górne Pasmo Vol"))
        fig_garch1.add_trace(go.Scatter(x=np.arange(n_sim), y=-1.96*np.sqrt(sigma2), mode="lines", line=dict(color="#ff1744", width=2), name="Dolne Pasmo Vol"))
        
        fig_garch1.update_layout(
            title="Symulacja Volatility Clustering (GARCH 1,1)", height=300,
            xaxis=dict(title="Czas (Dni)", gridcolor="#1c1c2e"),
            yaxis=dict(title="Zwroty (%)", gridcolor="#1c1c2e"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_garch1, use_container_width=True)

with col_g2:
    st.markdown(f"""
    <div style='{CARD}'>
    <div style='{H3}'>Volatility Risk Premium</div>
    <p style='{NOTE}'>
    Implikowana Zmienność (IV — podyktowana przez opcje, np. VIX) jest zazwyczaj wyższa niż Zrealizowana Zmienność (RV — faktyczne ruchy ceny). Dlaczego? Traderzy płacą "premię ubezpieczeniową" za zabezpieczenie od krachu.<br><br>
    <b>Wniosek rynkowy:</b> Zamiast kupować, sprzedawanie opcji ma statystycznie pozytywną Wartość Oczekiwaną, ale charakteryzuje się ogromnym ryzykiem ruiny podczas tzw. <i>Black Swans</i>.
    </p></div>
    """, unsafe_allow_html=True)
    
    st.markdown("<b>Wiedza Opcji (The Greeks):</b>", unsafe_allow_html=True)
    c_g1, c_g2 = st.columns(2)
    with c_g1:
        st.markdown("<span style='color:#a855f7'><b>Δ Delta:</b></span> Kierunek ceny", unsafe_allow_html=True)
        st.markdown("<span style='color:#3498db'><b>Γ Gamma:</b></span> Prędkość przyspieszenia", unsafe_allow_html=True)
    with c_g2:
        st.markdown("<span style='color:#ff1744'><b>Θ Theta:</b></span> Wypalanie czasu (złodziej kwantowy)", unsafe_allow_html=True)
        st.markdown("<span style='color:#00e676'><b>ν Vega:</b></span> Czułość na zmienność", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SEKCJA 13 — WFO (WALK-FORWARD ANALYSIS)
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## ⏱️ Sekcja 13 — Backtest Bias i Walk-Forward Analysis")
st.markdown("<p style='color:#bbb;font-size:14px'>Największe kłamstwo quantów: Backtest zawsze wygląda wspaniale, na którym traci się pieniądze w realnym świecie. Dopasowanie modelu do przeszłości (Overfitting) prowadzi do katastrofy The Reality Check (White 2000).</p>", unsafe_allow_html=True)

col_wf1, col_wf2 = st.columns([7, 5])

with col_wf1:
    fig_wfo = go.Figure()

    # Rysowanie okien Walk-Forward (wizualizacja)
    windows = 5
    colors_is = "rgba(52, 152, 219, 0.3)"
    colors_oos = "rgba(255, 23, 68, 0.4)"
    
    y_pos = np.arange(windows, 0, -1)
    
    for i in range(windows):
        # In-sample window
        start_is = i * 20
        end_is = start_is + 50
        # Out-of-sample window
        start_oos = end_is
        end_oos = start_oos + 15
        
        # Rect for IS
        fig_wfo.add_shape(type="rect", x0=start_is, y0=y_pos[i]-0.3, x1=end_is, y1=y_pos[i]+0.3, fillcolor=colors_is, line=dict(color="#3498db", width=2))
        fig_wfo.add_annotation(x=(start_is+end_is)/2, y=y_pos[i], text="In-Sample (Train)", showarrow=False, font=dict(color="white"))
        
        # Rect for OOS
        fig_wfo.add_shape(type="rect", x0=start_oos, y0=y_pos[i]-0.3, x1=end_oos, y1=y_pos[i]+0.3, fillcolor=colors_oos, line=dict(color="#ff1744", width=2))
        fig_wfo.add_annotation(x=(start_oos+end_oos)/2, y=y_pos[i], text="OOS", showarrow=False, font=dict(color="white"))
        
    fig_wfo.update_layout(
        title="Walk-Forward Optimization (WFO) Matrix", height=350,
        xaxis=dict(title="Czas / Dane Historyczne", showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(title="Rozkład Modeli (Walks)", showgrid=False, zeroline=False, showticklabels=False, range=[0, windows+1]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_wfo, use_container_width=True)

with col_wf2:
    st.markdown(f"""
    <div style='background:#1a1c28;border-left:4px solid #f39c12;padding:12px;margin-bottom:12px'>
    <b>Deflation Factor</b><br>
    Zaawansowane modele (ML, Quant) po przetestowaniu tysięcy kombinacji parametrów zawyżają rzeczywiste osiągi (Sharpe). Lopez de Prado zaleca stosowanie "Deflated Sharpe Ratio (DSR)", który redukuje wynik w zależności od liczby przeprowadzonych prób (Multiple Testing Framework).
    </div>
    """, unsafe_allow_html=True)
    
    ins_sharpe = st.number_input("In-Sample Sharpe Ratio (na danych testowych)", 0.5, 5.0, 2.0, 0.1)
    decay_pct = st.slider("Zakładany Decay na nowych danych (%)", 10, 80, 40, 5)
    oos_sharpe = ins_sharpe * (1 - decay_pct/100)
    
    st.markdown(f"""
    Sharpe po rygorystycznym WFO i urealnieniu kosztów: 
    <b style="color:{'#00e676' if oos_sharpe > 1.0 else '#ff1744'};font-size:24px;">{oos_sharpe:.2f}</b>
    """)
    if oos_sharpe < 1.0:
        st.error("System prawdopodobnie zapadnie się pod sobą (wpadnie w stratę) z powodu ukrytego szumu na rynku po wyjściu z backtestu.")

# ═══════════════════════════════════════════════════════════
# SEKCJA 14 — GRA SKOŃCZONA VS NIESKOŃCZONA
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## ♾️ Sekcja 14 — Psychologia: Gra Skończona vs Nieskończona (Nassim Taleb)")
st.markdown("<p style='color:#bbb;font-size:14px'>Wybuchy furii, overtrading i 'mieszanie' w algorytmach wynikają z grania w Grę Skończoną (chcę zrobić +1000 PLN dzisiaj). Profesjonaliści grają w Grę Nieskończoną (chcę tu być za 10 lat). Przetrwanie nadrzędne nad zyskiem czasowym (Ergodyczność).</p>", unsafe_allow_html=True)

col_ig1, col_ig2 = st.columns([1, 1])

with col_ig1:
    hr_time = st.slider("Horyzont czasowy (Twoje Dni na Rynku)", 100, 2500, 500, 100)
    profit_per_day = st.slider("Średni zysk dzienny (Edge mierzony w długim terminie)", 1.0, 50.0, 5.0, 1.0)
    ruin_prob = st.slider("Szansa na wyzerowanie konta / Rezygnację per miesiąc (%)", 0.1, 5.0, 1.0, 0.1)
    
    # Symulacja przetrwania (Survival function)
    # P_survive(t) = (1 - ruin_prob_daily)^t
    ruin_daily = (ruin_prob / 100) / 20 # ok. 20 dni w mies
    t_days = np.arange(1, hr_time+1)
    p_survival = (1 - ruin_daily)**t_days
    
    # Expected Value considering survival (EV = value * P(survival))
    base_value = t_days * profit_per_day
    ergodic_value = base_value * p_survival
    
    fig_ig = go.Figure()
    fig_ig.add_trace(go.Scatter(x=t_days, y=base_value, mode="lines", line=dict(color="#3498db", width=2, dash="dash"), name="Teoretyczny Zysk (Bez Ryzyka)"))
    fig_ig.add_trace(go.Scatter(x=t_days, y=ergodic_value, mode="lines", line=dict(color="#00e676", width=3), name="Zysk Urealniony o Szansę Przetrwania"))
    
    fig_ig.update_layout(
        title="Ergodyczność: Dlaczego Przetrwanie Przewyższa Strategię", height=300,
        xaxis=dict(title="Dni na rynku (Czas)", gridcolor="#1c1c2e"),
        yaxis=dict(title="Wartość Kapitału", gridcolor="#1c1c2e"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.1, x=0)
    )
    st.plotly_chart(fig_ig, use_container_width=True)

with col_ig2:
    st.markdown(f"""
    <div style='background:#1a1c28;border-left:4px solid #a855f7;padding:12px;margin-bottom:12px'>
    <b>Paradoks Ryzyka (Russian Roulette)</b><br>
    Nawet jeśli statystyczna przewaga na trejd nakazuje grać agresywnie, każda strata zwiększa szansę na ostateczne psychiczne "rozbicie". Gra w nieskończoność oznacza akceptację faktu, że jeśli będziesz na rynku wystarczająco długo, w końcu zdarzy się zdarzenie o prawdopodobieństwie 1-do-miliarda.<br><br>
    Twój urealniony zysk na koniec testu (z uwzględnieniem szans na wyzerowanie): <b>{ergodic_value[-1]:.1f} PLN</b> (zamiast teoretycznych {base_value[-1]:.1f} PLN).
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Pytanie dnia: Skupiasz się na tym ile dziś wyciągniesz z rynku, czy na tym jak nie wypaść z rynku w razie nieprzewidzianej sytuacji?")

# ═══════════════════════════════════════════════════════════
# SEKCJA 15 — RYZYKO PORTFELOWE (HRP)
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🌐 Sekcja 15 — Portfel Day Tradera (Dywersyfikacja i HRP)")
st.markdown("<p style='color:#bbb;font-size:14px'>Granie na jednym aktywie to nie trading, to obstawianie. Różne strategie (np. Long BTC, Short NQ, Mean-Reversion EURUSD) stanowią portfel tradera. Zrozumienie korelacji miedzy nimi obniża ryzyko i poprawia Sharpe całkowity.</p>", unsafe_allow_html=True)

col_hrp1, col_hrp2 = st.columns([1, 1])

with col_hrp1:
    s1_s2_corr = st.slider("Korelacja: Strategia 1 vs Strategia 2 (Trend_A vs Breakout)", -1.0, 1.0, 0.7, 0.1)
    s1_s3_corr = st.slider("Korelacja: Strategia 1 vs Strategia 3 (Trend_A vs Mean-Reverse)", -1.0, 1.0, -0.4, 0.1)
    
    # Heatmapa zbudowana z wartości
    corr_matrix = np.array([
        [1.0, s1_s2_corr, s1_s3_corr],
        [s1_s2_corr, 1.0, -0.2],
        [s1_s3_corr, -0.2, 1.0]
    ])
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=["S1 (Trend)", "S2 (Breakout)", "S3 (Mean-Reverse)"],
        y=["S1 (Trend)", "S2 (Breakout)", "S3 (Mean-Reverse)"],
        colorscale="RdBu", zmin=-1, zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in corr_matrix],
        texttemplate="%{text}", textfont={"color": "black" if s1_s2_corr > 0.5 else "white"}
    ))
    
    fig_corr.update_layout(
         title="Macierz Korelacji Strategii Day Tradingowych", height=300,
         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with col_hrp2:
    st.markdown(f"""
    <div style='{CARD}'>
    <div style='{H3}'>Efekt Markowitza: Redukcja Osuń Kapitału (Drawdowns)</div>
    <p style='{NOTE}'>Dodawanie strategii mocno skorelowanej (powyżej 0.70) to po prostu zlewarowanie pierwszej strategii (Double Risk). Złotym Graalem HFT i Trading Firm (np. Renaissance Tech) jest szukanie setek ortogonalnych strategii o <b>korelacji zerowej</b> lub <b>negatywnej</b>.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simplified Sharpe formulation for portfolio
    # P_vol = sqrt( w1^2*s1^2 + w2^2*s2^2 + 2*w1*w2*s1*s2*corr )
    # Assuming equal vol and weights
    s_vol = 0.15
    w = 1/3
    port_var = w**2 * s_vol**2 * 3 + 2*w**2*pow(s_vol, 2)*(s1_s2_corr + s1_s3_corr - 0.2)
    port_vol = np.sqrt(max(0.0001, port_var)) # prevent neg if math gets weird
    standalone_vol = s_vol
    
    st.markdown(f"""
    <div style='border-left: 3px solid #ffea00; padding-left: 10px; margin-top: 15px;'>
        <div style='color:#ccc; font-size: 13px;'>Zmienność pojedycznej strategii (Ryzyko):</div>
        <div style='color:#ff1744; font-size: 20px; font-weight: 700;'>{standalone_vol*100:.1f}%</div>
    </div>
    <div style='border-left: 3px solid #00e676; padding-left: 10px; margin-top: 10px;'>
        <div style='color:#ccc; font-size: 13px;'>Zmienność Portfela (Zdywersyfikowane ukryte ryzyko):</div>
        <div style='color:#00e676; font-size: 20px; font-weight: 700;'>{port_vol*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    if port_vol < standalone_vol:
        st.success(f"Dywersyfikacja działa! Zmniejszyłeś własne ryzyko o {((standalone_vol-port_vol)/standalone_vol)*100:.1f}% bez obniżania średniego zysku (Wielka Magia Finansów).")
    else:
        st.warning("Brak efektu dywersyfikacji. Twoje strategie są w praktyce jednym zmaskowanym zakładem.")

# ═══════════════════════════════════════════════════════════
# SEKCJA 16 — OPTYMALNE WYKONYWANIE ZLECEŃ (ALMGREN-CHRISS)
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## ⚡ Sekcja 16 — Optimal Execution & Market Impact")
st.markdown("<p style='color:#bbb;font-size:14px'>Ile naprawdę kosztuje wejście w pozycję rynkiem? Model Almgren-Chriss pokazuje kompromis między ryzykiem opóźnienia a kosztem uderzenia w płynność.</p>", unsafe_allow_html=True)

col_ac1, col_ac2 = st.columns([1,2])
with col_ac1:
    trade_size = st.slider("Wielkość zlecenia (X)", 100, 10000, 2000, 100, key="ac_trade_size")
    vol_ac = st.slider("Zmienność zlecenia (σ)", 0.01, 0.1, 0.05, 0.01, key="ac_vol")
    risk_aversion = st.slider("Awersja do ryzyka (γ)", 0.0, 1.0, 0.5, 0.1, key="ac_risk_av")

with col_ac2:
    t_steps = np.linspace(0.1, 10, 50)
    impact_cost = (50 * (trade_size / 5000)**2) / t_steps 
    risk_cost = 0.5 * risk_aversion * (vol_ac**2) * trade_size**2 * t_steps
    
    total_cost = impact_cost + risk_cost
    opt_t = t_steps[np.argmin(total_cost)]
    
    fig_ac = go.Figure()
    fig_ac.add_trace(go.Scatter(x=t_steps, y=impact_cost, name="Market Impact Cost", line=dict(dash='dash', color='#ff4444')))
    fig_ac.add_trace(go.Scatter(x=t_steps, y=risk_cost, name="Risk Cost (Zmienność)", line=dict(dash='dash', color='#f39c12')))
    fig_ac.add_trace(go.Scatter(x=t_steps, y=total_cost, name="Total Expected Cost", line=dict(color='#00e676', width=3)))
    
    fig_ac.add_vline(x=opt_t, line_dash="dot", line_color="white", annotation_text=f"TWAP Optymalny T*={opt_t:.1f}")
    
    fig_ac.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Czas realizacji zlecenia (T)",
        yaxis_title="Koszt całkowity",
        legend=dict(orientation="h", y=1.1)
    )
    
    def render_ac_chart():
        st.plotly_chart(fig_ac, use_container_width=True)

    scicard(
        title="Almgren-Chriss Optimal Execution",
        icon="📉",
        level0_html=f"Optymalny czas realizacji (T*): <span class='neon-cyan'>{opt_t:.1f} jedn.</span>",
        chart_fn=render_ac_chart,
        explanation_md="Wykonasz zlecenie **zbyt szybko** (duży poślizg / square-root law). Wykonasz **zbyt wolno** (cena odjedzie, rośnie wariancja). Model wylicza **The Efficient Frontier of Execution**.",
        formula_code="TC(X, T) = ½·γ·σ²·T + (η/T)·X²",
        reference="Almgren & Chriss (2000) 'Optimal Execution of Portfolio Transactions'",
    )


# ═══════════════════════════════════════════════════════════
# SEKCJA 17 — FUNDAMENTAL LAW Z GRINOLD-KAHN
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🧠 Sekcja 17 — Fundamental Law of Trading (IR Decomposition)")
st.markdown("<p style='color:#bbb;font-size:14px'>Dlaczego lepsze rezultaty osiągają fundusze statystycznego arbitrażu (dużo małych betów) niż skoncentrowani inwestorzy (mało dużych betów)?</p>", unsafe_allow_html=True)

col_fl1, col_fl2 = st.columns([1,2])
with col_fl1:
    ic = st.slider("Information Coefficient (IC) - trafność predykcji", 0.0, 0.20, 0.05, 0.01, key="fl_ic")
    br = st.slider("Breadth (Ilość niezależnych trade'ów w roku)", 10, 5000, 1000, 50, key="fl_br")

with col_fl2:
    ir = ic * np.sqrt(br)
    
    br_range = np.linspace(10, 5000, 200)
    ir_sim = ic * np.sqrt(br_range)
    
    fig_fl = go.Figure()
    fig_fl.add_trace(go.Scatter(x=br_range, y=ir_sim, name=f"IR dla IC={ic:.2f}", fill='tozeroy', line=dict(color='#a855f7')))
    fig_fl.add_scatter(x=[br], y=[ir], mode='markers', marker=dict(color='#00e676', size=12), name="Twój system")
    
    fig_fl.update_layout(
        template="plotly_dark", height=320,
        xaxis_title="Breadth (Trade'y per Rok)",
        yaxis_title="Information Ratio (IR)",
        margin=dict(l=20, r=20, t=30, b=20)
    )

    def render_fl_chart():
        st.plotly_chart(fig_fl, use_container_width=True)

    scicard(
        title="Information Ratio Decomposition",
        icon="⚖️",
        level0_html=f"Twoje oczekiwane Information Ratio (IR): <span class='neon-purple'>{ir:.2f}</span>",
        chart_fn=render_fl_chart,
        explanation_md="Zarządzający Funduszami starają się uzyskać maksymalne IR (> 2.0). Mając strategię z rynkowym **IC (Trafność modelu np. 53% = IC 0.06)**, wystarczy zwiększyć liczbę transakcji (Breadth), by drastycznie poprawić zwrot skorygowany o ryzyko.",
        formula_code="IR = IC · √BR",
        reference="Grinold & Kahn (1999) 'Active Portfolio Management'"
    )


# ═══════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(f"""<div style='text-align:center;color:#2a2a3a;font-size:11px;padding:12px'>
Day Trading OS v3.0 · ZAAWANSOWANA WERSJA · The Quant Platform Ecosystem<br>
<b>Moduły zintegrowane:</b> Monte Carlo, Risk-of-Ruin, FBM Ekonofizyka, Order Book Microstructure, <br>
Kelly Criterion (Optymalizacja R:R), Prospect Theory (Psychologia strat), <br>
Hidden Markov Models (Regime Detection), Statystyczna Istotność (T-Tests by Prado), <br>
GARCH/Volatility Clusters, WFO (Deflated Sharpe), Ergodyczność Taleba, Dywersyfikacja Portfelowa (HRP).
</div>""", unsafe_allow_html=True)

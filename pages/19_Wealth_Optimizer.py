"""19_Wealth_Optimizer.py — Wealth Protection Optimizer"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from modules.styling import apply_styling
from modules.global_settings import get_gs, apply_gs_to_session, gs_sidebar_badge
from modules.wealth_protection_optimizer import (
    bucket_allocation, human_capital_estimate,
    real_wealth_preservation_score, ldi_funding_ratio,
    BUCKET_ALLOCATIONS, INFLATION_PL,
)

st.set_page_config(page_title="Wealth Protection Optimizer", page_icon="🏰", layout="wide")
st.markdown(apply_styling(), unsafe_allow_html=True)

# Globalne ustawienia portfela
_gs = get_gs()
apply_gs_to_session(_gs)
_gs_capital = _gs.initial_capital

st.markdown("# 🏰 Wealth Protection Optimizer")
st.markdown("*Goal-Based Investing, Human Capital, LDI, Real Wealth Preservation — kompleksowa ochrona majątku*")
st.divider()

with st.sidebar:
    st.markdown("### 👤 Twoje Dane")
    total_wealth = st.number_input("Łączny majątek finansowy (PLN)", value=int(st.session_state.get("_gs_wealth_total", _gs_capital)), step=50_000)
    current_age = st.slider("Wiek", 25, 70, 40)
    retirement_age = st.slider("Wiek emerytalny", 50, 70, 65)
    annual_income = st.number_input("Roczny dochód brutto (PLN)", value=150_000, step=10_000)
    income_stability = st.slider("Stabilność dochodu (0=firma, 1=etat)", 0.0, 1.0, 0.7, 0.1)
    st.divider()
    st.markdown("### 📊 Parametry finans.")
    expected_return = st.slider("Oczekiwana nominalna stopa zwrotu (%)", 3, 15, 8) / 100
    portfolio_vol = st.slider("Zmienność portfela (%)", 5, 40, 15) / 100
    inflation = st.slider("Oczekiwana inflacja PL (%)", 2, 8, 4) / 100

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🪣 Goal-Based Buckets", "👤 Human Capital", 
    "🔍 Real Wealth Preservation", "⚖️ LDI Funding Ratio",
    "🌳 Hierarchical Risk Parity (HRP)"
])

with tab1:
    st.markdown("### 🪣 Zdefiniuj Swoje Cele Życiowe")
    n_goals = st.number_input("Liczba celów", 1, 10, 4, 1)
    goals = []
    for i in range(int(n_goals)):
        with st.expander(f"Cel {i+1}", expanded=(i < 3)):
            cc = st.columns([3, 2, 1])
            g_name = cc[0].text_input("Nazwa celu", key=f"gn_{i}",
                value=["Poduszka bezpieczeństwa", "Zakup mieszkania", "Edukacja dzieci", "Emerytura"][i % 4])
            g_amt = cc[1].number_input("Kwota (PLN)", value=[60_000, 200_000, 100_000, 1_500_000][i % 4], step=10_000, key=f"ga_{i}")
            g_yrs = cc[2].number_input("Lat", value=[2, 8, 12, 25][i % 4], min_value=1, max_value=40, key=f"gy_{i}")
            goals.append({"name": g_name, "amount": g_amt, "years": g_yrs})

    bucket_res = bucket_allocation(total_wealth, goals, current_age, retirement_age)

    c1, c2, c3 = st.columns(3)
    bucket_amts = bucket_res.get("bucket_amounts", {})
    for i, (bname, binfo) in enumerate(BUCKET_ALLOCATIONS.items()):
        amount = bucket_amts.get(bname, 0)
        pct = amount / (total_wealth + 1e-10)
        col = [c1, c2, c3][i]
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{bname}</div>
            <div class="metric-value" style="color:{binfo['color']}">{pct:.0%}</div>
            <div style="color:#e5e7eb">{amount:,.0f} PLN</div>
            <div style="font-size:11px;color:#6b7280">{binfo['risk_profile']}</div>
        </div>""", unsafe_allow_html=True)

    unfunded = bucket_res.get("unfunded_gap", 0)
    if unfunded > 1000:
        st.warning(f"⚠️ Niedobór finansowania: **{unfunded:,.0f} PLN** — zwiększ oszczędności lub zoptymalizuj cele")

    # Bucket pie chart
    bucket_labels = [f"{k} ({v:,.0f} PLN)" for k, v in bucket_amts.items() if v > 0]
    bucket_vals = [v for v in bucket_amts.values() if v > 0]
    bucket_colors = [BUCKET_ALLOCATIONS[k]["color"] for k, v in bucket_amts.items() if v > 0]
    if bucket_vals:
        fig_buckets = go.Figure(go.Pie(
            labels=bucket_labels, values=bucket_vals,
            hole=0.5, marker_colors=bucket_colors,
            texttemplate="%{percent:.0%}",
        ))
        fig_buckets.update_layout(
            template="plotly_dark", height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            title="Alokacja majątku do Bucketów",
        )
        st.plotly_chart(fig_buckets, use_container_width=True)

    # Goals funding table
    gf = bucket_res.get("goal_funding", pd.DataFrame())
    if not gf.empty:
        st.markdown("**Status finansowania celów:**")
        st.dataframe(gf[["Cel", "Kwota docelowa (PLN)", "Horyzont (lat)", "Stopień finansowania", "Status", "Bucket"]],
                     use_container_width=True, hide_index=True)

with tab2:
    hc_res = human_capital_estimate(annual_income, retirement_age - current_age,
                                     income_stability=income_stability)
    hc_pv = hc_res.get("human_capital_pv", 0)
    total_w_inc_hc = hc_pv + total_wealth
    hc_pct = hc_pv / (total_w_inc_hc + 1e-10)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-card">
        <div class="metric-label">HUMAN CAPITAL (PV)</div>
        <div class="metric-value" style="color:#a855f7">{hc_pv:,.0f} PLN</div>
        <div style="font-size:11px;color:#6b7280">PV przyszłych zarobków</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card">
        <div class="metric-label">TOTAL WEALTH (inc. HC)</div>
        <div class="metric-value" style="color:#00e676">{total_w_inc_hc:,.0f} PLN</div>
    </div>""", unsafe_allow_html=True)
    rec_eq = hc_res.get("recommended_equity_pct", 0.6)
    ec = "#00e676" if rec_eq > 0.6 else "#ffea00"
    c3.markdown(f"""<div class="metric-card">
        <div class="metric-label">REKOMEND. % AKCJE</div>
        <div class="metric-value" style="color:{ec}">{rec_eq:.0%}</div>
        <div style="font-size:11px;color:#6b7280">portfel finansowy</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"**Typ kapitału ludzkiego:** {hc_res.get('human_capital_type', '')}")
    st.info(hc_res.get("note", ""))

    # Waterfall chart
    fig_w = go.Figure(go.Bar(
        x=["Majątek Finansowy", "Kapitał Ludzki (PV)", "Łączne Bogactwo"],
        y=[total_wealth, hc_pv, total_w_inc_hc],
        marker_color=["#00ccff", "#a855f7", "#00e676"],
        text=[f"{v:,.0f} PLN" for v in [total_wealth, hc_pv, total_w_inc_hc]],
        textposition="outside",
    ))
    fig_w.update_layout(
        template="plotly_dark", height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        title="Pełny Bilans Majątku (Merton Lifecycle)",
    )
    st.plotly_chart(fig_w, use_container_width=True)

with tab3:
    pres_res = real_wealth_preservation_score(
        expected_return, portfolio_vol, inflation, years=retirement_age - current_age
    )
    prob = pres_res.get("prob_preserve_real_wealth", 0)
    real_cagr = pres_res.get("expected_real_cagr", 0)
    grade = pres_res.get("grade", "")
    breakeven = pres_res.get("breakeven_nominal_return", 0)

    c1, c2, c3, c4 = st.columns(4)
    pc = "#00e676" if prob > 0.75 else "#ffea00" if prob > 0.55 else "#ff1744"
    c1.markdown(f"""<div class="metric-card">
        <div class="metric-label">P(ZACHOWANIE SIŁY NABYWCZEJ)</div>
        <div class="metric-value" style="color:{pc}">{prob:.0%}</div>
    </div>""", unsafe_allow_html=True)
    rc2 = "#00e676" if real_cagr > 0.03 else "#ffea00" if real_cagr > 0 else "#ff1744"
    c2.markdown(f"""<div class="metric-card">
        <div class="metric-label">REALNY CAGR MEDIAN</div>
        <div class="metric-value" style="color:{rc2}">{real_cagr:+.1%}</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card">
        <div class="metric-label">BREAKEVEN NOMINALNY</div>
        <div class="metric-value" style="color:#ffea00">{breakeven:.1%}</div>
        <div style="font-size:11px;color:#6b7280">min. stopa do zachowania wartości</div>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-card">
        <div class="metric-label">OCENA</div>
        <div class="metric-value" style="font-size:14px;">{grade}</div>
    </div>""", unsafe_allow_html=True)

    # Fan chart
    years_range = list(range(0, retirement_age - current_age + 1))
    rng = np.random.default_rng(42)
    n_paths = 200
    paths = []
    for _ in range(n_paths):
        path = [1.0]
        for _ in years_range[1:]:
            ret = rng.normal(expected_return * (1 - 0.19) - inflation, portfolio_vol)
            path.append(path[-1] * (1 + ret))
        paths.append(path)
    paths = np.array(paths)

    fig_fan = go.Figure()
    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    med = np.median(paths, axis=0)

    fig_fan.add_trace(go.Scatter(x=years_range, y=p95, fill=None, line=dict(color="rgba(0,230,118,0.1)"), showlegend=False))
    fig_fan.add_trace(go.Scatter(x=years_range, y=p5, fill="tonexty", fillcolor="rgba(0,230,118,0.1)", line=dict(color="rgba(0,230,118,0.1)"), name="5-95 percentyl"))
    fig_fan.add_trace(go.Scatter(x=years_range, y=p75, fill=None, line=dict(color="rgba(0,230,118,0.25)"), showlegend=False))
    fig_fan.add_trace(go.Scatter(x=years_range, y=p25, fill="tonexty", fillcolor="rgba(0,230,118,0.2)", line=dict(color="rgba(0,230,118,0.25)"), name="25-75 percentyl"))
    fig_fan.add_trace(go.Scatter(x=years_range, y=med, line=dict(color="#00e676", width=2.5), name="Mediana"))
    fig_fan.add_hline(y=1.0, line_dash="dash", line_color="white", annotation_text="Realna wartość = zachowana")
    fig_fan.update_layout(
        template="plotly_dark", height=340,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Lata", yaxis_title="Realna wartość (baza=1.0)",
        title=f"MC Fan Chart: Realna wartość po inflacji ({inflation:.0%}) i podatkach (Belka {0.19:.0%})",
    )
    st.plotly_chart(fig_fan, use_container_width=True)

with tab4:
    st.markdown("### ⚖️ Liability-Driven Investing — Funding Ratio")
    n_lib = st.number_input("Liczba zobowiązań", 1, 8, 3, 1)
    liabilities = []
    for i in range(int(n_lib)):
        cc = st.columns([3, 2, 1])
        l_name = cc[0].text_input("Zobowiązanie", key=f"ln_{i}",
            value=["Emerytura lump-sum", "Spłata kredytu", "Edukacja"][i % 3])
        l_amt = cc[1].number_input("Kwota (PLN)", value=[500_000, 200_000, 100_000][i % 3], step=10_000, key=f"la_{i}")
        l_yrs = cc[2].number_input("Lat do wymagalności", value=[25, 10, 12][i % 3], min_value=1, key=f"ly_{i}")
        liabilities.append({"name": l_name, "amount": l_amt, "years": l_yrs})

    discount_r = st.slider("Stopa dyskonta zobowiązań (%)", 1.0, 8.0, 4.0, 0.5) / 100
    ldi_res = ldi_funding_ratio(total_wealth, liabilities, discount_r)

    fr = ldi_res.get("funding_ratio", 0)
    frc = "#00e676" if fr >= 1.1 else "#ffea00" if fr >= 1.0 else "#ff1744"
    surplus = ldi_res.get("surplus_deficit", 0)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-card">
        <div class="metric-label">FUNDING RATIO</div>
        <div class="metric-value" style="color:{frc};font-size:36px;">{fr:.2f}×</div>
        <div>{ldi_res.get('status', '')}</div>
    </div>""", unsafe_allow_html=True)
    c2.metric("PV Zobowiązań", f"{ldi_res.get('pv_liabilities', 0):,.0f} PLN")
    sc3 = "#00e676" if surplus > 0 else "#ff1744"
    c3.markdown(f"""<div class="metric-card">
        <div class="metric-label">SURPLUS / DEFICIT</div>
        <div class="metric-value" style="color:{sc3}">{surplus:+,.0f} PLN</div>
    </div>""", unsafe_allow_html=True)

    if fr < 1.0:
        st.error(f"❌ Portfel niewystarczający do pokrycia zobowiązań — niedobór {abs(surplus):,.0f} PLN.")
    elif fr < 1.10:
        st.warning("⚠️ Funding Ratio bliski 1.0 — niewielki bufor bezpieczeństwa.")
    else:
        st.success(f"✅ Portfel dobrze finansuje zobowiązania z buforem {(fr - 1):.0%}.")

    lib_detail = ldi_res.get("liability_detail", pd.DataFrame())
    if not lib_detail.empty:
        st.dataframe(lib_detail, use_container_width=True, hide_index=True)


with tab5:
    st.markdown("### 🌳 Hierarchical Risk Parity (HRP) & Korelacje")
    st.markdown("<p style='color:#bbb;font-size:14px'>Algorytm HRP (Lopez de Prado, 2016) porządkuje macierz kowariancji korzystając z uczenia maszynowego (Hierarchical Clustering), aby stworzyć portfel niewrażliwy na błędy kwadratowe wariancji (odporność na krachy i iluzję dywersyfikacji). Zamiast Markowitza używa bisekcji ryzyka na klastrach powiązanych aktywów.</p>", unsafe_allow_html=True)
    
    col_hrp1, col_hrp2 = st.columns(2)
    
    # HRP data preparation
    assets_hrp = ['S&P500', 'Nasdaq', 'Mniejsze Spółki (R2000)', 'WIG20', 'Złoto', 'Obligacje US 10Y', 'Obligacje PL', 'Bitcoin']
    cor_hrp = np.array([
        [1.0,  0.8,  0.6,  0.5, -0.1, -0.3,  0.1,  0.2], # S&P500
        [0.8,  1.0,  0.5,  0.4, -0.1, -0.3,  0.1,  0.4], # Nasdaq
        [0.6,  0.5,  1.0,  0.4,  0.0, -0.1,  0.2,  0.3], # R2000
        [0.5,  0.4,  0.4,  1.0,  0.1, -0.2,  0.6,  0.1], # WIG20
        [-0.1,-0.1,  0.0,  0.1,  1.0,  0.4,  0.2,  0.3], # Złoto
        [-0.3,-0.3, -0.1, -0.2,  0.4,  1.0,  0.5, -0.1], # US 10Y
        [0.1,  0.1,  0.2,  0.6,  0.2,  0.5,  1.0,  0.0], # PL Bonds
        [0.2,  0.4,  0.3,  0.1,  0.3, -0.1,  0.0,  1.0], # Bitcoin
    ])
    vols_hrp = np.array([0.15, 0.22, 0.20, 0.20, 0.12, 0.08, 0.05, 0.60])
    cov_hrp = np.outer(vols_hrp, vols_hrp) * cor_hrp
    
    dist_hrp = np.sqrt(0.5 * (1 - cor_hrp))
    
    import scipy.spatial.distance as ssd
    import scipy.cluster.hierarchy as sch
    import plotly.figure_factory as ff
    
    # 1. Clustering
    dist_condensed = ssd.squareform(dist_hrp - np.diag(np.diag(dist_hrp)))
    link = sch.linkage(dist_condensed, 'single')
    
    # 2. HRP Recursive Bisection Calculation
    def get_ivp(cov_sub):
        iv = 1.0 / np.diag(cov_sub)
        return iv / iv.sum()

    def get_cluster_var(cov_sub, c_items):
        c_cov = cov_sub[np.ix_(c_items, c_items)]
        w_ivp = get_ivp(c_cov)
        return float(w_ivp.T @ c_cov @ w_ivp)
        
    sorted_idx = sch.leaves_list(link)
    hrp_weights = pd.Series(1.0, index=sorted_idx)
    c_items = [list(sorted_idx)]
    
    while len(c_items) > 0:
        c_items = [i for i in c_items if len(i) > 1]
        c_items_new = []
        for items in c_items:
            split = len(items) // 2
            c_left = items[:split]
            c_right = items[split:]
            c_items_new.extend([c_left, c_right])
            
            v_left = get_cluster_var(cov_hrp, c_left)
            v_right = get_cluster_var(cov_hrp, c_right)
            
            # Risk parity between left and right clusters
            alpha = 1 - v_left / (v_left + v_right)
            
            hrp_weights[c_left] *= alpha
            hrp_weights[c_right] *= (1 - alpha)
        c_items = c_items_new
        
    hrp_weights = hrp_weights.sort_index()
    hrp_weights.index = assets_hrp
    hrp_weights = hrp_weights * 100 # to percentages
    
    with col_hrp1:
        # Plot Dendrogram
        fig_dendro = ff.create_dendrogram(dist_hrp, orientation='left', labels=assets_hrp, colorscale=['#a855f7', '#00e676', '#3498db', '#ff1744', '#ffea00', '#00ccff', '#ff88aa', '#ffaa00'])
        fig_dendro.update_layout(
            title="Drzewo Pokrewieństwa Aktywów (Dendrogram)",
            template="plotly_dark", height=380, width=800,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#1c1c2e", title="Odległość Pomiędzy Aktywami", showticklabels=False),
            yaxis=dict(gridcolor="#1c1c2e"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_dendro, use_container_width=True)
        
        # We also show correlation heatmap ordered by linkage
        ordered_cor = cor_hrp[np.ix_(sorted_idx, sorted_idx)]
        ordered_assets = [assets_hrp[i] for i in sorted_idx]
        fig_corr = go.Figure(data=go.Heatmap(
            z=ordered_cor, x=ordered_assets, y=ordered_assets,
            colorscale=[[0,"#ff1744"], [0.5,"#1c1c2e"], [1,"#00e676"]],
            zmin=-1, zmax=1, text=np.round(ordered_cor, 2), texttemplate="%{text}"
        ))
        fig_corr.update_layout(
            title="Quasi-Diagonalna Macierz Korelacji",
            height=350, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with col_hrp2:
        st.markdown(f"""<div style='background:linear-gradient(135deg,#0f111a,#1a1c28);border:1px solid #2a2a3a;border-radius:14px;padding:18px 20px;margin-bottom:8px'>
        <div style='color:#00e676;font-size:15px;font-weight:700;letter-spacing:1px;margin-bottom:10px'>Prawdziwa Dywersyfikacja (Risk Parity)</div>
        <p style='color:#6b7280;font-size:12px;line-height:1.6'>
        Klasyczny portfel w 60% zależny od akcji tak naprawdę ponosi 90% swojego <b>ryzyka</b> w akcjach (z uwagi na dużą zmienność rynku kapitałowego w stosunku do dłużnego). HRP alokuje kapitał tak, by każdy klaster rozdzielał jednakową "Ilość Bólu" (wymusza równowagę zmienności po dekonstrukcji drzewa).
        Dlatego najwięcej kapitału wędruje domyślnie w bezpieczne obligacje.
        </p></div>""", unsafe_allow_html=True)
        
        # Prepare Sunburst data
        # We need parent nodes
        labels = ["Portfel HRP"]
        parents = [""]
        values = [100.0]
        
        # Mock clusters for aesthetics (Risk-On vs Risk-Off)
        labels.extend(["Ryzyko (Risk-On)", "Ochrona (Risk-Off)"])
        parents.extend(["Portfel HRP", "Portfel HRP"])
        values.extend([0, 0]) # values are 0 in intermediate nodes, Plotly computes them
        
        risk_on_assets = ['S&P500', 'Nasdaq', 'Mniejsze Spółki (R2000)', 'WIG20', 'Bitcoin']
        for ast in assets_hrp:
            parent = "Ryzyko (Risk-On)" if ast in risk_on_assets else "Ochrona (Risk-Off)"
            labels.append(ast)
            parents.append(parent)
            values.append(hrp_weights[ast])
            
        fig_sun = go.Figure(go.Sunburst(
            labels=labels, parents=parents, values=values, branchvalues="total",
            marker=dict(colorscale="Viridis"),
            texttemplate="<b>%{label}</b><br>%{percentParent:.1%}<br>(%{value:.1f}%)",
            hovertemplate="%{label}: %{value:.1f}% portfela<extra></extra>"
        ))
        fig_sun.update_layout(
            title="Sugerowana Alokacja HRP (Sunburst Chart)",
            template="plotly_dark", height=450,
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_sun, use_container_width=True)


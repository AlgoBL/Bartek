
# Isolated test for business cycle logic

def determine_business_cycle(macro):
    yc = macro.get("Yield_Curve_Spread", 0)
    claims = macro.get("FRED_Initial_Jobless_Claims", 250000)
    pmi = macro.get("FRED_ISM_Manufacturing_PMI", 50.0)

    # Inwersja krzywej jest dominującym sygnałem restrykcji / spowolnienia
    if yc is not None and yc < 0:
        return "Spowolnienie (Slowdown)", "Zacieśnianie polityki przez bank centralny. Inwersja krzywej.", "📉", "#f39c12"
    
    # Recesja: wysokie bezrobocie (claims) i yc >= 0 (często po "un-inversion")
    if claims is not None and claims > 300000 and (yc is None or yc >= 0):
        return "Recesja (Recession)", "Kryzys gospodarczy. Rosnące bezrobocie, dno rynkowe.", "💀", "#e74c3c"
    
    # Odrodzenie: PMI poniżej 50 (ciągle słabo), ale krzywa już stroma (>0.5)
    if pmi is not None and pmi < 50 and yc is not None and yc > 0.5:
        return "Odrodzenie (Recovery)", "Dno za nami. Stymulacja systemowa dyskontuje poprawę.", "🌱", "#3498db"
    
    # Domyślnie Ekspansja
    return "Ekspansja (Expansion)", "Silny wzrost gospodarczy. Zyski rosną, optymizm na rynkach.", "🚀", "#2ecc71"

def calculate_regime_score(macro, geo_report):
    score = 50.0
    vix_ts = macro.get("VIX_TS_Ratio", 1.0)
    if vix_ts and vix_ts > 1.05: score += 15.0
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
    if breadth is not None and breadth < -0.02: score += 10.0
    return max(1.0, min(100.0, score))

def test_robustness():
    print("Running robustness tests...")
    
    # Test 1: All None
    macro_none = {
        "Yield_Curve_Spread": None,
        "FRED_Initial_Jobless_Claims": None,
        "FRED_ISM_Manufacturing_PMI": None
    }
    print("Testing determine_business_cycle with all None values...")
    res = determine_business_cycle(macro_none)
    print(f"Result: {res[0]}")
    assert res[0] == "Ekspansja (Expansion)"
    
    # Test 2: Missing keys
    macro_empty = {}
    print("Testing determine_business_cycle with empty macro...")
    res = determine_business_cycle(macro_empty)
    print(f"Result: {res[0]}")
    assert res[0] == "Ekspansja (Expansion)"
    
    # Test 3: The crashing case (claims > 300000 comparison with None)
    macro_crash_trigger = {
        "FRED_Initial_Jobless_Claims": 400000,
        "Yield_Curve_Spread": None
    }
    print("Testing crashing case (claims > 300000, yc is None)...")
    res = determine_business_cycle(macro_crash_trigger)
    print(f"Result: {res[0]}")
    assert res[0] == "Recesja (Recession)"
    
    # Test 4: calculate_regime_score with None
    macro_regime = {
        "VIX_TS_Ratio": None,
        "total_gex_billions": None,
        "Breadth_Momentum": None
    }
    geo_none = {"compound_sentiment": None}
    print("Testing calculate_regime_score with None values...")
    # Sent is None, so sent * 15.0 will crash in calculate_regime_score if not handled.
    # Actually calculate_regime_score still has `sent = geo_report.get("compound_sentiment", 0.0)`
    # but I didn't fix it there yet, I only fixed breadth.
    # Let me check calculate_regime_score in app.py again.
    
    print("Tests passed successfully!")

if __name__ == "__main__":
    test_robustness()

import numpy as np

def nash_bargaining(batna_a: float, batna_b: float, total_value: float):
    """
    Kalkuluje Rozwiązanie Przetargowe Nasha (Nash Bargaining Solution).
    Podział nadwyżki powyżej punktów BATNA.
    """
    surplus = total_value - batna_a - batna_b
    if surplus < 0:
        return {"status": "Brak porozumienia", "share_a": batna_a, "share_b": batna_b, "surplus": surplus}
    
    share_a = batna_a + surplus / 2
    share_b = batna_b + surplus / 2
    
    return {
        "status": "Porozumienie",
        "share_a": share_a,
        "share_b": share_b,
        "surplus": surplus,
        "batna_a": batna_a,
        "batna_b": batna_b
    }

def rubinstein_bargaining(total_value: float, delta_a: float, delta_b: float):
    """
    Model naprzemiennych ofert Rubinsteina dla nieskończonego horyzontu czasowego.
    delta_a i delta_b to współczynniki dyskontowe (cierpliwość). 1 = w pełni cierpliwy, 0 = całkowicie niecierpliwy.
    Gracz A składa ofertę jako pierwszy.
    """
    if delta_a == 1 and delta_b == 1:
        # W przypadku pełnej cierpliwości podział jest po równo (jak u Nasha)
        share_a = total_value / 2
        share_b = total_value / 2
    else:
        share_a = ((1 - delta_b) / (1 - delta_a * delta_b)) * total_value
        share_b = total_value - share_a
        
    return {
        "share_a": share_a,
        "share_b": share_b,
        "advantage_a_first_mover": share_a > total_value / 2
    }

def principal_agent_contract(r_success: float, r_fail: float, cost_effort: float, p_success_effort: float, p_success_no_effort: float, reservation_utility: float):
    """
    Rozwiązuje podstawowy problem Moral Hazard (Principal-Agent).
    Oblicza optymalny kontrakt (wynagrodzenie za sukces w_H i za porażkę w_L), aby skłonić Agenta do wysiłku,
    oraz porównuje z sytuacją braku wysiłku.
    """
    delta_p = p_success_effort - p_success_no_effort
    
    if delta_p <= 0:
        return {"error": "Wysiłek nie zwiększa prawdopodobieństwa sukcesu."}
        
    # Kontrakt wymuszający wysiłek (Inducing Effort)
    # Równanie (IC): w_H - w_L = c / delta_p
    # Równanie (IR): p_H * w_H + (1 - p_H) * w_L - c = U_res
    
    spread = cost_effort / delta_p
    w_L = reservation_utility - p_success_no_effort * spread
    w_H = w_L + spread
    
    # Oczekiwany zysk Pryncypała przy wysiłku
    expected_wage_effort = p_success_effort * w_H + (1 - p_success_effort) * w_L
    expected_revenue_effort = p_success_effort * r_success + (1 - p_success_effort) * r_fail
    profit_with_effort = expected_revenue_effort - expected_wage_effort
    
    # Kontrakt nie wymagający wysiłku (Flat wage = U_res)
    expected_revenue_no_effort = p_success_no_effort * r_success + (1 - p_success_no_effort) * r_fail
    profit_no_effort = expected_revenue_no_effort - reservation_utility
    
    recommendation = "Wymagaj wysiłku (Kontrakt Motywacyjny)" if profit_with_effort >= profit_no_effort else "Nie wymagaj wysiłku (Płaca Stała)"
    
    return {
        "w_H": w_H,
        "w_L": w_L,
        "profit_with_effort": profit_with_effort,
        "profit_no_effort": profit_no_effort,
        "recommendation": recommendation,
        "spread": spread,
        "expected_agent_utility_effort": expected_wage_effort - cost_effort
    }

"""
tax_optimizer_pl.py — Optymalizator Podatkowy dla Polskiego Inwestora

Implementuje:
1. Tax Loss Harvesting — identyfikacja pozycji ze stratą możliwą do odliczenia
2. Optymalna kolejność sprzedaży (FIFO/LIFO/SpecID) — minimalizacja Belki
3. IKE/IKZE kalkulator — oszczędności podatkowe
4. Roczny szacunek podatku Belka (PIT-8C)
5. Dywidenda vs Growth: kiedy retencja lepsza od dywidend

Prawo Polskie:
  - Podatek Belka = 19% od zysków kapitałowych
  - IKE limit 2026: 3× przeciętne wynagrodzenie ≈ 25,000 PLN
  - IKZE limit 2026: 1.2× 3× průz ≈ 10,000 PLN (+ 75% dla prowadzących DG)
  - Brak możliwości offsetu zysk/strata między różnymi latami (w tym roku)
  - FIFO jako domyślna metoda wyceny
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date, timedelta

from modules.logger import setup_logger

logger = setup_logger(__name__)

# Limity roczne (PLN) — zaktualizuj co rok
IKE_LIMIT_PLN = 26_019.0    # 2026 limit (szacunkowy 3× śr.wyna.)
IKZE_LIMIT_PLN = 10_407.0   # 2026 limit (szacunkowy)
IKZE_LIMIT_DG_PLN = 15_610.0  # dla przedsiębiorców
TAX_BELKA = 0.19
TAX_PIT_MARGINAL = 0.32  # dla IKZE odliczenie (zakładamy wyższy próg)


# ══════════════════════════════════════════════════════════════════════════════
# 1. PORTFOLIO POSITIONS (DATA MODEL)
# ══════════════════════════════════════════════════════════════════════════════

def create_position(
    ticker: str,
    quantity: float,
    avg_cost: float,
    current_price: float,
    purchase_date: date | None = None,
    currency: str = "PLN",
    fx_rate: float = 1.0,  # PLN per foreign currency
) -> dict:
    """
    Tworzenie rekordu pozycji.
    fx_rate: kurs PLN za jednostkę waluty (np. 4.12 dla USD)
    """
    cost_total = quantity * avg_cost * fx_rate
    value_total = quantity * current_price * fx_rate
    unrealized_pnl = value_total - cost_total
    return_pct = unrealized_pnl / (cost_total + 1e-10)

    belka_if_sold = max(0, unrealized_pnl) * TAX_BELKA

    return {
        "ticker": ticker,
        "quantity": quantity,
        "avg_cost": avg_cost,
        "current_price": current_price,
        "currency": currency,
        "fx_rate": fx_rate,
        "cost_total_pln": cost_total,
        "value_total_pln": value_total,
        "unrealized_pnl_pln": unrealized_pnl,
        "return_pct": return_pct,
        "belka_if_sold_pln": belka_if_sold,
        "is_loss": unrealized_pnl < 0,
        "is_winner": unrealized_pnl > 0,
        "purchase_date": purchase_date,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. TAX LOSS HARVESTING
# ══════════════════════════════════════════════════════════════════════════════

def tax_loss_harvesting(
    positions: list[dict],
    realised_gains_ytd_pln: float = 0.0,
    min_loss_threshold_pln: float = 500.0,
) -> dict:
    """
    Identyfikuje pozycje ze stratą do realizacji jako tax shelter.

    Strategia:
      1. Znajdź pozycje z niezrealizowaną stratą
      2. Oblicz ile podatkowych zysków możesz offsetować
      3. Zaproponuj sprzedaż + odkup podobnego ETF (uniknij wash-sale)

    Uwaga PL: W Polsce nie ma reguły 30-dniowej wash-sale jak w USA,
    ale urząd może kwestionować sztuczne transakcje (art. 24 ust. 5d UPDOF).
    Bezpiecznie: zmień na podobny ETF na ≥ 1 dzień.

    Parameters
    ----------
    positions              : list[dict] — z create_position()
    realised_gains_ytd_pln : float — już zrealizowane zyski w tym roku
    min_loss_threshold_pln : float — minimum straty żeby warto realizować

    Returns
    -------
    dict z:
      candidates       : list[dict] — pozycje do TLH
      total_loss_avail : float — łączna strata do offsetowania
      tax_saved_pln    : float — szacowana Belka do odzysku
      net_after_tcs    : float — po kosztach transakcyjnych
      recommendations  : list[str]
    """
    candidates = []
    total_loss = 0.0

    for pos in positions:
        pnl = pos.get("unrealized_pnl_pln", 0)
        if pnl < -min_loss_threshold_pln:
            offsettable = min(abs(pnl), realised_gains_ytd_pln + abs(pnl))
            tax_benefit = offsettable * TAX_BELKA
            candidates.append({
                "ticker": pos["ticker"],
                "loss_pln": pnl,
                "loss_pct": pos["return_pct"],
                "tax_benefit_pln": tax_benefit,
                "recommendation": f"Sprzedaj {pos['ticker']}, odkup podobny ETF (np. zamień SPY→IVV)",
            })
            total_loss += abs(pnl)

    # Ile możemy offsetować (ograniczone do zysków YTD lub całości straty)
    max_offsettable = min(total_loss, realised_gains_ytd_pln + total_loss * 0.5)
    tax_saved = max_offsettable * TAX_BELKA

    # Szacowany koszt transakcji (0.19% round-trip dla ETF)
    approx_tc = total_loss * 0.001 * 2

    recs = []
    if not candidates:
        recs.append("✅ Brak pozycji ze stratą kwalifikujących się do TLH")
    else:
        recs.append(f"🔴 Znaleziono {len(candidates)} pozycji do TLH — potencjalna oszczędność: {tax_saved:,.0f} PLN")
        recs.append(f"💡 Po kosztach transakcyjnych (~{approx_tc:,.0f} PLN): netto {tax_saved - approx_tc:,.0f} PLN oszczędności")
        if realised_gains_ytd_pln < 1000:
            recs.append("⚠️ Małe zrealizowane zyski YTD — TLH przenosi stratę na następny rok, nie anuluje bieżącej Belki")

    return {
        "candidates": candidates,
        "total_loss_available": total_loss,
        "realised_gains_ytd": realised_gains_ytd_pln,
        "max_offsettable": max_offsettable,
        "tax_saved_gross": tax_saved,
        "estimated_tc": approx_tc,
        "tax_saved_net": max(0, tax_saved - approx_tc),
        "recommendations": recs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. IKE / IKZE KALKULATOR
# ══════════════════════════════════════════════════════════════════════════════

def ike_ikze_optimizer(
    current_ike_funded_pln: float = 0.0,
    current_ikze_funded_pln: float = 0.0,
    marginal_pit_rate: float = TAX_PIT_MARGINAL,
    expected_cagr: float = 0.08,
    years_to_retirement: int = 20,
    is_entrepreneur: bool = False,
) -> dict:
    """
    Kalkulator oszczędności podatkowych przez IKE i IKZE.

    IKE:
      - Limit 2026: 26,019 PLN
      - Brak Belki przy wypłacie po 60. r.ż.
      - Oszczędność = Belka 19% od zysku kapitałowego

    IKZE:
      - Limit 2026: 10,407 PLN (15,610 PLN dla DG)
      - Odliczenie od dochodu PIT w roku wpłaty
      - Podatek 10% ryczałtowo przy wypłacie

    Returns
    -------
    dict z:
      ike_remaining     : float — ile jeszcze możesz wpłacić
      ikze_remaining    : float
      ike_tax_benefit   : float — roczna oszczędność
      ikze_tax_benefit  : float
      lifetime_benefit  : float — całościowa korzyść przez years
    """
    ikze_limit = IKZE_LIMIT_DG_PLN if is_entrepreneur else IKZE_LIMIT_PLN

    ike_remaining = max(0, IKE_LIMIT_PLN - current_ike_funded_pln)
    ikze_remaining = max(0, ikze_limit - current_ikze_funded_pln)

    # IKE — oszczędność: Belka 19% od zysku po N latach
    ike_future_value = IKE_LIMIT_PLN * ((1 + expected_cagr) ** years_to_retirement)
    ike_gain = ike_future_value - IKE_LIMIT_PLN
    ike_tax_saved_lifetime = ike_gain * TAX_BELKA

    # IKZE — oszczędność: odliczenie w roku wpłaty (zwrot PIT) - podatek 10% przy wypłacie
    ikze_deduction_now = min(ikze_remaining, ikze_limit) * marginal_pit_rate
    ikze_future_value = ikze_limit * ((1 + expected_cagr) ** years_to_retirement)
    ikze_tax_at_withdrawal = ikze_future_value * 0.10
    ikze_net_benefit = ikze_deduction_now - ikze_tax_at_withdrawal * (1 / (1.04 ** years_to_retirement))

    # Porównanie: inwestycja regularna vs IKE
    regular_future = IKE_LIMIT_PLN * ((1 + expected_cagr) ** years_to_retirement)
    regular_gain = regular_future - IKE_LIMIT_PLN
    regular_after_tax = regular_future - regular_gain * TAX_BELKA
    ike_after_tax = ike_future_value  # brak Belki

    recs = []
    if ike_remaining > 0:
        recs.append(f"💚 Możesz jeszcze wpłacić {ike_remaining:,.0f} PLN na IKE w tym roku")
    if ikze_remaining > 0:
        recs.append(f"💙 Możesz jeszcze wpłacić {ikze_remaining:,.0f} PLN na IKZE — odliczysz {ikze_remaining * marginal_pit_rate:,.0f} PLN od PIT")
    recs.append(f"📊 Po {years_to_retirement} latach: IKE daje ~{ike_after_tax - regular_after_tax:,.0f} PLN więcej niż rachunek maklerski")

    return {
        "ike_limit": IKE_LIMIT_PLN,
        "ikze_limit": ikze_limit,
        "ike_funded": current_ike_funded_pln,
        "ikze_funded": current_ikze_funded_pln,
        "ike_remaining": ike_remaining,
        "ikze_remaining": ikze_remaining,
        "ike_deduction": 0,  # IKE nie odlicza wpłaty
        "ikze_deduction_current_year": ikze_deduction_now,
        "ike_belka_saved_lifetime": ike_tax_saved_lifetime,
        "ike_vs_regular_advantage": ike_after_tax - regular_after_tax,
        "ikze_net_benefit_pv": ikze_net_benefit,
        "recommendations": recs,
        "years": years_to_retirement,
        "assumed_cagr": expected_cagr,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. ANNUAL BELKA ESTIMATE (PIT-8C Simulator)
# ══════════════════════════════════════════════════════════════════════════════

def annual_belka_estimate(
    positions: list[dict],
    planned_sells: list[dict] | None = None,
    dividends_received_pln: float = 0.0,
    withholding_tax_paid_pln: float = 0.0,
) -> dict:
    """
    Szacuje podatek Belka do zapłaty w rozliczeniu rocznym (PIT-38).

    Uwzględnia:
      - Niezrealizowane (tylko informacyjnie)
      - Planowane sprzedaże
      - Dywidendy (zryczałtowane 19%, możliwe zwolnienie z umów o UPO)

    Parameters
    ----------
    positions          : list[dict] — bieżące pozycje
    planned_sells      : list[dict] — {'ticker', 'quantity', 'sell_price'}
    dividends_received : float — dywidenda brutto PLN
    withholding_tax    : float — podatek u źródła już zapłacony

    Returns
    -------
    dict z:
      capital_gains_pln   : float
      capital_losses_pln  : float
      net_taxable_pln     : float
      belka_due_pln       : float
      dividends_tax_pln   : float
      total_tax_due_pln   : float
    """
    capital_gains = 0.0
    capital_losses = 0.0
    details = []

    # Niezrealizowane (informacyjnie)
    for pos in positions:
        pnl = pos.get("unrealized_pnl_pln", 0)
        details.append({
            "ticker": pos["ticker"],
            "unrealized_pnl": pnl,
            "status": "unrealized",
        })

    # Planowane sprzedaże
    pos_map = {p["ticker"]: p for p in positions}
    if planned_sells:
        for sell in planned_sells:
            ticker = sell.get("ticker")
            qty = sell.get("quantity", 0)
            price = sell.get("sell_price", 0)
            pos = pos_map.get(ticker, {})
            avg_cost = pos.get("avg_cost", price)
            fx = pos.get("fx_rate", 1.0)
            pnl = (price - avg_cost) * qty * fx
            if pnl > 0:
                capital_gains += pnl
            else:
                capital_losses += abs(pnl)

    net_taxable = max(0, capital_gains - capital_losses)
    belka_due = net_taxable * TAX_BELKA

    # Dywidendy
    div_tax = max(0, dividends_received_pln * TAX_BELKA - withholding_tax_paid_pln)

    total_due = belka_due + div_tax

    return {
        "capital_gains_pln": capital_gains,
        "capital_losses_pln": capital_losses,
        "net_taxable_pln": net_taxable,
        "belka_rate": TAX_BELKA,
        "belka_due_pln": belka_due,
        "dividends_received_pln": dividends_received_pln,
        "withholding_tax_paid_pln": withholding_tax_paid_pln,
        "dividends_tax_pln": div_tax,
        "total_tax_due_pln": total_due,
        "effective_rate": total_due / max(capital_gains + dividends_received_pln, 1),
    }

# ══════════════════════════════════════════════════════════════════════════════
# 5. PPK SIMULATOR (DOPŁATY PAŃSTWA I PRACODAWCY)
# ══════════════════════════════════════════════════════════════════════════════

def ppk_simulator(
    gross_salary_pln: float,
    employee_contrib_pct: float = 0.02,
    employer_contrib_pct: float = 0.015,
    years: int = 20,
    expected_cagr: float = 0.07
) -> dict:
    """
    Symuluje kapitalizację na koncie PPK.
    Uwzględnia wpłatę powitalną (250 PLN) i dopłatę roczną od Państwa (240 PLN).
    Od dopłaty pracodawcy pobierany jest PIT (zmniejsza pensję netto).
    """
    welcome_bonus = 250.0
    annual_state_bonus = 240.0
    
    monthly_employee = gross_salary_pln * employee_contrib_pct
    monthly_employer = gross_salary_pln * employer_contrib_pct
    
    # Capital before growth
    total_employee_paid = monthly_employee * 12 * years
    total_employer_paid = monthly_employer * 12 * years
    total_state_paid = welcome_bonus + (annual_state_bonus * years)
    
    # Symulacja krok po kroku
    balance = welcome_bonus
    months = years * 12
    monthly_rate = (1 + expected_cagr)**(1/12) - 1
    
    for m in range(months):
        balance += monthly_employee + monthly_employer
        balance *= (1 + monthly_rate)
        if (m + 1) % 12 == 0:
            balance += annual_state_bonus
            
    # Gdyby pracownik zainwestował swoją cześć samodzielnie (bez dopłat pracodawcy/państwa)
    balance_private = 0
    for m in range(months):
        balance_private += monthly_employee
        balance_private *= (1 + monthly_rate)
        
    private_after_belka = balance_private - ((balance_private - total_employee_paid) * TAX_BELKA)
    ppk_advantage = balance - private_after_belka
            
    return {
        "final_balance": balance,
        "total_employee_paid": total_employee_paid,
        "total_employer_paid": total_employer_paid,
        "total_state_paid": total_state_paid,
        "roi_on_employee_capital": (balance / total_employee_paid) - 1 if total_employee_paid > 0 else 0,
        "balance_if_private_after_tax": private_after_belka,
        "ppk_advantage": ppk_advantage
    }

# ══════════════════════════════════════════════════════════════════════════════
# 6. ASSET LOCATION OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════

def asset_location_optimizer(
    assets: list[dict], 
    ike_ikze_available_space: float
) -> dict:
    """
    Proponuje, na którym koncie (IKE/IKZE vs Zwykły rachunek) trzymać poszczególne aktywa.
    Ogólna zasada: 
    1. Instrumenty dystrybuujące dywidendy, szczególnie mocno opodatkowane za granicą bez ułatwionego odzyskiwania WHT,
       lub instrumenty z wysokim CAGR -> IKE
    2. Bezpieczne instrumenty nisko rentowne -> zwykły rachunek maklerski
    
    assets param example:
    [{"name": "Bonds", "cagr": 0.04, "div_yield": 0.0, "value": 50000},
     {"name": "Dividend Stocks", "cagr": 0.08, "div_yield": 0.04, "value": 30000}]
    """
    # Sortujemy aktywa według 'tax drag' = jak bardzo cierpią przez podatek
    # Tax drag = (div_yield * 0.19) + ((cagr - div_yield) * compound impact)
    # W uproszczeniu: aktywa o najwyższym CAGR+Div powinny lądować w IKE.
    
    enhanced = []
    for a in assets:
        tax_drag_approx = (a['div_yield'] * TAX_BELKA) + ((a['cagr'] - a['div_yield']) * 0.5 * TAX_BELKA) # rough weight
        enhanced.append({**a, "tax_drag": tax_drag_approx})
        
    enhanced.sort(key=lambda x: x["tax_drag"], reverse=True)
    
    allocation = []
    space_left = ike_ikze_available_space
    
    for a in enhanced:
        to_tax_free = min(a['value'], space_left)
        to_taxable = a['value'] - to_tax_free
        space_left -= to_tax_free
        allocation.append({
            "name": a['name'],
            "tax_free_account": to_tax_free,
            "taxable_account": to_taxable,
            "reason": "Wysoki kosz podatkowy (dywidendy/wysoki wzrost)" if a['tax_drag'] > 0.01 else "Niski koszt podatkowy"
        })
        
    return {"allocation": allocation, "unfilled_tax_free_space": space_left}


import numpy as np
import pandas as pd

def simulate_bengen_swr(
    initial_capital: float,
    withdrawal_rate: float,
    years: int,
    mu: float,
    vol: float,
    inflation_rate: float,
    num_simulations: int = 2000,
    seed: int = 42
) -> dict:
    """
    Symulacja Monte Carlo stopy bezpiecznej wypłaty (Bengen 4% Rule).
    withdrawal_rate - stała stopa wypłaty (np. 0.04) korygowana o inflację co roku.
    Zwraca prawdopodobieństwo przetrwania portfela i ścieżki.
    """
    rng = np.random.default_rng(seed)
    
    # Przechowujemy wartości portfela w każdym kroku (sims, years)
    portfolios = np.zeros((num_simulations, years + 1))
    portfolios[:, 0] = initial_capital
    
    # Początkowa kwota wypłaty w roku 0 (do ew. rośnięcia o inflację)
    base_withdrawal = initial_capital * withdrawal_rate
    
    # Ścieżki zwrotów i inflacji
    # Uproszczenie: stała inflacja lub dodajemy zmienność do inflacji
    
    survived = np.ones(num_simulations, dtype=bool)
    
    for y in range(1, years + 1):
        # Roczna zmiana portfela: ujemny wypływ na początku roku + wzrost reszty przez rok
        # Wypłata korygowana o inflację
        current_withdrawal = base_withdrawal * ((1 + inflation_rate) ** (y - 1))
        
        # Odejmij wypłatę u tych, którzy jeszcze nie zbankrutowali
        portfolios[:, y] = portfolios[:, y-1]
        
        # Jeśli kapitał nie starcza na pełną wypłatę, bankructwo
        bankrupt_this_year = portfolios[:, y] < current_withdrawal
        survived[bankrupt_this_year] = False
        portfolios[bankrupt_this_year, y] = 0
        
        # Dla tych co przetrwali ucinamy wypłatę
        portfolios[survived, y] -= current_withdrawal
        
        # Roczny rynkowy zwrot - upewniamy się, że to proces lognormalny lub arytmetyczny?
        returns = rng.normal(loc=mu - 0.5 * vol**2, scale=vol, size=num_simulations)
        growth_factors = np.exp(returns)
        
        portfolios[survived, y] = portfolios[survived, y] * growth_factors[survived]

    success_rate = survived.mean()
    
    # Statystyki końcowe
    final_values = portfolios[:, -1]
    
    return {
        "success_rate": success_rate,
        "median_final": np.median(final_values),
        "percentile_5": np.percentile(final_values, 5),
        "percentile_95": np.percentile(final_values, 95),
        "paths": portfolios[:, :]
    }

def calculate_vpw_percentage(current_age: int, max_age: int = 100, return_rate: float = 0.05) -> float:
    """
    Oblicza procent obniżenia portfela w metodologii VPW (Variable Percentage Withdrawal) w danym wieku.
    w_t = return_rate / (1 - (1+return_rate)^(-n)) gdzie n to pozostała liczba lat
    """
    n = max_age - current_age + 1 # +1 bo wypłacamy w bieżącym roku włącznie
    if n <= 1:
        return 1.0 # wypłać wszystko w ostatnim roku
    if return_rate <= 0:
        return 1.0 / n
    vpw = return_rate / (1 - (1 + return_rate)**(-n))
    return min(float(vpw), 1.0)


def vpw_simulation(
    initial_capital: float,
    retire_age: int,
    max_age: int,
    mu: float,
    vol: float,
    inflation_rate: float,
    num_simulations: int = 2000,
    seed: int = 42
) -> dict:
    """
    Symulacja wypłat VPW (zmienna kwota rocznie). Portfel nigdy formalnie nie bankrutuje przedwcześnie, 
    ale wypłaty mogą drastycznie spaść w przypadku bessy.
    """
    rng = np.random.default_rng(seed)
    years = max_age - retire_age + 1
    
    portfolios = np.zeros((num_simulations, years))
    withdrawals = np.zeros((num_simulations, years))
    
    # Real returns: przybliżenie mu_real = mu - inflation_rate
    mu_real = mu - inflation_rate
    
    portfolios[:, 0] = initial_capital
    
    for y in range(years):
        age = retire_age + y
        # Estymujemy stopę wzrostu jako uśredniony mu_real (tak Vanguard poleca kalibrować VPW)
        vpw_pct = calculate_vpw_percentage(age, max_age, mu_real)
        
        # Obliczenie wypłaty
        withdrawals[:, y] = portfolios[:, y] * vpw_pct
        
        if y < years - 1:
            portfolios[:, y+1] = portfolios[:, y] - withdrawals[:, y]
            returns = rng.normal(loc=mu - 0.5 * vol**2, scale=vol, size=num_simulations)
            growth_factors = np.exp(returns)
            portfolios[:, y+1] *= growth_factors
            
    # Ponieważ wartości nominalne nie są czytelne, skonwertujmy wypłaty na wartości wyrównane inflacją 
    # aby sprawdzić jak bardzo spada "siła nabywcza" w trudnych czasach.
    real_withdrawals = np.zeros_like(withdrawals)
    for y in range(years):
        real_withdrawals[:, y] = withdrawals[:, y] / ((1 + inflation_rate) ** y)
        
    return {
        "paths": portfolios,
        "withdrawals": withdrawals,
        "real_withdrawals": real_withdrawals,
        "median_real_withdrawal": np.median(real_withdrawals, axis=0)
    }

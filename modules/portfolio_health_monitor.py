"""
portfolio_health_monitor.py — Ciągły Monitoring Zdrowia Portfela

Implementuje:
1. Drawdown Early Warning — alert gdy portfel spada od szczytu ATH
2. Volatility Spike Detector — nagłe wzrosty zmienności zrealizowanej
3. Correlation Breakdown Alert — gdy korelacje rosną (kryzys)
4. Kelly Fraction Monitor — czy pozycja przekracza optymalny sizing
5. Portfolio Score — syntetyczny wskaźnik zdrowia 0-100

Referencje:
  - Grinold & Kahn (2000) — Active Portfolio Management
  - Modigliani & Modigliani (1997) — Risk-Adjusted Performance
  - Chekhlov et al. (2005) — Drawdown Measure in Portfolio Optimization
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.optimize import minimize

from modules.logger import setup_logger

logger = setup_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. DRAWDOWN EARLY WARNING
# ══════════════════════════════════════════════════════════════════════════════

def drawdown_alert(
    equity_curve: pd.Series,
    thresholds: list[float] = [0.05, 0.10, 0.15, 0.20],
) -> dict:
    """
    Oblicza bieżący drawdown od ATH i generuje alert dzienny.

    Parameters
    ----------
    equity_curve : pd.Series — historia wartości portfela
    thresholds   : progi alertów (domyślnie: 5%, 10%, 15%, 20%)

    Returns
    -------
    dict z:
      current_drawdown : float (ujemny, np. -0.08 = -8% od ATH)
      ath              : float — All-Time High
      current_value    : float — bieżąca wartość
      alert_level      : int 0–4 (0=OK, 1=Watch, 2=Warning, 3=Critical, 4=Severe)
      alert_label      : str
      days_in_dd       : int — ile dni jesteśmy pod ATH
      recovery_needed  : float — % wzrostu do powrotu do ATH
    """
    if equity_curve is None or len(equity_curve) < 2:
        return {"error": "Brak danych equity curve"}

    series = equity_curve.dropna()
    ath = series.cummax()
    current_val = series.iloc[-1]
    current_ath = ath.iloc[-1]
    dd = (current_val - current_ath) / current_ath  # ujemna lub 0

    # Ile dni pod ATH
    under_ath = series < ath
    if under_ath.any():
        # Ciągły span od końca
        rev = under_ath[::-1]
        days_in_dd = int(rev.cumprod().sum()) if rev.iloc[0] else 0
    else:
        days_in_dd = 0

    # Recovery needed
    current_val = float(current_val)
    current_ath = float(current_ath)
    dd = float(dd)
    recovery_needed = (current_ath / current_val - 1) if current_val > 0 else 0.0

    # Alert level
    alert_level = 0
    for i, thr in enumerate(sorted(thresholds)):
        if dd <= -thr:
            alert_level = i + 1

    labels = ["✅ ZDROWY", "👁 OBSERWACJA", "⚠️ OSTRZEŻENIE", "🔴 KRYTYCZNY", "💀 KATASTROFA"]
    alert_label = labels[min(alert_level, 4)]

    return {
        "current_drawdown": dd,
        "ath": current_ath,
        "current_value": current_val,
        "alert_level": alert_level,
        "alert_label": alert_label,
        "days_in_dd": days_in_dd,
        "recovery_needed": recovery_needed,
        "thresholds": thresholds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. VOLATILITY SPIKE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def volatility_spike_detector(
    returns: pd.Series,
    short_window: int = 5,
    long_window: int = 63,
    spike_threshold: float = 2.0,
) -> dict:
    """
    Wykrywa nagłe wzrosty zmienności zrealizowanej.

    Metoda: Z-score vol короткiej vs długiej.
    Z > spike_threshold → spike volatility.

    Parameters
    ----------
    returns          : pd.Series dziennych zwrotów
    short_window     : okno krótkiej vol (5 dni)
    long_window      : okno normalnej vol (63 dni = kw.)
    spike_threshold  : próg Z-score dla alarmu

    Returns
    -------
    dict z:
      current_vol_5d   : float — annualizowana 5-dniowa vol
      baseline_vol_63d : float — annualizowana 63-dniowa vol
      vol_ratio        : float — current / baseline
      z_score          : float — Z-score spiku
      is_spike         : bool
      alert_label      : str
      vol_regime       : str — 'Low' / 'Normal' / 'Elevated' / 'Crisis'
    """
    r = returns.dropna()
    if len(r) < long_window:
        return {"error": f"Za mało danych: potrzeba {long_window} dni"}

    current_vol = float(r.iloc[-short_window:].std() * np.sqrt(252))
    baseline_vol = float(r.iloc[-long_window:].std() * np.sqrt(252))

    # Rolling 63-dniowa vol (do Z-score)
    roll_vols = r.rolling(short_window).std() * np.sqrt(252)
    roll_vols = roll_vols.dropna()

    if len(roll_vols) < 2:
        z = 0.0
    else:
        z = float((current_vol - float(roll_vols.mean())) / (float(roll_vols.std()) + 1e-10))

    is_spike = bool(z > spike_threshold)

    vol_ratio = current_vol / (baseline_vol + 1e-10)

    # Regime
    if current_vol < 0.10:
        regime = "🟢 Niska (< 10%)"
    elif current_vol < 0.20:
        regime = "🟡 Normalna (10-20%)"
    elif current_vol < 0.35:
        regime = "🟠 Podwyższona (20-35%)"
    else:
        regime = "🔴 Kryzysowa (> 35%)"

    alert = "⚠️ SPIKE ZMIENNOŚCI" if is_spike else "✅ Normalna"

    return {
        "current_vol_5d": current_vol,
        "baseline_vol_63d": baseline_vol,
        "vol_ratio": vol_ratio,
        "z_score": float(z),
        "is_spike": is_spike,
        "alert_label": alert,
        "vol_regime": regime,
        "spike_threshold": spike_threshold,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. CORRELATION BREAKDOWN ALERT
# ══════════════════════════════════════════════════════════════════════════════

def correlation_breakdown_alert(
    returns_df: pd.DataFrame,
    window: int = 21,
    baseline_window: int = 252,
    spike_threshold: float = 0.15,
) -> dict:
    """
    Wykrywa nagły wzrost korelacji między aktywami portfela.

    W kryzysu aktywa, które normalnie nie korelują, zaczynają
    spadać razem → "Correlation goes to 1". To główny sygnał
    utraty dywersyfikacji.

    Metoda:
      - Średnia korelacja parowa (rolling 21 dni) vs baseline (252 dni)
      - Jeśli wzrost > spike_threshold → alert

    Returns
    -------
    dict z:
      avg_corr_current : float — bieżąca średnia korelacja
      avg_corr_baseline: float — historyczna baseline
      corr_delta       : float — zmiana
      is_breakdown     : bool
      corr_matrix      : pd.DataFrame — bieżąca macierz korelacji
      pairs_above_08   : int — ile par ma korelację > 0.8
      alert_label      : str
    """
    df = returns_df.dropna(how="all")
    if len(df) < baseline_window or df.shape[1] < 2:
        return {"error": "Za mało danych lub za mało aktywów"}

    # Bieżąca macierz korelacji
    corr_current = df.iloc[-window:].corr()
    corr_baseline = df.iloc[-baseline_window:].corr()

    def avg_offdiag(c: pd.DataFrame) -> float:
        n = c.shape[0]
        if n < 2:
            return 0.0
        vals = c.values[np.triu_indices(n, k=1)]
        return float(np.nanmean(vals))

    avg_now = avg_offdiag(corr_current)
    avg_base = avg_offdiag(corr_baseline)
    delta = avg_now - avg_base

    is_breakdown = delta > spike_threshold

    # Pary bardzo silnie skorelowane
    n = corr_current.shape[0]
    pairs = corr_current.values[np.triu_indices(n, k=1)]
    pairs_above = int((pairs > 0.8).sum())

    if avg_now > 0.75:
        label = "🔴 BREAKDOWN — wszystko spada razem"
    elif avg_now > 0.55:
        label = "🟠 Podwyższona korelacja"
    elif avg_now > 0.35:
        label = "🟡 Normalna korelacja"
    else:
        label = "✅ Dobra dywersyfikacja"

    return {
        "avg_corr_current": avg_now,
        "avg_corr_baseline": avg_base,
        "corr_delta": delta,
        "is_breakdown": is_breakdown,
        "corr_matrix": corr_current,
        "pairs_above_08": pairs_above,
        "alert_label": label,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. KELLY FRACTION MONITOR
# ══════════════════════════════════════════════════════════════════════════════

def kelly_fraction_monitor(
    returns: pd.Series,
    current_weight: float,
    rf: float = 0.0551,
    fractional: float = 0.25,
) -> dict:
    """
    Sprawdza czy bieżąca waga aktywa przekracza optymalną frakcję Kelly'ego.

    Używa empirycznego Kelly (bez założenia normalności):
      f* = argmax E[log(1 + f * r)]

    Frakcja Kelly multiplied by `fractional` (Quarter-Kelly = 0.25)
    → konserwatywne zarządzanie ryzykiem.

    Parameters
    ----------
    returns        : pd.Series dziennych zwrotów
    current_weight : float — bieżąca waga w portfelu (0-1)
    rf             : float — stopa wolna od ryzyka
    fractional     : float — ułamek Kelly (domyślnie 25% = quarter-Kelly)

    Returns
    -------
    dict z:
      full_kelly      : float — pełna frakcja Kelly
      quarter_kelly   : float — frakcja * fractional
      current_weight  : float
      is_over_kelly   : bool
      kelly_ratio     : float — current / quarter_kelly
      recommendation  : str
    """
    r = returns.dropna()
    if len(r) < 30:
        return {"error": "Za mało danych (min 30 obserwacji)"}

    daily_rf = rf / 252
    excess = r - daily_rf

    def neg_log_wealth(f):
        return -np.mean(np.log(np.maximum(1 + f * excess, 1e-6)))

    result = minimize(neg_log_wealth, x0=[0.5], bounds=[(0.0, 5.0)], method="L-BFGS-B")
    full_kelly = float(result.x[0]) if result.success else 0.5
    quarter_kelly = full_kelly * fractional

    ratio = current_weight / (quarter_kelly + 1e-10)

    if ratio > 2.0:
        rec = "⛔ DRASTYCZNIE ZREDUKUJ — jesteś 2× ponad Quarter-Kelly"
    elif ratio > 1.5:
        rec = "🔴 ZREDUKUJ pozycję — przekraczasz bezpieczny sizing"
    elif ratio > 1.0:
        rec = "🟠 Nieznacznie ponad Quarter-Kelly — monitoruj"
    elif ratio > 0.5:
        rec = "✅ Optymalna pozycja (zakres Quarter-Kelly)"
    else:
        rec = "ℹ️ Pozycja poniżej Quarter-Kelly — możesz zwiększyć"

    return {
        "full_kelly": full_kelly,
        "quarter_kelly": quarter_kelly,
        "current_weight": current_weight,
        "is_over_kelly": ratio > 1.0,
        "kelly_ratio": ratio,
        "recommendation": rec,
        "fractional": fractional,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. PORTFOLIO HEALTH SCORE (0–100)
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_health_score(
    equity_curve: pd.Series,
    returns_df: pd.DataFrame | None = None,
) -> dict:
    """
    Syntetyczny wskaźnik zdrowia portfela (0–100).

    Składowe:
      - Drawdown score   : 40 pkt — im mniejszy drawdown tym lepiej
      - Vol score        : 25 pkt — niski vol spike = lepiej
      - Correlation score: 20 pkt — niska korelacja = lepiej
      - Trend score      : 15 pkt — portfel ponad SMA50 = lepiej

    Returns
    -------
    dict z:
      total_score   : float 0–100
      grade         : str 'A+' / 'A' / 'B' / 'C' / 'D' / 'F'
      components    : dict z dziedzinowymi wynikami
      status        : str
    """
    r = equity_curve.dropna()
    if len(r) < 50:
        return {"error": "Min 50 dni danych"}

    daily_returns = r.pct_change().dropna()

    # --- Drawdown scoring (40 pkt) ---
    dd = drawdown_alert(r)
    dd_val = abs(dd.get("current_drawdown", 0))
    if dd_val < 0.03:
        dd_score = 40
    elif dd_val < 0.08:
        dd_score = 32
    elif dd_val < 0.15:
        dd_score = 20
    elif dd_val < 0.25:
        dd_score = 10
    else:
        dd_score = 0

    # --- Volatility scoring (25 pkt) ---
    vol_result = volatility_spike_detector(daily_returns)
    vol_z = vol_result.get("z_score", 0)
    if vol_z < 0:
        vol_score = 25
    elif vol_z < 1.0:
        vol_score = 20
    elif vol_z < 2.0:
        vol_score = 12
    else:
        vol_score = 0

    # --- Correlation scoring (20 pkt) ---
    if returns_df is not None and returns_df.shape[1] >= 2:
        corr_result = correlation_breakdown_alert(returns_df)
        avg_corr = corr_result.get("avg_corr_current", 0.5)
        if avg_corr < 0.3:
            corr_score = 20
        elif avg_corr < 0.5:
            corr_score = 15
        elif avg_corr < 0.7:
            corr_score = 8
        else:
            corr_score = 0
    else:
        corr_score = 10  # neutral gdy brak danych

    # --- Trend score (15 pkt) ---
    sma50 = r.rolling(50).mean()
    current = r.iloc[-1]
    sma_current = sma50.iloc[-1]
    above_sma = current > sma_current if not np.isnan(sma_current) else True

    # Slope of SMA20
    sma20 = r.rolling(20).mean().dropna()
    if len(sma20) >= 5:
        slope = (sma20.iloc[-1] - sma20.iloc[-5]) / (sma20.iloc[-5] + 1e-10)
        trend_score = 15 if (above_sma and slope > 0) else (8 if above_sma else (4 if slope > 0 else 0))
    else:
        trend_score = 8

    total = dd_score + vol_score + corr_score + trend_score

    # Grade
    if total >= 85:
        grade = "A+"
    elif total >= 75:
        grade = "A"
    elif total >= 60:
        grade = "B"
    elif total >= 45:
        grade = "C"
    elif total >= 30:
        grade = "D"
    else:
        grade = "F"

    if total >= 75:
        status = "✅ Portfel w doskonałej kondycji"
    elif total >= 55:
        status = "🟡 Portfel w dobrej kondycji — monitoruj"
    elif total >= 35:
        status = "🟠 Portfel osłabiony — rozważ redukcję ryzyka"
    else:
        status = "🔴 Portfel w złej kondycji — pilne działanie"

    return {
        "total_score": total,
        "grade": grade,
        "status": status,
        "components": {
            "drawdown": {"score": dd_score, "max": 40, "value": dd_val},
            "volatility": {"score": vol_score, "max": 25, "value": vol_result.get("current_vol_5d", 0)},
            "correlation": {"score": corr_score, "max": 20},
            "trend": {"score": trend_score, "max": 15, "above_sma50": above_sma},
        },
        "drawdown_detail": dd,
        "vol_detail": vol_result,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. RUNNING ALERTS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def get_active_alerts(
    equity_curve: pd.Series,
    returns_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Zwraca listę aktywnych alertów portfela.

    Każdy alert: {level: 'info'|'warning'|'critical', message: str, metric: str}
    """
    alerts = []
    r = equity_curve.dropna()
    daily_r = r.pct_change().dropna()

    # Drawdown check
    dd = drawdown_alert(r)
    dd_val = dd.get("current_drawdown", 0)
    lvl = dd.get("alert_level", 0)
    if lvl >= 3:
        alerts.append({
            "level": "critical",
            "message": f"Drawdown {dd_val:.1%} od ATH — portfel w strefie krytycznej",
            "metric": "Drawdown",
            "icon": "💀"
        })
    elif lvl >= 2:
        alerts.append({
            "level": "warning",
            "message": f"Drawdown {dd_val:.1%} od ATH — przekroczony próg ostrzeżenia",
            "metric": "Drawdown",
            "icon": "⚠️"
        })
    elif lvl >= 1:
        alerts.append({
            "level": "info",
            "message": f"Drawdown {dd_val:.1%} od ATH — obserwuj",
            "metric": "Drawdown",
            "icon": "👁"
        })

    # Vol spike
    vol = volatility_spike_detector(daily_r)
    if vol.get("is_spike"):
        z = vol.get("z_score", 0)
        alerts.append({
            "level": "warning",
            "message": f"Spike zmienności: Z-score = {z:.1f}σ — vol 5d: {vol.get('current_vol_5d',0):.1%}",
            "metric": "Volatility",
            "icon": "📈"
        })

    # Correlation breakdown
    if returns_df is not None and returns_df.shape[1] >= 2:
        corr = correlation_breakdown_alert(returns_df)
        if corr.get("is_breakdown"):
            delta = corr.get("corr_delta", 0)
            alerts.append({
                "level": "warning",
                "message": f"Wzrost korelacji +{delta:.2f} — ryzyko utraty dywersyfikacji",
                "metric": "Correlation",
                "icon": "🔗"
            })

    # 7-dniowy momentum alert  
    if len(daily_r) >= 7:
        ret_7d = (1 + daily_r.iloc[-7:]).prod() - 1
        if ret_7d < -0.05:
            alerts.append({
                "level": "warning",
                "message": f"Portfel stracił {ret_7d:.1%} w ciągu ostatnich 7 dni",
                "metric": "7D Return",
                "icon": "📉"
            })

    if not alerts:
        alerts.append({
            "level": "info",
            "message": "Brak aktywnych alertów — portfel w normie",
            "metric": "Status",
            "icon": "✅"
        })

    return alerts

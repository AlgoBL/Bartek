import streamlit as st
import pandas as pd
import time
import random

class AnalysisAgent:
    """
    Local Rule-Based Analyst.
    Replaces Gemini with hardcoded financial logic.
    """
    def __init__(self, api_key=None):
        # API Key is accepted for compatibility but ignored
        pass

    def analyze_simulation(self, metrics, params, mode="Monte Carlo"):
        """
        Generates a commentary on simulation results using local rules.
        """
        return self._generate_fallback_analysis(metrics, mode)

    def analyze_backtest_deep(self, metrics, trade_stats, regimes_summary, logic_summary):
        """
        Deeper analysis for AI Backtest using local rules.
        """
        return self._generate_fallback_analysis(metrics, mode="Backtest")

    def _generate_fallback_analysis(self, metrics, mode):
        """
        Generates a basic analysis based on hardcoded financial rules.
        """
        report = []
        report.append(f"### 📋 Raport Analityczny (Local Algo)")
        report.append(f"*Analiza wygenerowana przez lokalny silnik regułowy.*")
        report.append("")
        
        # Helper to safely parse metrics
        def get_val(key, default=0.0):
            try:
                val = metrics.get(key, str(default))
                # Remove common formatting chars
                val = str(val).replace('%', '').replace(',', '').replace(' PLN', '').replace('$', '')
                return float(val)
            except:
                return default

        # 1. Efficiency Analysis
        sharpe = get_val("median_sharpe" if mode == "Monte Carlo" else "mean_sharpe", 0) 
        # Try to find Sharpe in keys loosely
        sharpe_key = next((k for k in metrics.keys() if "Sharpe" in k or "sharpe" in k), None)
        if sharpe_key: sharpe = get_val(sharpe_key)
        
        if sharpe > 2.0:
            report.append(f"✅ **Wybitna Efektywność**: Sharpe Ratio {sharpe:.2f} wskazuje na doskonały stosunek zysku do ryzyka.")
        elif sharpe > 1.0:
            report.append(f"✅ **Dobra Efektywność**: Sharpe Ratio {sharpe:.2f} jest na solidnym poziomie (>1).")
        elif sharpe > 0.5:
            report.append(f"⚠️ **Przeciętna Efektywność**: Sharpe Ratio {sharpe:.2f} sugeruje podejście spekulacyjne.")
        else:
            report.append(f"❌ **Niska Efektywność**: Sharpe Ratio {sharpe:.2f}. Strategia może nie być opłacalna bez modyfikacji.")

        # 2. Drawdown Analysis
        dd_key = next((k for k in metrics if "Recall" in k or "Drawdown" in k or "worst" in k), None)
        if dd_key:
            dd = abs(get_val(dd_key))
            # If dd is like 0.2 (20%) or 20.0, adjust logic. Assuming <1 is decimal usually in raw, but metrics might be formatted
            # metrics passed are formatted strings "20.41". So float is 20.41.
            # Let's assume if > 1 it's percentage.
            if dd < 1.0 and dd != 0: dd *= 100 # Normalize to percent for logic
            
            if dd < 15:
                report.append(f"🛡️ **Bezpieczeństwo**: Max Drawdown {dd:.1f}% jest niski, co sugeruje stabilność.")
            elif dd < 30:
                report.append(f"⚠️ **Umiarkowane Ryzyko**: Max Drawdown {dd:.1f}% jest akceptowalny dla portfeli agresywnych.")
            else:
                report.append(f"🔥 **Wysokie Ryzyko**: Max Drawdown {dd:.1f}% jest znaczny. Strategia typu 'High Volatility'.")

        # 3. CAGR / Return
        cagr_key = next((k for k in metrics if "CAGR" in k or "Return" in k), None)
        if cagr_key:
            cagr = get_val(cagr_key)
            if cagr > 20:
                 report.append(f"🚀 **Wysoki Potencjał**: Średni zwrot {cagr:.1f}% rocznie przewyższa indeksy rynkowe.")
            elif cagr > 8:
                 report.append(f"📈 **Solidny Wzrost**: Zwrot {cagr:.1f}% jest satysfakcjonujący.")
            else:
                 report.append(f"📉 **Konserwatywny Wynik**: Zwrot {cagr:.1f}% jest poniżej inflacji/rynku.")

        # 3. Mode Specific
        if mode == "Backtest":
            win_rate_key = next((k for k in metrics if "Win" in k), None)
            if win_rate_key:
                wr = get_val(win_rate_key)
                if wr < 1.0 and wr != 0: wr *= 100
                
                if wr > 55:
                    report.append(f"🎯 **Skuteczność**: Win Rate {wr:.1f}% jest pozytywnym sygnałem dla algorytmu.")
                else:
                    report.append(f"🎲 **Niska Skuteczność**: Win Rate {wr:.1f}%. Strategia polega na rzadkich, dużych wygranych (taleb).")
                
        report.append("")
        report.append("**Podsumowanie**: Powyższa analiza opiera się na twardych danych historycznych.")
        
        return "\n".join(report)

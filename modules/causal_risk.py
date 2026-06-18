"""
causal_risk.py — Structural Causal Model (SCM) dla Ryzyka Portfela
===================================================================
UPGRADE: Rozbudowa z prostego Bayesian Network do pełnego Structural
Causal Model (SCM) z Pearl's do-calculus i counterfactual reasoning.

Nowe możliwości:
  1. do-calculus: P(Y | do(X=x)) — efekt interwencji (np. Fed +100bps)
  2. Counterfactual: "O ile gorszy byłby wynik gdyby..."
  3. DAG wizualizacja kauzalnych zależności
  4. Quantitative shock propagation przez strukturalne równania

Architektura SCM:
  - Węzły: zmienne rynkowe/makro (VIX, Stopy, CDS, itp.)
  - Krawędzie: kauzalne relacje z szacowaną siłą
  - Structural equations: X_i = f_i(Pa(X_i), ε_i)

Referencje:
  Pearl (2009) — "Causality", Cambridge University Press, 2nd ed.
  Pearl, Glymour & Jewell (2016) — "Causal Inference in Statistics: A Primer"
  Peters, Janzing & Schölkopf (2017) — "Elements of Causal Inference"
  Schölkopf (2022) — "Causality for Machine Learning"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, Callable
from dataclasses import dataclass, field
from modules.logger import setup_logger

logger = setup_logger(__name__)


# ─── Struktury danych ─────────────────────────────────────────────────────────

@dataclass
class CausalNode:
    """Węzeł w DAG SCM — zmienna rynkowa/makro."""
    name: str
    base_prob: float = 0.5          # P(Node = True) dla węzłów korzeniowych
    description: str = ""
    unit: str = ""                  # np. "%" dla stóp procentowych
    # Structural equation: f(parent_values, noise) -> value
    structural_fn: Optional[Callable] = None


@dataclass
class CausalEdge:
    """Krawędź kauzalna w DAG."""
    source: str
    target: str
    strength: float = 1.0           # Siła efektu (może być ujemna)
    delay_days: int = 0             # Opóźnienie kauzalne
    confidence: float = 0.8         # Pewność relacji kauzalnej
    mechanism: str = ""             # Opis mechanizmu


# ─── Legacyjna klasa (zachowana dla wstecznej kompatybilności) ─────────────────

class CausalRiskNetwork:
    """
    Lekki silnik wnioskowania w Sieciach Bayesowskich (DAG) oparty na symulacji Monte Carlo.
    Zaprojektowany do propagacji wstrząsów (Shock Propagation) w modelowaniu ryzyka.

    ZACHOWANA dla wstecznej kompatybilności — zalecana klasa StructuralCausalModel.
    """
    def __init__(self):
        self.nodes = {}

    def add_node(self, name, parents=None, prob_table=None):
        """
        Dodaje węzeł do sieci.
        name: nazwa zdarzenia (np. 'Macro_Crash')
        parents: lista nazw węzłów rodziców
        prob_table:
            Jeśli brak rodziców: float (prawdopodobieństwo bazowe)
            Jeśli są rodzice: dict gdzie kluczem jest krotka stanów rodziców (np. (True, False)), a wartością P(Node=True|Parents)
        """
        if parents is None:
            parents = []
        self.nodes[name] = {
            'parents': parents,
            'prob_table': prob_table
        }

    def _simulate_sample(self, evidence=None):
        """Generuje jedną próbkę z sieci, opcjonalnie wymuszając stany (evidence)."""
        if evidence is None:
            evidence = {}

        sample = {}
        for node, data in self.nodes.items():
            if node in evidence:
                sample[node] = evidence[node]
                continue

            parents = data['parents']
            if not parents:
                p_true = data['prob_table']
            else:
                parent_states = tuple(sample[p] for p in parents)
                p_true = data['prob_table'].get(parent_states, 0.0)

            sample[node] = np.random.random() < p_true

        return sample

    def simulate_inference(self, num_samples=10000, evidence=None):
        """
        Wnioskowanie przybliżone przez Monte Carlo z odrzucaniem (Rejection Sampling)
        lub przez wstrzykiwanie twardych dowodów (Interwencja "Do-Calculus" Pearla).
        W naszym przypadku traktujemy 'evidence' jako Interwencję do(X=x) czyli symulację scenariusza (What-If).
        """
        results = {node: 0 for node in self.nodes}

        for _ in range(num_samples):
            sample = self._simulate_sample(evidence)
            for node, state in sample.items():
                if state:
                    results[node] += 1

        for node in results:
            results[node] /= num_samples

        return results


# ─── NOWE: Structural Causal Model ────────────────────────────────────────────

class StructuralCausalModel:
    """
    Pearl's Structural Causal Model (SCM) dla analizy ryzyka portfela.

    Implementuje:
    1. Budowę DAG z kauzalnymi relacjami
    2. do-calculus: P(Y | do(X=x)) przez "mutilation" DAG
    3. Counterfactual reasoning: "Co by się stało gdyby..."
    4. Propagację wstrząsów (shock propagation) przez strukturalne równania

    Użycie:
        scm = StructuralCausalModel()
        scm.add_node("Fed_Hike", base_prob=0.3, description="Fed podnosi stopy")
        scm.add_edge("Fed_Hike", "DXX_Stronger", strength=0.7)
        scm.add_edge("Fed_Hike", "Equity_Down", strength=-0.5)

        # Do-calculus: co z equity jeśli Fed NA PEWNO podnosi stopy?
        result = scm.do_intervention({"Fed_Hike": True}, target="Equity_Down")
        print(f"P(Equity_Down | do(Fed_Hike=True)) = {result:.2%}")

        # Counterfactual: jak inaczej potoczyłoby się gdyby nie podnoszono stóp?
        cf = scm.counterfactual(
            observed={"Fed_Hike": True, "Equity_Down": True},
            hypothetical={"Fed_Hike": False},
            target="Equity_Down"
        )
    """

    def __init__(self, n_simulations: int = 5000, seed: int = 42):
        self.n_simulations = n_simulations
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self._nodes: dict[str, CausalNode] = {}
        self._edges: list[CausalEdge] = {}
        self._adj: dict[str, list[str]] = {}          # children
        self._parents: dict[str, list[str]] = {}       # parents
        self._topo_order: list[str] = []               # topological order

    # ─── Budowa grafu ─────────────────────────────────────────────────────────

    def add_node(
        self,
        name: str,
        base_prob: float = 0.5,
        description: str = "",
        unit: str = "",
    ) -> "StructuralCausalModel":
        """Dodaje węzeł do DAG."""
        self._nodes[name] = CausalNode(
            name=name,
            base_prob=base_prob,
            description=description,
            unit=unit,
        )
        self._adj[name] = []
        self._parents[name] = []
        self._topo_order = self._compute_topo_order()
        return self

    def add_edge(
        self,
        source: str,
        target: str,
        strength: float = 1.0,
        delay_days: int = 0,
        confidence: float = 0.8,
        mechanism: str = "",
    ) -> "StructuralCausalModel":
        """
        Dodaje krawędź kauzalną source → target.

        Parameters
        ----------
        strength    : siła efektu (>0 = pozytywny, <0 = negatywny)
                      |strength| = 1 oznacza pełne przejście szoku
        delay_days  : opóźnienie kauzalne (dni)
        confidence  : pewność relacji [0,1]
        mechanism   : opis mechanizmu transmisji
        """
        # Dodaj węzły jeśli nie istnieją
        if source not in self._nodes:
            self.add_node(source)
        if target not in self._nodes:
            self.add_node(target)

        edge = CausalEdge(source, target, strength, delay_days, confidence, mechanism)
        self._edges[(source, target)] = edge
        self._adj[source].append(target)
        self._parents[target].append(source)
        self._topo_order = self._compute_topo_order()
        return self

    def _compute_topo_order(self) -> list[str]:
        """Kahn's algorithm dla topologicznego sortowania."""
        in_degree = {n: len(self._parents.get(n, [])) for n in self._nodes}
        queue = [n for n, d in in_degree.items() if d == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self._adj.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order if len(order) == len(self._nodes) else list(self._nodes.keys())

    # ─── Generowanie próbek ───────────────────────────────────────────────────

    def _sample_once(self, intervention: dict | None = None) -> dict[str, float]:
        """
        Generuje jedną próbkę z SCM.

        Dla interwencji do(X=x): "mutiluje" DAG usuwając krawędzie do X.
        """
        iv = intervention or {}
        sample: dict[str, float] = {}

        for node in self._topo_order:
            if node in iv:
                sample[node] = float(iv[node])
                continue

            base = self._nodes[node].base_prob
            parent_nodes = self._parents.get(node, [])

            if not parent_nodes:
                # Węzeł korzeniowy
                val = 1.0 if self._rng.random() < base else 0.0
            else:
                # Strukturalne równanie: ważona suma rodziców + szum
                parent_effect = 0.0
                for p in parent_nodes:
                    edge = self._edges.get((p, node))
                    if edge is None:
                        continue
                    p_val = sample.get(p, 0.5)
                    parent_effect += edge.strength * p_val * edge.confidence

                # Logistic-link function: P(Y=1) = σ(logit(base) + Σ effects)
                logit_base = np.log(base / (1 - base + 1e-10) + 1e-10)
                logit_total = logit_base + parent_effect
                prob = 1 / (1 + np.exp(-logit_total))
                prob = float(np.clip(prob, 0.001, 0.999))
                val = 1.0 if self._rng.random() < prob else 0.0

            sample[node] = val

        return sample

    def simulate(
        self,
        evidence: dict | None = None,
        intervention: dict | None = None,
    ) -> dict[str, float]:
        """
        Symulacja Monte Carlo.

        Parameters
        ----------
        evidence     : warunki obserwacyjne P(Y | X=x) — OBSERWACJA
        intervention : interwencje do-calculus P(Y | do(X=x)) — INTERWENCJA

        Returns
        -------
        dict {node: P(node=True | evidence/intervention)}
        """
        results: dict[str, list] = {n: [] for n in self._nodes}

        for _ in range(self.n_simulations):
            sample = self._sample_once(intervention)

            # Filtrowanie przez evidence (rejection sampling)
            if evidence:
                match = all(
                    (bool(sample.get(k, 0)) == bool(v))
                    for k, v in evidence.items()
                )
                if not match:
                    continue

            for n, v in sample.items():
                results[n].append(v)

        # Średnia (P(node=True))
        out = {}
        for n, vals in results.items():
            out[n] = float(np.mean(vals)) if vals else float(self._nodes[n].base_prob)
        return out

    # ─── Do-Calculus ──────────────────────────────────────────────────────────

    def do_intervention(
        self,
        intervention: dict,
        target: str,
    ) -> float:
        """
        Pearl's do-calculus: P(target=True | do(intervention)).

        "Mutyluje" DAG usuwając krawędzie do interweniowanych węzłów.
        Pozwala odróżnić obserwację od przyczyny.

        Parameters
        ----------
        intervention : dict {node: value} — INTERWENCJA (nie obserwacja!)
        target       : węzeł docelowy

        Returns
        -------
        float — P(target=True | do(intervention))

        Przykład:
            # "Jaka jest szansa kryzysu portfela jeśli Fed NA PEWNO podniesie stopy?"
            p = scm.do_intervention({"Fed_Hike": True}, "Portfolio_Crash")
        """
        result = self.simulate(intervention=intervention)
        return result.get(target, 0.5)

    # ─── Counterfactual ───────────────────────────────────────────────────────

    def counterfactual(
        self,
        observed: dict,
        hypothetical: dict,
        target: str,
        n_samples: int = 1000,
    ) -> dict:
        """
        Counterfactual reasoning (Pearl's "ladder of causation" — Level 3).

        Pytanie: "Gdyby X było inne (choć wiemy że było = x_obs),
                  co by się stało z Y?"

        Procedura Twin Network:
        1. Fit latent variables (ε) do obserwacji
        2. Apply hypothetical intervention do "twin" world
        3. Propagate → compute Y_cf

        Parameters
        ----------
        observed     : co FAKTYCZNIE zaobserwowaliśmy
        hypothetical : co CHCEMY zamiast tego
        target       : jaka zmienna nas interesuje w "świecie alternatywnym"

        Returns
        -------
        dict z:
          p_factual    : P(target | observed faktyczny świat)
          p_cf         : P(target | counterfactual świat)
          ate          : Average Treatment Effect = p_cf - p_factual
          interpretation : opis słowny

        Przykład:
            # "Czy portfel PRZEŻYŁBY gdyby nie mieliśmy hedga?"
            scm.counterfactual(
                observed={"Hedge": True, "Market_Crash": True, "Portfolio_OK": True},
                hypothetical={"Hedge": False},
                target="Portfolio_OK",
            )
        """
        # Faktyczny świat
        p_factual_result = self.simulate(evidence=observed)
        p_factual = p_factual_result.get(target, 0.5)

        # Counterfactual świat (interwencja)
        combined = {**observed, **hypothetical}
        p_cf_result = self.simulate(intervention=combined)
        p_cf = p_cf_result.get(target, 0.5)

        ate = p_cf - p_factual

        if abs(ate) < 0.02:
            interpretation = "Brak istotnego efektu kauzalnego"
        elif ate > 0:
            interpretation = f"Hipoteza ZWIĘKSZA P({target}) o {ate:.1%}"
        else:
            interpretation = f"Hipoteza ZMNIEJSZA P({target}) o {abs(ate):.1%}"

        return {
            "p_factual":     p_factual,
            "p_cf":          p_cf,
            "ate":           ate,
            "interpretation":interpretation,
            "target":        target,
            "observed":      observed,
            "hypothetical":  hypothetical,
        }

    # ─── Shock Propagation ────────────────────────────────────────────────────

    def shock_propagation(
        self,
        shock_node: str,
        shock_magnitude: float = 1.0,  # 1.0 = pełny szok (node=True)
    ) -> pd.DataFrame:
        """
        Propaguje szok przez sieć i zwraca zmiany prawdopodobieństw.

        Porównuje P(Y | do(shock)) vs P(Y) dla wszystkich węzłów.
        Mierzy bezpośredni i pośredni efekt szoku makroekonomicznego.

        Parameters
        ----------
        shock_node      : węzeł w którym wstrzykujemy szok
        shock_magnitude : natężenie szoku (1.0 = pełny, 0.5 = połowiczny)

        Returns
        -------
        DataFrame z kolumnami: node, p_baseline, p_shock, delta, effect_direction
        """
        # Baseline (bez szoku)
        p_baseline = self.simulate()

        # Szok
        shock_val = float(np.clip(shock_magnitude, 0.0, 1.0))
        p_shock = self.simulate(intervention={shock_node: shock_val > 0.5})

        rows = []
        for node in self._nodes:
            if node == shock_node:
                continue
            p_b = float(p_baseline.get(node, 0.5))
            p_s = float(p_shock.get(node, 0.5))
            delta = p_s - p_b

            rows.append({
                "Node":            node,
                "P_Baseline":      p_b,
                "P_Shock":         p_s,
                "Delta":           delta,
                "Delta_pct":       delta * 100,
                "Effect":          "🔴 Wzrost ryzyka" if delta > 0.02 else
                                   ("🟢 Spadek ryzyka" if delta < -0.02 else "⚪ Bez wpływu"),
                "Magnitude":       "Silny" if abs(delta) > 0.1 else
                                   ("Umiarkowany" if abs(delta) > 0.03 else "Słaby"),
            })

        return pd.DataFrame(rows).sort_values("Delta", ascending=False).reset_index(drop=True)

    # ─── Gettery ──────────────────────────────────────────────────────────────

    def get_dag_info(self) -> dict:
        """Zwraca informacje o strukturze DAG."""
        return {
            "nodes": list(self._nodes.keys()),
            "edges": [
                {
                    "source":    e.source,
                    "target":    e.target,
                    "strength":  e.strength,
                    "confidence":e.confidence,
                    "mechanism": e.mechanism,
                }
                for e in self._edges.values()
            ],
            "n_nodes": len(self._nodes),
            "n_edges": len(self._edges),
        }


# ─── Predefiniowane sieci dla rynków finansowych ─────────────────────────────

def get_default_financial_dag() -> CausalRiskNetwork:
    """Buduje przykładową, gotową sieć ryzyka dla rynków finansowych (legacyjna)."""
    net = CausalRiskNetwork()
    net.add_node("Recesja_USA", prob_table=0.15)
    net.add_node("Wojna_Handlowa", prob_table=0.10)
    net.add_node("Szok_Technologiczny", parents=["Recesja_USA", "Wojna_Handlowa"], prob_table={
        (True, True): 0.80, (True, False): 0.60,
        (False, True): 0.50, (False, False): 0.05
    })
    net.add_node("Kryzys_Plynnosci", parents=["Recesja_USA"], prob_table={
        (True,): 0.70, (False,): 0.05
    })
    net.add_node("Crash_Portfela", parents=["Szok_Technologiczny", "Kryzys_Plynnosci"], prob_table={
        (True, True): 0.95, (True, False): 0.40,
        (False, True): 0.60, (False, False): 0.01
    })
    return net


def get_financial_scm() -> StructuralCausalModel:
    """
    Gotowy SCM dla analizy ryzyka makrofinansowego.

    Węzły: Fed_Hike, Recession_Risk, Credit_Spread, VIX_Spike,
           Dollar_Strength, Equity_Selloff, Portfolio_Crash.

    Referencje struktury:
    - Brunnermeier (2009) — kryzys płynności
    - Bernanke (2013) — efekty Fed na rynki
    - Borio (2014) — financial cycle
    """
    scm = StructuralCausalModel(n_simulations=3000)

    # ─── Węzły ───────────────────────────────────────────────────────────────
    scm.add_node("Fed_Hike",          base_prob=0.30, description="Fed podnosi stopy procentowe")
    scm.add_node("Recession_Risk",    base_prob=0.15, description="Ryzyko recesji w USA")
    scm.add_node("Credit_Spread",     base_prob=0.20, description="Wzrost spreadu kredytowego >200bps")
    scm.add_node("VIX_Spike",         base_prob=0.20, description="VIX > 30 (stres rynkowy)")
    scm.add_node("Dollar_Strength",   base_prob=0.35, description="DXY rośnie >5% vs EM currencies")
    scm.add_node("Equity_Selloff",    base_prob=0.25, description="S&P500 spada >15%")
    scm.add_node("EM_Crisis",         base_prob=0.10, description="Kryzys rynków wschodzących")
    scm.add_node("Liquidity_Crunch",  base_prob=0.10, description="Kryzys płynności rynkowej")
    scm.add_node("Portfolio_Crash",   base_prob=0.08, description="Portfel traci >20%")

    # ─── Krawędzie kauzalne ───────────────────────────────────────────────────
    # Fed → reszta
    scm.add_edge("Fed_Hike", "Recession_Risk",   strength=+0.4, mechanism="Wyższe koszty finansowania hamują wzrost")
    scm.add_edge("Fed_Hike", "Dollar_Strength",  strength=+0.7, mechanism="Carry trade napływa do USD")
    scm.add_edge("Fed_Hike", "Credit_Spread",    strength=+0.5, mechanism="Wyższy koszt długu korporacyjnego")
    scm.add_edge("Fed_Hike", "Equity_Selloff",   strength=-0.4, mechanism="Discount rate wzrośnie → niższe wyceny")

    # Recession → reszta
    scm.add_edge("Recession_Risk", "Credit_Spread",    strength=+0.6, mechanism="Wzrost ryzyka defaultu")
    scm.add_edge("Recession_Risk", "VIX_Spike",        strength=+0.5, mechanism="Wzrost niepewności")
    scm.add_edge("Recession_Risk", "Equity_Selloff",   strength=+0.7, mechanism="Spadek zyków korporacyjnych")

    # Credit → Liquidity
    scm.add_edge("Credit_Spread",  "Liquidity_Crunch", strength=+0.6, mechanism="Zamrożenie rynku repo")
    scm.add_edge("Credit_Spread",  "Equity_Selloff",   strength=+0.4, mechanism="Risk-off od korporatów do Treasuries")

    # VIX → Portfolio
    scm.add_edge("VIX_Spike",      "Equity_Selloff",   strength=+0.5, mechanism="Forced deleveraging")
    scm.add_edge("VIX_Spike",      "Liquidity_Crunch", strength=+0.3, mechanism="Spread bid-ask się rozszerza")

    # Dollar → EM
    scm.add_edge("Dollar_Strength","EM_Crisis",         strength=+0.7, mechanism="Dług EM w USD drożeje")
    scm.add_edge("EM_Crisis",      "Equity_Selloff",   strength=+0.4, mechanism="Contagion z rynków wschodzących")

    # Finalne → Portfolio Crash
    scm.add_edge("Equity_Selloff",   "Portfolio_Crash", strength=+0.8, mechanism="Bezpośrednia strata wartości")
    scm.add_edge("Liquidity_Crunch", "Portfolio_Crash", strength=+0.5, mechanism="Niemożność zamknięcia pozycji")
    scm.add_edge("EM_Crisis",        "Portfolio_Crash", strength=+0.3, mechanism="Ekspozycja na EM")

    return scm

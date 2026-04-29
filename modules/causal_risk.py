import numpy as np

class CausalRiskNetwork:
    """
    Lekki silnik wnioskowania w Sieciach Bayesowskich (DAG) oparty na symulacji Monte Carlo.
    Zaprojektowany do propagacji wstrząsów (Shock Propagation) w modelowaniu ryzyka.
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
        # Topological sort nie jest tu ściśle potrzebny jeśli oceniamy rodziców rekurencyjnie (lub leniwie)
        # Aby ułatwić, zakładamy że zdefiniowano je w kolejności topologicznej.
        
        for node, data in self.nodes.items():
            if node in evidence:
                sample[node] = evidence[node]
                continue
                
            parents = data['parents']
            if not parents:
                # Węzeł korzenia
                p_true = data['prob_table']
            else:
                # Zbuduj krotkę stanów rodziców
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
                    
        # Normalizacja do prawdopodobieństw
        for node in results:
            results[node] /= num_samples
            
        return results

def get_default_financial_dag():
    """Buduje przykładową, gotową sieć ryzyka dla rynków finansowych."""
    net = CausalRiskNetwork()
    
    # Warstwa Makro
    net.add_node("Recesja_USA", prob_table=0.15)
    net.add_node("Wojna_Handlowa", prob_table=0.10)
    
    # Warstwa Sektorowa (Rodzice: Recesja, Wojna)
    # Tabela: {(Recesja, Wojna): P_True}
    net.add_node("Szok_Technologiczny", parents=["Recesja_USA", "Wojna_Handlowa"], prob_table={
        (True, True): 0.80,
        (True, False): 0.60,
        (False, True): 0.50,
        (False, False): 0.05
    })
    
    net.add_node("Kryzys_Plynnosci", parents=["Recesja_USA"], prob_table={
        (True,): 0.70,
        (False,): 0.05
    })
    
    # Warstwa Portfela
    net.add_node("Crash_Portfela", parents=["Szok_Technologiczny", "Kryzys_Plynnosci"], prob_table={
        (True, True): 0.95,
        (True, False): 0.40,
        (False, True): 0.60,
        (False, False): 0.01
    })
    
    return net

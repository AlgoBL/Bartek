import numpy as np
from modules.simulation import simulate_barbell_strategy, calculate_metrics

def test_simulation():
    print("Running Simulation Test...")
    try:
        # Test 1: Basic Run
        paths = simulate_barbell_strategy(n_years=5, n_simulations=100)
        assert paths.shape == (100, 5*252 + 1), f"Shape mismatch: {paths.shape}"
        print("✅ Basic Shape Check Passed")

        # Test 2: Safe Asset Only (Allocation 100%)
        # Should grow deterministicly at 5.51%
        paths_safe = simulate_barbell_strategy(n_years=1, n_simulations=10, alloc_safe=1.0)
        start_val = 10000
        expected_end = start_val * (1.0551)
        # Allow small floating point error
        final_val = np.mean(paths_safe[:, -1])
        assert np.isclose(final_val, expected_end, rtol=1e-3), f"Safe Asset growth wrong: Got {final_val}, Expected {expected_end}"
        print("✅ Safe Asset Growth Check Passed (5.51%)")
        
        # Test 3: Threshold Rebalancing runs without error
        paths_thresh = simulate_barbell_strategy(rebalance_strategy="Threshold", threshold_percent=0.1)
        print("✅ Threshold Rebalancing Check Passed")
        
        metrics = calculate_metrics(paths, 5)
        print(f"✅ Metrics Calculation Passed: CAGR={metrics['mean_cagr']:.2%}")
        
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simulation()

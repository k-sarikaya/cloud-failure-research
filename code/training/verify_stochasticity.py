"""
Verify Stochasticity in EdgeEnvV2
=================================
Ensures that sequential calls to make_scenario() using the same RNG
generate different network snapshots (different capacities and loads).
"""

import numpy as np
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.edge_env_v2 import make_scenario

def verify():
    rng = np.random.default_rng(42)
    
    print("Generating Scenario 1...")
    s1 = make_scenario(
        n_nodes=100, topo_type="ba", ba_m=3, max_steps=480,
        capacity_range=(50, 500), init_util_range=(0.4, 0.7), rng=rng
    )
    
    print("Generating Scenario 2...")
    s2 = make_scenario(
        n_nodes=100, topo_type="ba", ba_m=3, max_steps=480,
        capacity_range=(50, 500), init_util_range=(0.4, 0.7), rng=rng
    )
    
    # Check capacities
    cap_diff = np.sum(np.abs(s1.capacities - s2.capacities))
    load_diff = np.sum(np.abs(s1.loads - s2.loads))
    fail_diff = np.sum(np.abs(s1.fail_uniforms - s2.fail_uniforms))
    
    print("\n" + "="*40)
    print("STOCHASTICITY VERIFICATION")
    print("="*40)
    print(f"Scenario 1 Total Capacity: {s1.capacities.sum():.2f}")
    print(f"Scenario 2 Total Capacity: {s2.capacities.sum():.2f}")
    print(f"Capacity Diff (L1 Norm): {cap_diff:.2f}")
    print(f"Load Diff (L1 Norm):     {load_diff:.2f}")
    print(f"Failure Prob Diff:      {fail_diff:.2f}")
    
    if cap_diff > 0 and load_diff > 0:
        print("\nSUCCESS: Scenarios are unique across iterations.")
    else:
        print("\nFAILURE: Scenarios are identical!")

if __name__ == "__main__":
    verify()

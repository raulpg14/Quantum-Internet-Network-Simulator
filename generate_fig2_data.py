import time
import numpy as np
from qcn.engine.simulation import run_simulation
from qcn.engine.results_store import save_run
from qcn.engine.config import (
    NETWORK_TYPE_OFBQI, NETWORK_TYPE_SBQI, SIM_MODE_DISTRIBUTION
)

MC_REPS = 100
SEED    = 42

configs = [
    # Panel (a): SBQI, fixed rho=5e-3, vary N
    {"net": NETWORK_TYPE_SBQI, "nodes": 1000,  "rho": 5e-3,  "panel": "a"},
    {"net": NETWORK_TYPE_SBQI, "nodes": 10000, "rho": 5e-3,  "panel": "a"},
    #{"net": NETWORK_TYPE_SBQI, "nodes": 50000, "rho": 5e-3,  "panel": "a"},

    # Panel (b): SBQI, fixed N=1000, vary rho
    {"net": NETWORK_TYPE_SBQI, "nodes": 1000, "rho": 1e-4, "panel": "b"},
    {"net": NETWORK_TYPE_SBQI, "nodes": 1000, "rho": 2e-4, "panel": "b"},
    {"net": NETWORK_TYPE_SBQI, "nodes": 1000, "rho": 4e-4, "panel": "b"},

    # Panel (c): both types, N=1000, rho=2e-4
    {"net": NETWORK_TYPE_OFBQI, "nodes": 1000, "rho": 2e-4, "panel": "c"},
    {"net": NETWORK_TYPE_SBQI,  "nodes": 1000, "rho": 2e-4, "panel": "c"},
]

for cfg in configs:
    N   = cfg["nodes"]
    rho = cfg["rho"]
    R   = np.sqrt(N / (np.pi * rho))
    net = cfg["net"]

    print(f"Panel ({cfg['panel']}) | {net} | N={N} | rho={rho:.1e} | R={R:.1f}km ...")
    t0 = time.time()

    res = run_simulation({
        "nodes":    N,
        "radius":   R,
        "type":     net,
        "mc_iter":  MC_REPS,
        "sim_mode": SIM_MODE_DISTRIBUTION,
        "seed":     SEED,
    })

    duration = time.time() - t0

    if res.get("success"):
        run_id = save_run(res, duration_s=duration)
        print(f"  Saved run_id={run_id} | {duration:.1f}s")
    else:
        print(f"  ERROR: {res.get('error')}")
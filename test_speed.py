import time
from qcn.engine.simulation import run_simulation
from qcn.engine.config import NETWORK_TYPE_SBQI, SIM_MODE_EVOLUTION

params = {
    "nodes": 500, "radius": 2, "type": NETWORK_TYPE_SBQI,
    "mc_iter": 1, "nets_per_mc": 5, "rad_incr": 200,
    "sim_mode": SIM_MODE_EVOLUTION, "seed": 42,
}

t0 = time.time()
run_simulation({**params, "approx_path_samples": 0})
print(f"Exact:  {time.time()-t0:.1f}s")

t0 = time.time()
run_simulation({**params, "approx_path_samples": 200})
print(f"Approx: {time.time()-t0:.1f}s")
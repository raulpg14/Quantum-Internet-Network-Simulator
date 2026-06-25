import numpy as np
from qcn.engine.math_util import log_func, power_func
from qcn.engine.simulation import run_simulation
from qcn.engine.config import NETWORK_TYPE_OFBQI, NETWORK_TYPE_SBQI, SIM_MODE_EVOLUTION

# Test functions importable
x = np.array([200, 400, 600, 800, 1000], dtype=float)
print("log_func:", log_func(x, 0.17, 1.4))
print("power_func:", power_func(x, 0.4, 0.45))

# Test simulation still works
for net_type in [NETWORK_TYPE_OFBQI, NETWORK_TYPE_SBQI]:
    res = run_simulation({
        "nodes": 200, "radius": 2, "type": net_type,
        "mc_iter": 1, "nets_per_mc": 4, "rad_incr": 100,
        "sim_mode": SIM_MODE_EVOLUTION, "seed": 42,
    })
    fp = res.get("fit_params")
    print(f"{net_type} ({fp['type']}): {fp['formula']}")
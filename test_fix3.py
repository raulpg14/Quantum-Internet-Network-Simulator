import time
from qcn.engine.simulation import run_simulation
from qcn.engine.results_store import save_run, get_metrics
from qcn.engine.config import NETWORK_TYPE_OFBQI, SIM_MODE_DISTRIBUTION, DEFAULT_RADIUS_DISTRIBUTION

res = run_simulation({
    "nodes": 200,
    "radius": DEFAULT_RADIUS_DISTRIBUTION,
    "type": NETWORK_TYPE_OFBQI,
    "mc_iter": 2,
    "sim_mode": SIM_MODE_DISTRIBUTION,
    "seed": 42,
})

print(f"mean_degree: {res.get('mean_degree')}")
print(f"clustering:  {res.get('clustering')}")
print(f"ng_ratio:    {res.get('ng_ratio')}")

run_id = save_run(res, duration_s=0.0)
metrics = get_metrics(run_id)
metric_names = [m['metric'] for m in metrics]
print(f"Saved metrics: {metric_names}")
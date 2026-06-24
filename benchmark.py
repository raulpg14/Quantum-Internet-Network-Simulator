import time
import numpy as np
import cProfile
import pstats
import io
from qcn.engine.network import Network
from qcn.engine.config import NETWORK_TYPE_OFBQI, NETWORK_TYPE_SBQI

def benchmark_network(net_type, n_nodes, radius, label):
    print(f"\n{'='*55}")
    print(f"{label} | {net_type} | N={n_nodes} | R={radius:.0f}")
    print(f"{'='*55}")

    # --- add_nodes ---
    t0 = time.perf_counter()
    net = Network()
    net.add_nodes(n_nodes, radius, seed=42)
    t_nodes = time.perf_counter() - t0
    print(f"  add_nodes:      {t_nodes*1000:.1f} ms")

    # --- connect_nodes ---
    t0 = time.perf_counter()
    net.connect_nodes(net_type)
    t_connect = time.perf_counter() - t0
    print(f"  connect_nodes:  {t_connect*1000:.1f} ms  ({len(net.edges)} edges)")

    # --- to_networkx ---
    t0 = time.perf_counter()
    G, pos = net.to_networkx()
    t_nx = time.perf_counter() - t0
    print(f"  to_networkx:    {t_nx*1000:.1f} ms")

    # --- giant component ---
    import networkx as nx
    t0 = time.perf_counter()
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_calc = G.subgraph(largest_cc)
    else:
        G_calc = G
    t_gcc = time.perf_counter() - t0
    print(f"  giant component:{t_gcc*1000:.1f} ms  ({len(G_calc)} nodes)")

    # --- path length ---
    t0 = time.perf_counter()
    if len(G_calc) > 1:
        l = nx.average_shortest_path_length(G_calc)
    t_path = time.perf_counter() - t0
    print(f"  avg path length:{t_path*1000:.1f} ms  (l={l:.3f})")

    # --- diameter ---
    t0 = time.perf_counter()
    if len(G_calc) > 1:
        d = nx.diameter(G_calc)
    t_diam = time.perf_counter() - t0
    print(f"  diameter:       {t_diam*1000:.1f} ms  (d={d})")

    total = t_nodes + t_connect + t_nx + t_gcc + t_path + t_diam
    print(f"  {'─'*40}")
    print(f"  TOTAL:          {total*1000:.1f} ms")
    print(f"  path+diam:      {(t_path+t_diam)*100/total:.0f}% of total")

    return {
        "net_type": net_type,
        "n_nodes": n_nodes,
        "t_nodes": t_nodes,
        "t_connect": t_connect,
        "t_nx": t_nx,
        "t_path": t_path,
        "t_diam": t_diam,
        "total": total,
    }

# ── Configs to benchmark ───────────────────────────────────────────────────
configs = [
    (NETWORK_TYPE_OFBQI, 1000,  1261,  "small"),
    (NETWORK_TYPE_SBQI,  1000,  1261,  "small"),
    (NETWORK_TYPE_OFBQI, 5000,  2821,  "medium"),
    (NETWORK_TYPE_SBQI,  5000,  2821,  "medium"),
    (NETWORK_TYPE_OFBQI, 10000, 3989,  "large"),
    (NETWORK_TYPE_SBQI,  10000, 3989,  "large"),
]

results = []
for net_type, n, r, label in configs:
    res = benchmark_network(net_type, n, r, label)
    results.append(res)

# ── Summary table ──────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"{'Type':<8} {'N':>7} {'connect':>10} {'path+diam':>12} {'total':>10}")
print(f"{'─'*55}")
for r in results:
    pd_ms = (r['t_path'] + r['t_diam']) * 1000
    print(f"{r['net_type']:<8} {r['n_nodes']:>7} "
          f"{r['t_connect']*1000:>9.1f}ms "
          f"{pd_ms:>11.1f}ms "
          f"{r['total']*1000:>9.1f}ms")
import numpy as np
import networkx as nx
import logging
from scipy.optimize import curve_fit

from qcn.engine.network import Network
from qcn.engine.config import (
    DEFAULT_FALLBACK_RADIUS,
    DEFAULT_MC_REPS,
    DEFAULT_NODE_INCREMENT,
    DEFAULT_STEPS,
    POISSON_DENSITY,
    SIM_MODE_DISTRIBUTION,
    SIM_MODE_EVOLUTION,
    NETWORK_TYPE_SBQI,
)

from qcn.engine.math_util import log_normal_fit, poisson_fit, log_func, power_func

logger = logging.getLogger(__name__)


def _get_giant_component(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component of G, or G itself if already connected."""
    if nx.is_connected(G):
        return G
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc)


def _compute_path_metrics(G: nx.Graph) -> tuple[float, float]:
    """
    Compute average shortest path length and diameter on a graph.
    Returns (0, 0) if the graph has fewer than 2 nodes.
    """
    if len(G) < 2:
        return 0.0, 0.0
    return nx.average_shortest_path_length(G), nx.diameter(G)


def _compute_clustering(G: nx.Graph) -> float:
    """Compute average clustering coefficient. Returns 0 for graphs with < 2 nodes."""
    if len(G) < 2:
        return 0.0
    return nx.average_clustering(G)


def _compute_mean_degree(G: nx.Graph) -> float:
    """Compute average degree ⟨k⟩ of the graph."""
    if len(G) == 0:
        return 0.0
    degrees = [d for _, d in G.degree()]
    return float(np.mean(degrees))


def _fit_logarithmic(x: list, y: list) -> dict | None:
    """
    Fit logarithmic curve l = a*ln(N) + b for SBQI.
    PRX Quantum 2021: SBQI displays small-world property, l ~ ln(N).
    """
    try:
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        popt, _ = curve_fit(log_func, x_arr, y_arr)
        sign = "+" if popt[1] >= 0 else "-"
        return {
            "type":    "logarithmic",
            "a":       popt[0],
            "b":       popt[1],
            "formula": f"l = {popt[0]:.4f} * ln(N) {sign} {abs(popt[1]):.4f}"
        }
    except Exception as e:
        logger.warning("Logarithmic fit failed: %s", e)
        return None


def _fit_powerlaw(x: list, y: list, density: float) -> dict | None:
    """
    Fit power law curve l = b * N^alpha for OFBQI.
    PRL 2020: OFBQI has NO small-world property, l ~ sqrt(N).
    Expected: alpha ~ 0.5, b ~ 5e-5 / density.
    """
    try:
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        popt, _ = curve_fit(power_func, x_arr, y_arr, p0=[5e-5, 0.5])
        return {
            "type":    "powerlaw",
            "b":       popt[0],
            "alpha":   popt[1],
            "formula": f"l = {popt[0]:.6f} * N^{popt[1]:.4f}"
        }
    except Exception as e:
        logger.warning("Power law fit failed: %s", e)
        return None


def run_simulation(data: dict) -> dict:
    """
    Main simulation controller. Accepts a configuration dict and returns results.

    Required keys:
        nodes       (int)   : number of nodes
        radius      (float) : network radius in km (distribution mode)
                              or density coefficient × 1e-4 (evolution mode)
        type        (str)   : network type — 'OFBQI' or 'SBQI'

    Optional keys:
        mc_iter     (int)   : Monte Carlo repetitions (default: DEFAULT_MC_REPS)
        sim_mode    (str)   : 'Degree Distribution' or 'Evolution (N)'
        nets_per_mc (int)   : number of N steps in evolution mode
        rad_incr    (int)   : node increment per step in evolution mode
        seed        (int)   : random seed for reproducibility
        stop_event          : threading.Event for cancellation
        queue               : queue.Queue for progress messages
    """
    seed = None
    try:
        # --- Parse inputs ---
        input_nodes_val = int(data['nodes'])
        input_param_val = float(data['radius'])
        net_type        = data['type']
        mc_reps         = int(data.get('mc_iter', DEFAULT_MC_REPS))
        seed            = data.get('seed', None)
        sim_mode        = data.get('sim_mode', SIM_MODE_DISTRIBUTION)
        stop_event      = data.get('stop_event', None)
        progress_queue  = data.get('queue', None)

        def update_progress(msg: str) -> None:
            if progress_queue:
                progress_queue.put(msg)

        def is_cancelled() -> bool:
            return stop_event is not None and stop_event.is_set()

        # =========================================================
        # MODE 1: DEGREE DISTRIBUTION (fixed radius)
        # =========================================================
        if sim_mode == SIM_MODE_DISTRIBUTION:
            radius = input_param_val
            last_G, last_pos, last_degrees = None, None, []
            clustering_vals = []
            mean_degree_vals = []

            for rep in range(mc_reps):
                if is_cancelled():
                    return {"cancelled": True, "seed": seed}
                update_progress(f"Distribution: rep {rep + 1}/{mc_reps}")

                net = Network()
                net.add_nodes(input_nodes_val, radius, seed=seed)
                net.connect_nodes(net_type)

                raw_degrees = [node.connections for node in net.nodes.values()]
                G_temp, pos_temp = net.to_networkx()

                # Compute per-rep metrics on giant component
                G_calc = _get_giant_component(G_temp)
                clustering_vals.append(_compute_clustering(G_calc))
                mean_degree_vals.append(_compute_mean_degree(G_temp))

                if rep == mc_reps - 1:
                    last_G, last_pos = G_temp, pos_temp
                    last_degrees = raw_degrees

            if last_degrees:
                counts, bins = np.histogram(
                    last_degrees, bins=range(0, max(last_degrees) + 2)
                )
                dist_y = counts / len(last_degrees)
                dist_x = bins[:-1]
            else:
                dist_x, dist_y = [], []

            final_area   = np.pi * (radius ** 2)
            density_val  = input_nodes_val / final_area if final_area > 0 else 0.0
            ng_ratio     = len(_get_giant_component(last_G)) / len(last_G) if last_G else 0.0

            return {
                "success":       True,
                "mode":          "distribution",
                "G":             last_G,
                "pos":           last_pos,
                "dist_x":        dist_x,
                "dist_y":        dist_y,
                "type":          net_type,
                "final_radius":  radius,
                "final_n":       input_nodes_val,
                "density_val":   density_val,
                "mean_degree":   float(np.mean(mean_degree_vals)),
                "clustering":    float(np.mean(clustering_vals)),
                "ng_ratio":      ng_ratio,
                "seed":          seed,
            }

        # =========================================================
        # MODE 2: EVOLUTION (N) — fixed density, increasing N
        # =========================================================
        elif sim_mode == SIM_MODE_EVOLUTION:
            update_progress("Starting evolution sweep at fixed density...")

            # User enters density coefficient e.g. "2" → real density = 2 × 1e-4
            density_coeff = input_param_val
            real_density  = density_coeff * POISSON_DENSITY

            steps_count = int(data.get('nets_per_mc', DEFAULT_STEPS))
            node_incr   = int(data.get('rad_incr', DEFAULT_NODE_INCREMENT))
            steps_n     = [input_nodes_val + i * node_incr for i in range(steps_count)]

            results_n           = []
            results_path        = []
            results_diam        = []
            results_clustering  = []
            results_mean_degree = []
            results_ng_ratio    = []
            last_G, last_pos    = None, None
            final_r_viz         = 0.0

            for idx, n_count in enumerate(steps_n):
                if is_cancelled():
                    return {"cancelled": True, "seed": seed}

                # Derive radius from density: R = sqrt(N / (π × ρ))
                dynamic_radius = (
                    np.sqrt(n_count / (np.pi * real_density))
                    if real_density > 0
                    else DEFAULT_FALLBACK_RADIUS
                )

                temp_path        = []
                temp_diam        = []
                temp_clustering  = []
                temp_mean_degree = []
                temp_ng_ratio    = []

                for rep in range(mc_reps):
                    if is_cancelled():
                        return {"cancelled": True, "seed": seed}

                    update_progress(
                        f"N={n_count} | R={dynamic_radius:.1f} | rep {rep + 1}/{mc_reps}"
                    )

                    net = Network()
                    net.add_nodes(int(n_count), dynamic_radius, seed=seed)
                    net.connect_nodes(net_type)
                    G_temp, pos_temp = net.to_networkx()

                    if len(G_temp) > 0:
                        G_calc   = _get_giant_component(G_temp)
                        l, d     = _compute_path_metrics(G_calc)
                        c        = _compute_clustering(G_calc)
                        k_mean   = _compute_mean_degree(G_temp)
                        ng_ratio = len(G_calc) / len(G_temp)
                    else:
                        l, d, c, k_mean, ng_ratio = 0.0, 0.0, 0.0, 0.0, 0.0

                    temp_path.append(l)
                    temp_diam.append(d)
                    temp_clustering.append(c)
                    temp_mean_degree.append(k_mean)
                    temp_ng_ratio.append(ng_ratio)

                    if idx == len(steps_n) - 1 and rep == mc_reps - 1:
                        last_G, last_pos = G_temp, pos_temp
                        final_r_viz = dynamic_radius

                results_n.append(n_count)
                results_path.append(float(np.mean(temp_path)))
                results_diam.append(float(np.mean(temp_diam)))
                results_clustering.append(float(np.mean(temp_clustering)))
                results_mean_degree.append(float(np.mean(temp_mean_degree)))
                results_ng_ratio.append(float(np.mean(temp_ng_ratio)))

            # SBQI: logarithmic fit l ~ ln(N) — small-world (PRX Quantum 2021)
            # OFBQI: power law fit l ~ N^alpha — no small-world (PRL 2020)
            if net_type == NETWORK_TYPE_SBQI:
                fit_params      = _fit_logarithmic(results_n, results_path)
                fit_params_diam = _fit_logarithmic(results_n, results_diam)
            else:
                fit_params      = _fit_powerlaw(results_n, results_path, real_density)
                fit_params_diam = _fit_powerlaw(results_n, results_diam, real_density)

            return {
                "success":          True,
                "mode":             "evolution",
                "G":                last_G,
                "pos":              last_pos,
                "x_nodes":          results_n,
                "y_path":           results_path,
                "y_diameter":       results_diam,
                "y_clustering":     results_clustering,
                "y_mean_degree":    results_mean_degree,
                "y_ng_ratio":       results_ng_ratio,
                "type":             net_type,
                "final_radius":     final_r_viz,
                "final_n":          steps_n[-1],
                "density_val":      real_density,
                "density_coeff":    density_coeff,
                "fit_params":       fit_params,
                "fit_params_diam":  fit_params_diam,
                "seed":             seed,
            }

    except Exception as e:
        logger.exception("Simulation failed")
        return {"error": str(e), "seed": seed}
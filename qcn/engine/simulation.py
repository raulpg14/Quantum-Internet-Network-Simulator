import numpy as np
import networkx as nx
import queue
import logging
from qcn.engine.network import Network
from qcn.engine.config import (
    DEFAULT_FALLBACK_RADIUS,
    DEFAULT_MC_REPS,
    DEFAULT_NODE_INCREMENT,
    DEFAULT_STEPS,
    POISSON_DENSITY,
    SIM_MODE_DISTRIBUTION,
    SIM_MODE_EVOLUTION,
)

logger = logging.getLogger(__name__)

def run_simulation(data: dict):
    """
    Controlador lógico de la simulación.
    """
    try:
        # Inputs básicos
        input_nodes_val = int(data['nodes']) 
        input_param_val = float(data['radius']) # En modo evolución, esto es el COEFICIENTE DE DENSIDAD
        net_type = data['type']
        mc_reps = int(data.get('mc_iter', DEFAULT_MC_REPS)) 
        seed = data.get('seed', None)
        
        sim_mode = data.get('sim_mode', SIM_MODE_DISTRIBUTION)
        
        stop_event = data.get('stop_event', None) 
        progress_queue = data.get('queue', None)
        
        def update_progress(msg):
            if progress_queue: progress_queue.put(msg)

        # =========================================================
        # MODO 1: DEGREE DISTRIBUTION (Radio Fijo)
        # =========================================================
        if sim_mode == SIM_MODE_DISTRIBUTION:
            # En este modo, el input es directamente el RADIO
            radius = input_param_val
            last_G, last_pos, last_degrees = None, None, []
            
            for rep in range(mc_reps):
                if stop_event and stop_event.is_set(): return {"cancelled": True, "seed": seed}
                update_progress(f"Distribution: Net {rep+1}/{mc_reps}")
                
                net = Network()
                net.add_nodes(input_nodes_val, radius, seed=seed)
                net.connect_nodes(net_type)
                
                raw_degrees = [node.connections for node in net.nodes.values()]
                
                if rep == mc_reps - 1:
                    last_G, last_pos = net.to_networkx()
                    last_degrees = raw_degrees

            if last_degrees:
                counts, bins = np.histogram(last_degrees, bins=range(0, max(last_degrees) + 2))
                dist_y = counts / len(last_degrees)
                dist_x = bins[:-1]
            else: dist_x, dist_y = [], []

            # Calculamos la densidad resultante para mostrarla en conclusiones
            final_area = np.pi * (radius**2)
            density_val = input_nodes_val / final_area if final_area > 0 else 0

            return {
                "success": True, 
                "mode": "distribution",
                "G": last_G, 
                "pos": last_pos,
                "dist_x": dist_x, 
                "dist_y": dist_y,
                "type": net_type, 
                "final_radius": radius,
                "final_n": input_nodes_val,
                "density_val": density_val,
                "seed": seed
            }

        # =========================================================
        # MODO 2: EVOLUTION (N) -> DENSIDAD CONSTANTE
        # =========================================================
        elif sim_mode == SIM_MODE_EVOLUTION:
            update_progress("Starting Evolution Sweep (N) with Fixed Density...")
            
            # --- CORRECCIÓN MATEMÁTICA ---
            # El usuario introduce "2", interpretamos 2 * 10^-4
            density_coeff = input_param_val
            real_density = density_coeff * POISSON_DENSITY  # 1e-4
            
            steps_count = int(data.get('nets_per_mc', DEFAULT_STEPS))
            node_incr = int(data.get('rad_incr', DEFAULT_NODE_INCREMENT))
            
            # Generar lista de Nodos a simular
            steps_n = [input_nodes_val + (i * node_incr) for i in range(steps_count)]
            
            results_n = []
            results_path = []  
            results_diam = []  
            
            last_G, last_pos = None, None 
            final_r_viz = 0

            for idx, n_count in enumerate(steps_n):
                if stop_event and stop_event.is_set(): return {"cancelled": True, "seed": seed}
                
                # --- CÁLCULO DINÁMICO DEL RADIO ---
                # R = sqrt( N / (pi * rho) )
                if real_density > 0:
                    dynamic_radius = np.sqrt(n_count / (np.pi * real_density))
                else:
                    dynamic_radius = DEFAULT_FALLBACK_RADIUS # Fallback por seguridad
                
                temp_path = []
                temp_diam = []
                
                for rep in range(mc_reps):
                    if stop_event and stop_event.is_set(): return {"cancelled": True, "seed": seed}

                    msg = f"N={n_count} | R={dynamic_radius:.1f} | Rep {rep+1}/{mc_reps}"
                    update_progress(msg)
                    
                    net = Network()
                    net.add_nodes(int(n_count), dynamic_radius, seed=seed)
                    net.connect_nodes(net_type)
                    
                    G_temp, _ = net.to_networkx()
                    
                    # Calcular métricas solo sobre la Componente Gigante
                    if len(G_temp) > 0:
                        if nx.is_connected(G_temp):
                            G_calc = G_temp
                        else:
                            largest_cc = max(nx.connected_components(G_temp), key=len)
                            G_calc = G_temp.subgraph(largest_cc)

                        if len(G_calc) > 1:
                            l = nx.average_shortest_path_length(G_calc)
                            d = nx.diameter(G_calc)
                        else:
                            l, d = 0, 0
                    else:
                        l, d = 0, 0

                    temp_path.append(l)
                    temp_diam.append(d)

                    # Guardamos la última red para visualizar
                    if idx == len(steps_n)-1 and rep == mc_reps-1:
                        last_G, last_pos = G_temp, _
                        final_r_viz = dynamic_radius

                results_n.append(n_count)
                results_path.append(np.mean(temp_path))
                results_diam.append(np.mean(temp_diam))
            
            from scipy.optimize import curve_fit

            # Definimos la función matemática esperada
            def log_func(x, a, b):
                return a * np.log(x) + b
            
            # Intentamos ajustar la curva a los resultados obtenidos
            fit_params = None
            try:
                popt, pcov = curve_fit(log_func, results_n, results_path)
                fit_params = {
                    "a": popt[0],
                    "b": popt[1],
                    "formula": f"l = {popt[0]:.4f} * ln(N) + {popt[1]:.4f}"
                }
            except Exception as e:
                logger.warning("No se pudo hacer el ajuste: %s", e)

            return {
                "success": True, 
                "mode": "evolution",
                "G": last_G, 
                "pos": last_pos,
                "x_nodes": results_n,
                "y_path": results_path,
                "y_diameter": results_diam,
                "type": net_type, 
                "final_radius": final_r_viz,
                "final_n": steps_n[-1],
                "density_val": real_density,    # Valor real (0.0002)
                "density_coeff": density_coeff,  # Valor input (2)
                "seed": seed
            }
            
    except Exception as e:
        logger.exception("Simulation failed")
        return {"error": str(e), "seed": seed if 'seed' in locals() else None}

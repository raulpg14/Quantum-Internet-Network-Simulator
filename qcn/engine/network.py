import numpy as np
import networkx as nx
import logging
from qcn.engine.node import Node
# Asegúrate de que las importaciones de tus fórmulas matemáticas funcionen
from qcn.engine.math_util import (optic_fiber_connection_probability,
                                  photonic_optic_fiber_connection_probability, photonic_optic_fiber_connection_probability_vec,
                                  satellite_connection_probability,
                                  satellite_connection_probability_vec,
                                  optic_fiber_connection_probability_vec,
                                  photonic_optic_fiber_connection_probability_vec,)
from qcn.engine.config import NETWORK_TYPE_OFBQI, NETWORK_TYPE_SBQI

from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)

class Network:
    def __init__(self):
        self.nodes = {}
        self.edges = set()
    
    # --- GETTERS ---
    def get_nodes(self):
        return self.nodes
    
    def get_position(self, id_node):
        return self.nodes.get(id_node)
    
    def get_all_positions(self):
        return [node.coordinates for node in self.nodes]
    
    def get_edges(self):
        return self.edges

    # --- LÓGICA DE GENERACIÓN ---
    
    def add_nodes(self, num_nodes, radius, seed=None):
        """Generación vectorizada de nodos (ya era rápida, pero mantenemos estructura)"""
        if seed is not None:
            np.random.seed(seed)

        r = np.sqrt(np.random.uniform(0, radius**2, num_nodes)) # Ojo: suele ser 0 a R^2
        theta = np.random.uniform(0, 2*np.pi, num_nodes)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Creamos los objetos Node (necesario por tu arquitectura actual)
        for i in range(num_nodes):
            self.nodes[i] = Node(i, x[i], y[i])
            
        # --- OPTIMIZACIÓN: Guardar matriz de posiciones ---
        # Forma (N, 2) -> [[x0, y0], [x1, y1], ...]
        self.pos_matrix = np.column_stack((x, y))
        self.ids_list = np.arange(num_nodes)

    def connect_nodes(self, network_type):
        """
        Lógica VECTORIZADA (100x más rápida que bucles for).
        """
        if self.pos_matrix is None:
            return

        num_nodes = len(self.nodes)
        if num_nodes < 2: return

        # 1. Calcular matriz de distancias (N x N)
        # pdist calcula distancias por pares, squareform la hace cuadrada
        dist_matrix = squareform(pdist(self.pos_matrix, 'euclidean'))
        
        # Matriz triangular superior para no repetir pares (i,j) y (j,i) ni (i,i)
        # k=1 excluye la diagonal
        triu_indices = np.triu_indices(num_nodes, k=1)
        
        # Extraemos las distancias relevantes (array 1D de tamaño N*(N-1)/2)
        distances = dist_matrix[triu_indices]
        
        # Generamos probabilidades aleatorias de una vez
        random_probs = np.random.uniform(0, 1, len(distances))
        
        # --- LÓGICA SEGÚN TIPO ---
        
        if network_type == NETWORK_TYPE_SBQI:
            idx_i, idx_j = triu_indices
            coords = self.pos_matrix
            real_probs = satellite_connection_probability_vec(coords[idx_i], coords[idx_j])
            success_indices = np.where(random_probs <= real_probs)[0]
            for idx in success_indices:
                u, v = idx_i[idx], idx_j[idx]
                self.edges.add((u, v))
                self.nodes[u].connections += 1
                self.nodes[v].connections += 1

        elif network_type == NETWORK_TYPE_OFBQI:
            idx_i, idx_j = triu_indices
            coords = self.pos_matrix
            
            # 1. Fibra Óptica (Paso 1)
            optic_probs = optic_fiber_connection_probability_vec(coords[idx_i], coords[idx_j])
            
            success_optic = np.where(random_probs <= optic_probs)[0]
            
            # Guardamos candidatos para el paso 2 (Edges auxiliares)
            # Usamos un set para búsqueda rápida O(1)
            edges_aux = set() 
            for idx in success_optic:
                edges_aux.add((idx_i[idx], idx_j[idx]))
            
            # 2. Fotónica (Paso 2) - Solo sobre los que pasaron el filtro 1
            if edges_aux:
                # Convertimos a lista para iterar
                aux_list = list(edges_aux)
                num_aux = len(aux_list)
                
                # Randoms para el segundo paso
                rand_probs_2 = np.random.uniform(0, 1, num_aux)
                
                aux_arr = np.array(aux_list)
                u_arr, v_arr = aux_arr[:, 0], aux_arr[:, 1]
                photonic_probs = photonic_optic_fiber_connection_probability_vec(coords[u_arr], coords[v_arr])
                success_photonic = np.where(rand_probs_2 <= photonic_probs)[0]
                for idx in success_photonic:
                    u, v = aux_list[idx]
                    self.edges.add((u, v))
                    self.nodes[u].connections += 1
                    self.nodes[v].connections += 1

    # --- NUEVO MÉTODO CRÍTICO ---
    def to_networkx(self):
        """Conversión optimizada"""
        G = nx.Graph()
        # Añadir nodos con atributos en lote (más rápido)
        G.add_nodes_from([
            (n_id, {"connections": node.connections}) 
            for n_id, node in self.nodes.items()
        ])
        G.add_edges_from(list(self.edges))
        
        # Posiciones
        pos = {i: self.pos_matrix[i] for i in range(len(self.nodes))}
        return G, pos
    
    

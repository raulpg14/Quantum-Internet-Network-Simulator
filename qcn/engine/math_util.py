import numpy as np
import random
from  scipy.stats import poisson, lognorm
import logging
from qcn.engine.config import (
    FIBER_ATTENUATION_DIVISOR,
    PHOTONIC_ATTENUATION_COEFF,
    PHOTONIC_ATTENUATION_DIVISOR,
    PHOTONIC_N_TRIALS,
    POISSON_A,
    POISSON_DENSITY,
    SAT_ETA_0,
    SAT_HEIGHT_KM,
    SAT_N_PAIRS,
    SAT_R_REC,
    SAT_W_LT,
)

logger = logging.getLogger(__name__)


def satellite_connection_probability(node_i, node_j):
    # Constantes
    node_0 = np.array([0,0])

    # Probabilidad de que un fotón llegue desde el satélite al nodo 1
    d1 = np.sqrt((SAT_HEIGHT_KM ** 2) + (np.linalg.norm(node_0 - np.array(node_i)) ** 2))
    p_1sat = SAT_ETA_0 * (1 - np.exp(-2 * ((SAT_R_REC ** 2) / ((SAT_W_LT * (d1 * 10 ** 3)) ** 2) )))

    # Probabilidad de que un fotón llegue desde el satélite al nodo 2
    d2 = np.sqrt((SAT_HEIGHT_KM ** 2) + (np.linalg.norm(node_0 - np.array(node_j)) ** 2))
    p_2sat= SAT_ETA_0 * (1 - np.exp(-2 * ((SAT_R_REC ** 2) / ((SAT_W_LT * (d2 * 10 ** 3)) ** 2) )))
    
    # Propabilidad de conexion
    p = 1 - np.power((1 - (p_1sat * p_2sat)), SAT_N_PAIRS)
    return p


def optic_fiber_connection_probability(node_i, node_j):
    d = np.linalg.norm(np.array(node_j) - np.array(node_i))
    p = np.exp(-d / FIBER_ATTENUATION_DIVISOR)
    return p

def photonic_optic_fiber_connection_probability(node_i, node_j):
    d = np.linalg.norm(np.array(node_j) - np.array(node_i))
    p1 = 10 ** ((-PHOTONIC_ATTENUATION_COEFF * d) / PHOTONIC_ATTENUATION_DIVISOR)
    p = 1 - np.power((1 - (p1)), PHOTONIC_N_TRIALS)
    return p

"""
def connectivity_distribution(G, k):
    # Cálculo de los parametros mu y sigma
    degrees = [degree for degree in G.degree()]
    logger.debug("%s", degrees)
    k1 = sum(degrees) / G.number_of_nodes()
    k2 =  sum(degree**2 for degree in degrees) / G.number_of_nodes()
    
    mu = (k1 ** 2) / np.sqrt(k2)
    sigma = np.sqrt(np.log((k2) / (k1**2)))
    p = ((1 / (k * sigma * np.sqrt(2*np.pi)))) ** ((-(np.log(k) - (mu ** 2))**2) / (2 * (sigma ** 2)))
    return p
"""

def poisson_fit(k_values: np.ndarray, density: float) -> np.ndarray:
    """
    Analytic Poisson fit for OFBQI degree distribution.
    From PRL 2020 eq.(2): P(k) = e^(-Aρ) * (Aρ)^k / k!
    where A = 5.2e4 is a network constant and ρ is the node density.
    The Poisson parameter λ = A·ρ is determined analytically, not fitted.
    """
    lambda_val = POISSON_A * density
    return poisson.pmf(k_values, lambda_val)


def log_normal_fit(k_values, n_connections, num_nodes):
    # Calcula el valor medio de conexiones por nodo
    k = np.sum(n_connections) / num_nodes
    
    # Calcula la media de los cuadrados de las conexiones por nodo
    k_2 = np.sum([x**2 for x in n_connections]) / num_nodes

    # Calcula el parámetro mu de la distribución log-normal
    mu = (k**2) / np.sqrt(k_2)
    
    # Calcula el parámetro sigma de la distribución log-normal
    sigma = np.sqrt(np.log(k_2/(k**2)))
   
    # Calcula los valores de la PDF log-normal para los k_values dados
    pdf_values = lognorm.pdf(k_values, s=sigma, scale=mu)

    # Devuelve los valores de la PDF
    return pdf_values

def satellite_connection_probability_vec(coords_i: np.ndarray, coords_j: np.ndarray) -> np.ndarray:
    """Vectorised SBQI satellite probability. coords_i, coords_j: shape (N, 2). Returns shape (N,)."""
    node_0 = np.array([0.0, 0.0])
    d1 = np.sqrt(SAT_HEIGHT_KM**2 + np.sum((coords_i - node_0)**2, axis=1))
    d2 = np.sqrt(SAT_HEIGHT_KM**2 + np.sum((coords_j - node_0)**2, axis=1))
    p1 = SAT_ETA_0 * (1 - np.exp(-2 * SAT_R_REC**2 / (SAT_W_LT * d1 * 1e3)**2))
    p2 = SAT_ETA_0 * (1 - np.exp(-2 * SAT_R_REC**2 / (SAT_W_LT * d2 * 1e3)**2))
    return 1 - np.power(1 - p1 * p2, SAT_N_PAIRS)

def optic_fiber_connection_probability_vec(coords_i: np.ndarray, coords_j: np.ndarray) -> np.ndarray:
    """Vectorised OFBQI optic fiber probability. coords_i, coords_j: shape (N, 2). Returns shape (N,)."""
    d = np.linalg.norm(coords_j - coords_i, axis=1)
    return np.exp(-d / FIBER_ATTENUATION_DIVISOR)

def photonic_optic_fiber_connection_probability_vec(coords_i: np.ndarray, coords_j: np.ndarray) -> np.ndarray:
    """Vectorised OFBQI photonic probability. coords_i, coords_j: shape (N, 2). Returns shape (N,)."""
    d = np.linalg.norm(coords_j - coords_i, axis=1)
    p1 = 10 ** ((-PHOTONIC_ATTENUATION_COEFF * d) / PHOTONIC_ATTENUATION_DIVISOR)
    return 1 - np.power(1 - p1, PHOTONIC_N_TRIALS)

def log_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Logarithmic model for SBQI: l = a * ln(N) + b"""
    return a * np.log(x) + b


def power_func(x: np.ndarray, b: float, alpha: float) -> np.ndarray:
    """Power law model for OFBQI: l = b * N^alpha (PRL 2020)"""
    return b * np.power(x, alpha)
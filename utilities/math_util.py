import numpy as np
import random
from  scipy.stats import poisson, lognorm

DIAMETER = 3600
H_SAT = 500


def satellite_connection_probability(node_i, node_j):
    # Constantes
    eta_0 = 0.1
    R_rec = 0.75
    w_LT = 0.25 * 1e-5
    n_pares = 50
    node_0 = np.array([0,0])

    # Probabilidad de que un fotón llegue desde el satélite al nodo 1
    d1 = np.sqrt((H_SAT ** 2) + (np.linalg.norm(node_0 - np.array(node_i)) ** 2))
    p_1sat = eta_0 * (1 - np.exp(-2 * ((R_rec ** 2) / ((w_LT * (d1 * 10 ** 3)) ** 2) )))

    # Probabilidad de que un fotón llegue desde el satélite al nodo 2
    d2 = np.sqrt((H_SAT ** 2) + (np.linalg.norm(node_0 - np.array(node_j)) ** 2))
    p_2sat= eta_0 * (1 - np.exp(-2 * ((R_rec ** 2) / ((w_LT * (d2 * 10 ** 3)) ** 2) )))
    
    # Propabilidad de conexion
    p = 1 - np.power((1 - (p_1sat * p_2sat)), n_pares)
    return p


def optic_fiber_connection_probability(node_i, node_j):
    d = np.linalg.norm(np.array(node_j) - np.array(node_i))
    p = np.exp(-d / 226)
    return p

def photonic_optic_fiber_connection_probability(node_i, node_j):
    d = np.linalg.norm(np.array(node_j) - np.array(node_i))
    p1 = 10 ** ((-0.2 * d) / 10)
    p = 1 - np.power((1 - (p1)), 1000)
    return p

"""
def connectivity_distribution(G, k):
    # Cálculo de los parametros mu y sigma
    degrees = [degree for degree in G.degree()]
    print(degrees)
    k1 = sum(degrees) / G.number_of_nodes()
    k2 =  sum(degree**2 for degree in degrees) / G.number_of_nodes()
    
    mu = (k1 ** 2) / np.sqrt(k2)
    sigma = np.sqrt(np.log((k2) / (k1**2)))
    p = ((1 / (k * sigma * np.sqrt(2*np.pi)))) ** ((-(np.log(k) - (mu ** 2))**2) / (2 * (sigma ** 2)))
    return p
"""
def poisson_fit(k_values):
    # Densidad
    density = 1e-4
    
    # Parámetro lambda de la distribución de Poisson
    A = 5.2 * (10**4)
    
    # Calcula los valores de la PMF de Poisson para los k_values dados
    poisson_values = poisson.pmf(k_values, A * density)

    # Devuelve los valores de la PMF de Poisson
    return poisson_values

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

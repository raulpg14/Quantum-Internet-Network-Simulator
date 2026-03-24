import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import tkinter as tk
import random
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import networkx.algorithms.community as nx_comm

from utilities.math_util import poisson_fit, log_normal_fit 
from utilities.network import Network
from utilities.csv_util import write_csv
from utilities.community_detector import community_analizer
from scipy.interpolate import make_interp_spline

import pandas as pd
import os

def redraw_network(entry, fig, ax, axR, figR, axD, figD, network_type, entry_iterations, keep_plot, shortest_path):
    array = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    if not shortest_path:
        num_nodes_str = entry.get()
        if num_nodes_str:
            num_nodes = int(num_nodes_str)
            #for i in range(len(array)):
            #for j in range(0, 20):
            N = Network()
            N.add_nodes(num_nodes)
            N.connect_nodes(network_type)
            N.draw_network(fig, ax)
            monte_carlo_model(num_nodes, N, axR, figR, network_type, entry_iterations, keep_plot)
            write_csv(N.get_neighbors(), filename='output.csv')
            """"
            # shortest path
            N_200 = Network()
            N_200.add_nodes(800)
            N_200.connect_nodes(network_type)
            write_csv(N_200.get_neighbors(), filename='output_200.csv')
            
            N_600 = Network()
            N_600.add_nodes(600)
            N_600.connect_nodes(network_type)
            write_csv(N_600.get_neighbors(), filename='output_600.csv')
            """
            community_analizer(ax=axD, fig=figD, keep_plot=keep_plot, network_type=network_type, num_nodes=num_nodes)
            #print(i, j)
        else:
            print("Please enter a valid number of nodes")
    else:
        draw_shortest_path_graphs('results.csv',figR, axR, figD, axD)

def monte_carlo_model(num_nodes, N, axR, figR, network_type, entry_iterations, keep_plot):
    # Obtiene el número de iteraciones del widget de entrada
    num_iterations_str = entry_iterations.get()
    
    # Verifica si se ingresó un número válido de iteraciones
    if num_iterations_str:
        # Convierte el número de iteraciones a un entero
        iterations = int(num_iterations_str)
        
        # Inicializa una distribución de grados total como un array de ceros
        total_degree_distribution = np.zeros(0)

        # Realiza el bucle de simulaciones de Monte Carlo
        for _ in range(iterations):
            # Crea una nueva instancia de la red
            N = Network()
            
            # Añade nodos a la red
            N.add_nodes(num_nodes)
            
            # Conecta los nodos según el tipo de red especificado
            N.connect_nodes(network_type)
            
            # Calcula la distribución de grados actual
            degree_distribution = np.array(N.calculate_degree_distribution())
            print(degree_distribution)
            
            # Asegura que total_degree_distribution tenga el tamaño adecuado
            total_degree_distribution.resize(max(len(degree_distribution), len(total_degree_distribution)))
            
            # Asegura que degree_distribution tenga el tamaño adecuado
            degree_distribution.resize(max(len(total_degree_distribution), len(degree_distribution)))
            
            # Suma la distribución de grados actual a la total
            total_degree_distribution += degree_distribution
        
        # Si no se debe mantener el gráfico anterior, limpia el gráfico
        if not keep_plot:
            axR.clear()

        # Crea un array de valores k para el eje x
        k_values = np.arange(0, len(total_degree_distribution))
        
        # Selecciona los parámetros de visualización y ajuste según el tipo de red
        if network_type == "OFBQI":
            color = 'red'
            marker = 'o'
            label = 'OFBQI (Poisson Fit)'
            # Realiza un ajuste de Poisson
            fit_values = poisson_fit(k_values)
        elif network_type == "SBQI":
            color = 'blue'
            marker = 'D'
            label = 'SBQI (Log-Normal Fit)'
            # Realiza un ajuste log-normal
            fit_values = log_normal_fit(k_values, [node.connections for node in N.get_nodes().values()], num_nodes)
        
        # Genera un array de grados
        degrees = np.arange(len(total_degree_distribution))
        
        # Dibuja el ajuste en el gráfico
        axR.plot(k_values, fit_values, label=label, color=color)
        
        # Establece el título del gráfico
        axR.set_title('Degree Distribution')
        
        # Dibuja la distribución de grados en el gráfico
        axR.scatter(degrees, total_degree_distribution / (num_nodes * iterations), alpha=0.5, color=color, marker=marker, label=f'{network_type}')
        
        # Establece las etiquetas de los ejes
        axR.set_ylabel('pk')
        axR.set_xlabel('k')
        
        # Añade la leyenda al gráfico
        axR.legend()
        
        # Actualiza el lienzo del gráfico
        figR.canvas.draw()
    else:
        # Imprime un mensaje si no se ingresó un número válido de iteraciones
        print("Please enter a valid number of iterations")

def draw_shortest_path_graphs(filename, figR, axR, figD, axD):
    # Obtener la ruta completa del archivo en la carpeta datasets
    base_dir = os.path.dirname(__file__)
    csv_filepath = os.path.join(base_dir, 'datasets', 'results.csv')
    #print(csv_filepath)
    data = pd.read_csv(csv_filepath)

    grouped_data = data.groupby(['Network type', 'Number of nodes']).mean().reset_index()
    ofbqi_data = grouped_data[grouped_data['Network type'] == 'OFBQI']
    sbqi_data = grouped_data[grouped_data['Network type'] == 'SBQI']

    number_of_nodes_array = ofbqi_data['Number of nodes'].to_numpy()
    ofbqi_shortest_path = ofbqi_data['Shortest path'].to_numpy()
    sbqi_shortest_path = sbqi_data['Shortest path'].to_numpy()

    axR.scatter(number_of_nodes_array, ofbqi_shortest_path, alpha=0.5, color='red', marker='o', label='OFBQI')
    # Crear una línea suave usando interpolación spline para OFBQI
    x_new = np.linspace(ofbqi_data['Number of nodes'].min(), ofbqi_data['Number of nodes'].max(), 300)
    spl = make_interp_spline(ofbqi_data['Number of nodes'], ofbqi_data['Shortest path'], k=3)
    y_smooth = spl(x_new)
    axR.plot(x_new, y_smooth, color='red', linestyle='-', linewidth=2)

    axR.scatter(number_of_nodes_array, sbqi_shortest_path, alpha=0.5, color='blue', marker='D', label='SBQI')
    x_new = np.linspace(sbqi_data['Number of nodes'].min(), sbqi_data['Number of nodes'].max(), 300)
    spl = make_interp_spline(sbqi_data['Number of nodes'], sbqi_data['Shortest path'], k=3)
    y_smooth = spl(x_new)
    axR.plot(x_new, y_smooth, color='blue', linestyle='-', linewidth=2)

    axR.set_ylabel('<l>')
    axR.set_xlabel('N')
    axR.legend()

    figR.canvas.draw()
    
    # Gráfico 2
    ofbqi_diameter = ofbqi_data['Diameter'].to_numpy()
    sbqi_diameter = sbqi_data['Diameter'].to_numpy()

    axD.scatter(number_of_nodes_array, ofbqi_diameter, alpha=0.5, color='red', marker='o', label='OFBQI')
    # Crear una línea suave usando interpolación spline para OFBQI
    x_new = np.linspace(ofbqi_data['Number of nodes'].min(), ofbqi_data['Number of nodes'].max(), 300)
    spl = make_interp_spline(ofbqi_data['Number of nodes'], ofbqi_data['Diameter'], k=3)
    y_smooth = spl(x_new)
    axD.plot(x_new, y_smooth, color='red', linestyle='-', linewidth=2)

    axD.scatter(number_of_nodes_array, sbqi_diameter, alpha=0.5, color='blue', marker='D', label='SBQI')
    x_new = np.linspace(sbqi_data['Number of nodes'].min(), sbqi_data['Number of nodes'].max(), 300)
    spl = make_interp_spline(sbqi_data['Number of nodes'], sbqi_data['Diameter'], k=3)
    y_smooth = spl(x_new)
    axD.plot(x_new, y_smooth, color='blue', linestyle='-', linewidth=2)

    axD.set_ylabel('<d>')
    axD.set_xlabel('N')
    axD.legend()

    figD.canvas.draw()

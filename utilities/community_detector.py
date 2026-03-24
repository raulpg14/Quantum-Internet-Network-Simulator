import networkx as nx
import networkx.algorithms.community as nx_comm
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import numpy as np

import os
import csv


def append_to_csv(filename, network_type, number_of_nodes, ng, shortest_path, diameter):
    # Comprobar si el archivo existe
    file_exists = os.path.isfile(filename)
    
    # Abrir el archivo en modo append
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Si el archivo no existe, escribir la cabecera
        if not file_exists:
            writer.writerow(["Network type", "Number of nodes", "Ng", "Shortest path", "Diameter"])
        
        # Escribir la nueva fila
        writer.writerow([network_type, number_of_nodes, ng, shortest_path, diameter])

def community_analizer(ax, fig, keep_plot, network_type, num_nodes):
    
    # Obtener la ruta completa del archivo en la carpeta datasets
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    csv_filepath = os.path.join(base_dir, 'datasets', 'output.csv')
    print(csv_filepath)
    network = pd.read_csv(csv_filepath)
    
    """
    csv_filepath_200 = os.path.join(base_dir, 'datasets', 'output_200.csv')
    network_200 = pd.read_csv(csv_filepath_200)

    csv_filepath_600 = os.path.join(base_dir, 'datasets', 'output_600.csv')
    network_600 = pd.read_csv(csv_filepath_600)
    """
    
    # Creación de la red
    G = nx.from_pandas_edgelist(df=network, source='from', target='to')
    #G_200 = nx.from_pandas_edgelist(df=network_200, source='from', target='to')
    #G_600 = nx.from_pandas_edgelist(df=network_600, source='from', target='to')

    # Detección de comunidades
    communities=nx_comm.louvain_communities(G, resolution=0)
    largest_community = max(communities, key=len)
    

    # Cálculo del camino más corto promedio
  
    #shortest_path_200 = nx.average_shortest_path_length(G_200)
    #shortest_path_600 = nx.average_shortest_path_length(G_600)
    
    #l = np.arange(shortest_path_200, shortest_path_600, shortest_path)
    #N = np.arange(200, 600, 1000)

    # Si no se debe mantener el gráfico anterior, limpia el gráfico
    """
    if not keep_plot:
        ax.clear()

    ax.plot(N, l, alpha=0.5)
    ax.set_ylabel('pk')
    ax.set_xlabel('k')
    fig.canvas.draw()
    """

    components = list(nx.connected_components(G))
    largest_component = max(communities, key=len)
    G_largest = G.subgraph(largest_component)
    shortest_path = nx.average_shortest_path_length(G_largest)
    diameter = nx.diameter(G_largest)
    # Printar por pantalla
    print(f"Número de nodos en la comunidad más grande: {len(largest_community)}")
    print(f"Shortest path: {shortest_path}")
    print(f"Diameter: {diameter}")

    # Para escribir en el archivo results
    #csv_filepath2 = os.path.join(base_dir, 'datasets', 'results.csv')
    #append_to_csv(csv_filepath2, network_type, num_nodes, len(largest_community), shortest_path, diameter)
    

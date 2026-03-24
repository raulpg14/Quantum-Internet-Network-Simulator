import random
import numpy as np

class Node():
    def __init__(self, id_node, x, y):
        self.id_node = id_node
        self.coordinates = (x,y)
        self.connections = 0

    # Getters
    def get_coordinates(self):
        return self.coordinates
    
import customtkinter as ctk
import matplotlib.pyplot as plt
import sys
import os

# Importar el panel de la interfaz
from ui.panels.dashboard_panel import DashboardPanel

class MainForm(ctk.CTk): 
    """Ventana principal (Contenedor) de la aplicación."""
    def __init__(self):
        super().__init__()
        
        self.title("Network Creator Tool")
        self.geometry("1400x850") 
        self.minsize(1000, 600)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.window_config()

    def window_config(self):
        """Instancia el DashboardPanel y lo empaqueta."""
        self.main_panel = DashboardPanel(self)
        self.main_panel.grid(row=0, column=0, sticky="nsew")

    def on_closing(self):
        """Se ejecuta cuando el usuario intenta cerrar la ventana para limpiar recursos."""
        
        print("Cerrando aplicación de forma segura...")
        
        # 1. *** NUEVO: Cancelar tareas programadas en el panel ***
        if self.main_panel:
            self.main_panel.cancel_all_after_tasks()

        # 2. Limpiar Matplotlib
        plt.close('all')
        
        # 3. Destruir la ventana (DEBE SER LA ÚLTIMA ACCIÓN)
        # Solo necesitamos self.destroy()
        self.destroy()

        os._exit(0)
import customtkinter as ctk
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk

# --- AJUSTE DE RUTA ---
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from ui.main_window import MainForm

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

if __name__ == "__main__":
    app = MainForm()

    # Evita que se muestren los errores de "check_dpi_scaling" al salir
    def silence_exit_errors(exc, val, tb):
        if "invalid command name" in str(val):
            return # Ignoramos errores de comandos inválidos al cerrar
        sys.__excepthook__(exc, val, tb) # Si es otro error, muéstralo normal

    app.report_callback_exception = silence_exit_errors

    app.mainloop()

import customtkinter as ctk
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading 
import networkx as nx 
import numpy as np 
from matplotlib.collections import LineCollection
import queue
import logging

# --- NUEVOS IMPORTS PARA LOS AJUSTES DE CURVA ---
from scipy.optimize import curve_fit
from scipy.stats import poisson, lognorm

from qcn.engine.simulation import run_simulation 
from qcn.engine.config import (
    DEFAULT_DENSITY_COEFF_EVOLUTION,
    DEFAULT_FALLBACK_RADIUS,
    DEFAULT_MC_REPS,
    DEFAULT_NODE_INCREMENT,
    DEFAULT_NODES,
    DEFAULT_RADIUS_DISTRIBUTION,
    DEFAULT_STEPS,
    NETWORK_TYPE_OFBQI,
    NETWORK_TYPE_SBQI,
    NETWORK_TYPES,
    SIM_MODE_DISTRIBUTION,
    SIM_MODE_EVOLUTION,
    STYLE_MAP,
)

logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN DE COLORES ---
COLOR_HEADER = "#2A3138"       
COLOR_BG_MAIN = "white"        
COLOR_ACCENT = "#3B8ED0"       
COLOR_DANGER = "#C0392B"
TEXT_TITLE = "#333333"         
TEXT_HEADER = "#A0A0A0"        
COLOR_DROPDOWN_BG = "#343B42"  
COLOR_DROPDOWN_HOVER = "#3B8ED0" 

MODE_DISTRIBUTION = SIM_MODE_DISTRIBUTION
MODE_EVOLUTION_N = SIM_MODE_EVOLUTION

# --- FUNCIONES MATEMÁTICAS PARA AJUSTE ---
def fit_poisson(k, lamb):
    """Función de probabilidad de Poisson para ajuste (k debe ser entero)."""
    return poisson.pmf(k, lamb)

def fit_lognormal(x, s, scale):
    """Función Log-Normal. s=sigma (shape), scale=exp(mu). loc fijado en 0."""
    return lognorm.pdf(x, s, 0, scale)

class DashboardPanel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color=COLOR_BG_MAIN)
        
        self.is_simulating = False
        self.simulation_thread = None
        self.simulation_result = None 
        self.after_id = None
        self.stop_event = threading.Event() 
        self.current_sim_params = {} 
        
        self.grid_rowconfigure(0, weight=0) 
        self.grid_rowconfigure(1, weight=1) 
        self.grid_columnconfigure(0, weight=1)

        self.create_top_menu()
        self.create_dashboard_content()
        
        # Inicializar UI
        self.update_inputs_visibility(MODE_DISTRIBUTION)
        self.init_matplotlib(mode=MODE_DISTRIBUTION) 
        
        self._check_simulation_status() 
        self.progress_queue = None

    # ============================================================
    #   LÓGICA DE VISIBILIDAD DE INPUTS
    # ============================================================
    def on_mode_change(self, choice):
        # 1. Actualizar visibilidad visual
        self.update_inputs_visibility(choice)
        
        # 2. Reiniciar gráficos
        self.init_matplotlib(mode=choice)
        
        # 3. CAMBIO INTELIGENTE DE VALORES POR DEFECTO
        # Limpiamos el input actual
        self.entry_radius.delete(0, "end")
        
        if choice == MODE_EVOLUTION_N:
            # Si pasamos a modo Evolución, ponemos el coeficiente de densidad (2)
            self.entry_radius.insert(0, str(DEFAULT_DENSITY_COEFF_EVOLUTION).rstrip("0").rstrip("."))
        else:
            # Si pasamos a modo Distribución, ponemos el radio estándar (1261)
            self.entry_radius.insert(0, str(DEFAULT_RADIUS_DISTRIBUTION).rstrip("0").rstrip("."))

    def update_inputs_visibility(self, mode):
        if mode == MODE_DISTRIBUTION:
            self.lbl_nodes.configure(text="Nodes")     
            self.lbl_radius.configure(text="Radius") 
            self.chk_keep_plot.pack(side="left", padx=5)
            self.frame_steps.pack_forget()
            self.frame_incr.pack_forget()
            
        elif mode == MODE_EVOLUTION_N:
            self.lbl_nodes.configure(text="Init N")    
            self.lbl_radius.configure(text="Dens (x10⁻⁴)") 
            self.chk_keep_plot.pack(side="left", padx=5)
            self.frame_steps.pack(side="left", padx=2)
            self.frame_incr.pack(side="left", padx=2) 
            self.lbl_incr.configure(text="N Incr")

    # ============================================================
    #   LÓGICA DE CONTROL
    # ============================================================
    
    def on_generate_click(self):
        if self.is_simulating: return
        self.is_simulating = True
        self.stop_event.clear()
        
        self.btn_generate.configure(state="disabled", text="RUNNING...")
        self.btn_stop.configure(state="normal")
        self.lbl_progress.configure(text="Starting...")
        
        self.progress_queue = queue.Queue()
        selected_mode = self.combo_mode.get()

        data = {
            'nodes': self.entry_nodos.get(),         
            'radius': self.entry_radius.get(),
            'type': self.network_type.get(),
            'mc_iter': self.entry_mc_iter.get(),       
            'nets_per_mc': self.entry_steps.get(),   
            'rad_incr': self.entry_incr.get(),       
            'keep_plot': self.keep_plot.get(),
            'queue': self.progress_queue,
            'stop_event': self.stop_event,
            'sim_mode': selected_mode 
        }
        self.current_sim_params = data
        
        self.simulation_thread = threading.Thread(target=self._run_logic_in_thread, args=(data,), daemon=True)
        self.simulation_thread.start()

    def on_stop_click(self):
        if self.is_simulating:
            self.lbl_progress.configure(text="Stopping...")
            self.stop_event.set()
            self.btn_stop.configure(state="disabled")

    def _run_logic_in_thread(self, data):
        try: self.simulation_result = run_simulation(data)
        except Exception as e:
            logger.exception("Simulation thread failed")
            self.simulation_result = {"error": str(e)}

    def _check_simulation_status(self):
        if self.is_simulating and self.progress_queue:
            msg = None
            try:
                while True: msg = self.progress_queue.get_nowait()
            except queue.Empty: pass
            if msg: self.lbl_progress.configure(text=msg)

        if self.is_simulating and not self.simulation_thread.is_alive():
            self.is_simulating = False
            self.btn_generate.configure(state="normal", text="RUN")
            self.btn_stop.configure(state="disabled")
            
            res = self.simulation_result
            if res and "cancelled" in res:
                self.lbl_progress.configure(text="Cancelled.")
            elif res and "success" in res:
                self.lbl_progress.configure(text="Done.")
                
                final_r = res.get('final_radius', DEFAULT_FALLBACK_RADIUS)
                self.draw_network_on_canvas(res['G'], res['pos'], final_r)
                
                mode = self.current_sim_params.get('sim_mode')
                keep = self.current_sim_params.get('keep_plot', False)
                
                if mode == MODE_DISTRIBUTION:
                    self.draw_distribution_chart(res['dist_x'], res['dist_y'], res['type'], keep)
                else:
                    self.draw_evolution_charts(res['x_nodes'], res['y_path'], res['y_diameter'], res['type'], keep)

            elif res and "error" in res:
                self.lbl_progress.configure(text="Error.")
                logger.error("Error: %s", res['error'])
        
        self.after_id = self.after(100, self._check_simulation_status)

    def cancel_all_after_tasks(self):
        if self.after_id:
            try: self.after_cancel(self.after_id)
            except Exception:
                logger.debug("Suppressed", exc_info=True)

    # ============================================================
    #   FUNCIONES DE DIBUJO
    # ============================================================

    def draw_network_on_canvas(self, G, pos, radius):
        try:
            self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            self.ax.set_box_aspect(1) 
            
            shifted_pos = {n: (c[0]+radius, c[1]+radius) for n, c in pos.items()}
            
            edge_coords = [[shifted_pos[u], shifted_pos[v]] for u, v in G.edges()]
            if edge_coords:
                lc = LineCollection(edge_coords, colors="#888888", linewidths=0.6, alpha=0.4)
                self.ax.add_collection(lc)
            
            x_nodes = [shifted_pos[n][0] for n in G.nodes()]
            y_nodes = [shifted_pos[n][1] for n in G.nodes()]
            degrees = [deg for node, deg in G.degree()]
            
            if x_nodes:
                cmap = plt.cm.get_cmap('RdYlBu_r') 
                sc = self.ax.scatter(x_nodes, y_nodes, s=35, c=degrees, cmap=cmap, 
                                     edgecolors='black', linewidths=0.8, alpha=1.0, zorder=10)
                
                cbar = self.fig.colorbar(sc, ax=self.ax, shrink=0.7, aspect=20, pad=0.04)
                cbar.set_label('k (Degree)', fontsize=8)
                cbar.ax.tick_params(labelsize=7)

            limit = radius * 2
            self.ax.set_xlim(0, limit); self.ax.set_ylim(0, limit)
            self.ax.grid(True, linestyle=':', alpha=0.6)
            self.ax.set_xlabel(f"Space (Size: {radius*2:.1f} u)", fontsize=8, color="#555")
            self.ax.set_title(f"Visual Representation (R={radius:.1f})", fontsize=10)
            
            self.fig.tight_layout() 
            self.canvas.draw()
        except Exception as e: logger.error("Net Error: %s", e)

    def draw_distribution_chart(self, x_values, y_values, net_type, keep_plot):
        try:
            if not keep_plot: self.ax_dist.clear()
            self.ax_dist.set_box_aspect(1)
            
            style = STYLE_MAP.get(net_type, {"color": "gray", "marker": "x", "label": net_type})
            
            # Dibujar puntos de simulación
            self.ax_dist.plot(x_values, y_values, color=style["color"], marker=style["marker"], 
                             linestyle='', markersize=4, label=f"{style['label']} (Sim)", alpha=0.6)
            
            if len(x_values) > 2:
                try:
                    # --- CORRECCIÓN LÓGICA AQUÍ ---
                    if net_type == NETWORK_TYPE_OFBQI:
                        # POISSON: Es discreta, solo funciona con ENTEROS.
                        # Usamos np.arange en lugar de linspace para que no devuelva 0 en decimales.
                        mean_k = np.average(x_values, weights=y_values)
                        popt, _ = curve_fit(fit_poisson, x_values, y_values, p0=[mean_k])
                        
                        x_fit_int = np.arange(min(x_values), max(x_values) + 1)
                        
                        # Usamos rf"..." para evitar el warning de la lambda
                        self.ax_dist.plot(x_fit_int, fit_poisson(x_fit_int, *popt), 
                                          color=style["color"], linestyle='--', alpha=0.8, 
                                          label=rf"Poisson Fit ($\lambda$={popt[0]:.1f})")

                    elif net_type == NETWORK_TYPE_SBQI:
                        # LOG-NORMAL: Es continua, necesita linspace (decimales) para verse suave.
                        x_fit_smooth = np.linspace(min(x_values), max(x_values), 200)
                        
                        p0 = [0.5, np.mean(x_values)] 
                        popt, _ = curve_fit(fit_lognormal, x_values, y_values, p0=p0, maxfev=5000)
                        
                        self.ax_dist.plot(x_fit_smooth, fit_lognormal(x_fit_smooth, *popt), 
                                          color=style["color"], linestyle='-', linewidth=1.5, alpha=0.8, 
                                          label="LogNormal Fit")
                                          
                except Exception as fit_err:
                    logger.warning("Fit Warning: %s", fit_err)

            self.ax_dist.set_title("Degree Distribution", fontsize=10, weight='bold', color="#333")
            self.ax_dist.set_xlabel("Degree ($k$)", fontsize=9)
            self.ax_dist.set_ylabel("$P(k)$", fontsize=9)
            self.ax_dist.grid(True, linestyle=':', alpha=0.4)
            self.ax_dist.legend(loc='upper right', fontsize=7, frameon=True)
            self.canvas_right.draw()
        except Exception as e: logger.error("Dist Error: %s", e)

    def draw_evolution_charts(self, x_nodes, y_path, y_diam, net_type, keep_plot):
        try:
            if not keep_plot:
                self.ax_l.clear()
                self.ax_d.clear()
                
            style = STYLE_MAP.get(net_type, {"color": "gray", "marker": "o", "label": net_type})
            label_text = f"{net_type}"
            
            # Gráfico L (Path)
            self.ax_l.plot(x_nodes, y_path, color=style["color"], marker='s', linestyle='--', markersize=4, label=label_text)
            self.ax_l.set_title(r"Avg Shortest Path $\langle l \rangle$", fontsize=9, weight='bold')
            self.ax_l.set_xlabel("$N$ (Nodes)", fontsize=8)
            self.ax_l.set_ylabel(r"$\langle l \rangle$", fontsize=8)
            self.ax_l.grid(True, linestyle=':', alpha=0.5)
            self.ax_l.legend(fontsize=7, loc='upper left')

            # Gráfico D (Diameter)
            self.ax_d.plot(x_nodes, y_diam, color=style["color"], marker='o', linestyle='--', markersize=4, label=label_text)
            self.ax_d.set_title(r"Network Diameter $\langle d \rangle$", fontsize=9, weight='bold')
            self.ax_d.set_xlabel("$N$ (Nodes)", fontsize=8)
            self.ax_d.set_ylabel(r"$\langle d \rangle$", fontsize=8)
            self.ax_d.grid(True, linestyle=':', alpha=0.5)
            
            self.fig_right.tight_layout()
            self.canvas_right.draw()
        except Exception as e: logger.error("Evo Error: %s", e)

    # ============================================================
    #   LAYOUT & INIT
    # ============================================================

    def create_top_menu(self):
        self.header = ctk.CTkFrame(self, fg_color=COLOR_HEADER, corner_radius=0, height=70)
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.grid_propagate(False)

        ctk.CTkLabel(self.header, text="NetworkTool", font=("Segoe UI", 20, "bold"), text_color="white").pack(side="left", padx=(20, 15))

        self.combo_mode = ctk.CTkComboBox(self.header, values=[MODE_DISTRIBUTION, MODE_EVOLUTION_N], 
                                          width=160, height=24, font=("Segoe UI", 12),
                                          fg_color=COLOR_DROPDOWN_BG, text_color="white",
                                          dropdown_fg_color=COLOR_DROPDOWN_BG,
                                          dropdown_text_color="white",
                                          dropdown_hover_color=COLOR_DROPDOWN_HOVER,
                                          button_color=COLOR_ACCENT, border_width=0,
                                          command=self.on_mode_change)
        self.combo_mode.pack(side="left", padx=10)

        ctk.CTkFrame(self.header, width=2, height=35, fg_color="#404B55").pack(side="left", padx=10)

        grp_common = ctk.CTkFrame(self.header, fg_color="transparent")
        grp_common.pack(side="left")
        
        self.lbl_nodes, self.entry_nodos = self._add_top_input(grp_common, "Nodes", str(DEFAULT_NODES), width=50)
        self._add_top_combo(grp_common, "Type", NETWORK_TYPES)
        
        self.lbl_radius, self.entry_radius = self._add_top_input(grp_common, "Radius", str(DEFAULT_RADIUS_DISTRIBUTION).rstrip("0").rstrip("."), width=50) 
        
        _, self.entry_mc_iter = self._add_top_input(grp_common, "MC Reps", str(DEFAULT_MC_REPS), width=40)

        self.frame_steps = ctk.CTkFrame(self.header, fg_color="transparent")
        _, self.entry_steps = self._add_top_input(self.frame_steps, "Steps", str(DEFAULT_STEPS), width=40)
        
        self.frame_incr = ctk.CTkFrame(self.header, fg_color="transparent")
        self.lbl_incr, self.entry_incr = self._add_top_input(self.frame_incr, "N Incr", str(DEFAULT_NODE_INCREMENT), width=40)
        
        self.keep_plot = ctk.BooleanVar(value=False)
        self.chk_keep_plot = ctk.CTkCheckBox(self.header, text="Keep Plot", variable=self.keep_plot, 
                                             text_color="#D0D0D0", font=("Segoe UI", 11), width=80)

        btn_frame = ctk.CTkFrame(self.header, fg_color="transparent")
        btn_frame.pack(side="right", padx=20)
        
        self.btn_stop = ctk.CTkButton(btn_frame, text="STOP", width=70, height=34, fg_color=COLOR_DANGER, state="disabled", command=self.on_stop_click)
        self.btn_stop.pack(side="right", padx=(10, 0))

        self.btn_generate = ctk.CTkButton(btn_frame, text="RUN", width=100, height=34, fg_color=COLOR_ACCENT, font=("Segoe UI", 12, "bold"), command=self.on_generate_click)
        self.btn_generate.pack(side="right")
        self.lbl_progress = ctk.CTkLabel(self.header, text="", text_color="#A0A0A0", font=("Segoe UI", 11))
        self.lbl_progress.pack(side="right", padx=5)

    def create_dashboard_content(self):
        container = ctk.CTkFrame(self, fg_color="transparent")
        container.grid(row=1, column=0, sticky="nsew", padx=20, pady=25)
        
        container.grid_columnconfigure(0, weight=1) 
        container.grid_columnconfigure(1, weight=5) 
        container.grid_columnconfigure(2, weight=0) 
        container.grid_columnconfigure(3, weight=5) 
        container.grid_columnconfigure(4, weight=1) 
        container.grid_rowconfigure(0, weight=1)

        # Panel Izquierdo
        frame_left = ctk.CTkFrame(container, fg_color="transparent")
        frame_left.grid(row=0, column=1, sticky="nsew")
        ctk.CTkLabel(frame_left, text="Visualization", font=("Segoe UI", 14, "bold"), text_color="#333").pack(pady=(0,5))
        self.net_canvas_area = ctk.CTkFrame(frame_left, fg_color="white", corner_radius=0)
        self.net_canvas_area.pack(fill="both", expand=True)

        ctk.CTkFrame(container, width=2, fg_color="#E5E5E5").grid(row=0, column=2, sticky="ns", padx=15)

        # Panel Derecho
        frame_right = ctk.CTkFrame(container, fg_color="transparent")
        frame_right.grid(row=0, column=3, sticky="nsew")
        
        ctk.CTkLabel(frame_right, text="Analysis", font=("Segoe UI", 14, "bold"), text_color="#333").pack(pady=(0,5))
        self.dist_canvas_area = ctk.CTkFrame(frame_right, fg_color="white", corner_radius=0)
        self.dist_canvas_area.pack(fill="both", expand=True)

    def init_matplotlib(self, mode=MODE_DISTRIBUTION):
        if not hasattr(self, 'fig'):
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.fig.subplots_adjust(left=0.15, bottom=0.15, right=0.90, top=0.90)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.net_canvas_area)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        for widget in self.dist_canvas_area.winfo_children():
            widget.destroy()

        if mode == MODE_DISTRIBUTION:
            self.fig_right, self.ax_dist = plt.subplots(figsize=(5, 5))
            self.fig_right.subplots_adjust(left=0.15, bottom=0.15, right=0.90, top=0.90)
            self.canvas_right = FigureCanvasTkAgg(self.fig_right, master=self.dist_canvas_area)
            self.canvas_right.draw()
            self.canvas_right.get_tk_widget().pack(side="top", fill="both", expand=True)
            
        elif mode == MODE_EVOLUTION_N:
            self.fig_right, (self.ax_l, self.ax_d) = plt.subplots(2, 1, figsize=(5, 5))
            self.fig_right.subplots_adjust(left=0.15, bottom=0.10, right=0.95, top=0.95, hspace=0.4)
            self.canvas_right = FigureCanvasTkAgg(self.fig_right, master=self.dist_canvas_area)
            self.canvas_right.draw()
            self.canvas_right.get_tk_widget().pack(side="top", fill="both", expand=True)

    def _add_top_input(self, parent, label_text, default, width=60):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(side="left", padx=2)
        lbl = ctk.CTkLabel(f, text=label_text, text_color=TEXT_HEADER, font=("Segoe UI", 10))
        lbl.pack(anchor="w")
        e = ctk.CTkEntry(f, width=width, height=24, fg_color="#343B42", text_color="white", border_width=0)
        e.insert(0, default)
        e.pack()
        return lbl, e

    def _add_top_combo(self, parent, label, values):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(side="left", padx=2)
        ctk.CTkLabel(f, text=label, text_color=TEXT_HEADER, font=("Segoe UI", 10)).pack(anchor="w")
        self.network_type = ctk.StringVar(value=values[0])
        ctk.CTkComboBox(f, values=values, variable=self.network_type, 
                        width=75, height=24, font=("Segoe UI", 12),
                        fg_color=COLOR_DROPDOWN_BG, text_color="white",
                        dropdown_fg_color=COLOR_DROPDOWN_BG,
                        dropdown_text_color="white",
                        dropdown_hover_color=COLOR_DROPDOWN_HOVER,
                        button_color=COLOR_ACCENT, border_width=0).pack()

import customtkinter as ctk
import sys
import logging
from qcn.interfaces.gui.main_window import MainForm

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = MainForm()

    # Evita que se muestren los errores de "check_dpi_scaling" al salir
    def silence_exit_errors(exc, val, tb):
        if "invalid command name" in str(val):
            return # Ignoramos errores de comandos inválidos al cerrar
        sys.__excepthook__(exc, val, tb) # Si es otro error, muéstralo normal

    app.report_callback_exception = silence_exit_errors

    app.mainloop()


if __name__ == "__main__":
    main()

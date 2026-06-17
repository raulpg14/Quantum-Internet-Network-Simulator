import csv
from pathlib import Path
import logging
from qcn.engine.config import DATA_DIR

logger = logging.getLogger(__name__)


def write_csv(data, filename):

    # Obtener la ruta completa del archivo en la carpeta data
    csv_filepath = DATA_DIR / filename

    # Abrir el archivo CSV en modo escritura para sobrescribir el contenido existente
    with open(csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Escribir la cabecera del CSV
        writer.writerow(["from", "to"])

        # Recorrer el diccionario y escribir las relaciones
        for key, values in data.items():
            for value in values:
                writer.writerow([key, value])

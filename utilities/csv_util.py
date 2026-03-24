import csv
import os


def write_csv(data, filename):

    # Obtener la ruta completa del archivo en la carpeta datasets
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_filepath = os.path.join(base_dir, 'datasets', filename)

    # Abrir el archivo CSV en modo escritura para sobrescribir el contenido existente
    with open(csv_filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Escribir la cabecera del CSV
        writer.writerow(["from", "to"])

        # Recorrer el diccionario y escribir las relaciones
        for key, values in data.items():
            for value in values:
                writer.writerow([key, value])

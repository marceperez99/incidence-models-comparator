import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import csv
import datetime
import os
from typing import List, Optional


def get_dataset(path, filter_levels: Optional[List[str]] = None,
                filter_diseases: Optional[List[str]] = None,
                filter_case_types: Optional[List[str]] = None):
    """
        Carga el CSV de casos y realiza preprocesamiento:
         - Filtrado de enfermedades deseadas
         - Conversión de fecha y extracción de t (días desde inicio +1)
         - Imputación de ceros en i_cases con ruido controlado
         - Cálculo de suma acumulada por unidad temporal
         - Generación de id_proy
        """
    # ---- 1. Leer CSV y parsear fechas ----
    df = pd.read_csv(path, parse_dates=['date'], dtype={'i_cases': 'float64'})

    # 2. Filtros condicionales
    if filter_levels:
        df = df[df['name'].isin(filter_levels)]
    if filter_diseases:
        # normalizamos a mayúsculas para evitar discrepancias
        allowed = {d.upper() for d in filter_diseases}
        df = df[df['disease'].str.upper().isin(allowed)]
    if filter_case_types:
        df = df[df['classification'].isin(filter_case_types)]

    # ---- 3. Ordenar para cumsum ----
    df = df.sort_values(['name', 'level', 'disease', 'classification', 'date'])

    # ---- 4. Imputar ceros de i_cases con ruido exponencial ----
    seed = 32
    rng = np.random.default_rng(seed)
    mask_zero = df['i_cases'] == 0
    # ruido: distribución exponencial con media ~0.1
    df.loc[mask_zero, 'i_cases'] = rng.exponential(scale=0.1, size=mask_zero.sum())

    # ---- 5. Calcular suma acumulada (csum) por grupo ----
    df['csum'] = df.groupby(['name', 'level', 'disease', 'classification'])['i_cases'].cumsum()

    # ---- 6. Generar id_proy como string compacto ----
    df['id_proy'] = (df['name'] + '_' + df['disease'] + '_' + df['classification'])

    # ---- 7. Crear columna t (días desde el inicio + 1) ----
    df['t'] = (df['date'] - df['date'].min()).dt.days + 1

    return df


def loss_function(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / np.mean(y_true)
    return (mae + mape + rmse + nrmse) / 4


def log_experiment(parameters: dict, loss: float, algorithm: str, filename="experiment_log.csv", dataset: str = ''):
    # Define the header
    headers = ["timestamp", "dataset", "algorithm", "loss", "parameters"]

    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert parameters dict to string for storage
    parameters_str = str(parameters)

    # Check if file exists to write headers
    file_exists = os.path.isfile(filename)

    # Append data to the CSV file
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write headers only if the file is new
        if not file_exists:
            writer.writerow(headers)

        # Write the new experiment log
        writer.writerow([timestamp, dataset, algorithm, loss, parameters_str])

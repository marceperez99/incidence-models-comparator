import os
import csv
from typing import Dict, Union


def log_model_metrics(
        method: str,
        disease: str,
        classification: str,
        level: str,
        prediction_weeks: int,
        mae: float = '',
        mape: float = '',
        rmse: float = '',
        nrmse: float = '',
        loss: float = '',
        hyperparams: Dict[str, Union[str, float, int]] = dict(),
        output_file: str = "outputs/metrics.csv"
) -> None:
    """
    Guarda las métricas y los hiperparámetros de un modelo en un archivo CSV.

    Si el archivo ya existe, agrega una nueva fila. Si no existe, lo crea con encabezados.

    Args:
        method (str): Nombre del modelo/método
        disease (str): Tipo de enfermedad (ej: Dengue)
        classification (str): Confirmado / Probable
        level (str): Nivel territorial (ej: Departamental)
        mae, mape, rmse, nrmse, loss (float): Métricas del modelo
        hyperparams (Dict): Hiperparámetros usados en el entrenamiento
        output_file (str): Ruta del archivo CSV de salida
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    row = {
        "Method": method.lower(),
        "Disease": disease.lower(),
        "Classification": classification.lower(),
        "Level": level.lower(),
        "Prediction Weeks": prediction_weeks,
        "MAE": round(mae, 8),
        "MAPE": round(mape, 8),
        "RMSE": round(rmse, 8),
        "NRMSE": round(nrmse, 8),
        "Loss": round(loss, 8),
        "Hyperparameters": str(hyperparams)
    }

    file_exists = os.path.isfile(output_file)
    with open(output_file, mode="a", newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"✅ Métricas guardadas en: {output_file}")

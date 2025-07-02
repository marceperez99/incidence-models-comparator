import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from evaluation.persist import save_as_csv
from .model import lstm_model
from utils import get_plot_directory, get_results_directory
from evaluation import graphing, metrics
import concurrent.futures
import os


def run_level(dataset, week):
    training_window = 4
    loss, x, y_true, y_pred, arquitectura = lstm_model(dataset, training_window, week, return_predictions=True)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / np.mean(y_true)

    y_pred = y_pred
    y_true = y_true

    # saving results
    disease = dataset['disease'].iloc[0].lower()
    level_name = dataset['name'].iloc[0].lower()
    classification = dataset['classification'].iloc[0].lower()
    plot_directory = get_plot_directory(disease, level_name, classification, 'lstm')
    results_directory = get_results_directory(disease, level_name, classification, 'lstm')
    filename = f"{dataset['name'].iloc[0]}_{dataset['classification'].iloc[0]}_{week}".lower()
    print(f"   💾 Guardando resultados en {results_directory}")
    save_as_csv(pd.DataFrame({'Observed': y_true, 'Predicted': y_pred}), f'{filename}.csv',
                output_dir=results_directory)

    title = f"Modelo LSTM ({dataset['disease'].iloc[0]})"
    descripcion = f'VP:{week} semanas VE: {training_window} semanas'
    print(f"   📊 Generando gráficos de predicción y dispersión para {filename}")
    graphing.plot_observed_vs_predicted(y_true, y_pred, f'plt_obs_pred_{filename}', output_dir=plot_directory,
                                        title=title, description=descripcion)
    graphing.plot_scatter(y_true, y_pred, f'plt_scatter_{filename}', 'lstm', title=title,
                          description=descripcion, output_dir=plot_directory)
    print(f"   📝 Log de métricas del modelo para {filename}")
    metrics.log_model_metrics('LSTM', disease, dataset['classification'].iloc[0],
                              dataset['name'].iloc[0], week, mae=mae, mape=mape, nrmse=nrmse, loss=loss, rmse=rmse,
                              hyperparams={'training_window': training_window, 'prediction_window': week,
                                           'network': arquitectura})

    return x, y_true, y_pred


def run_level_wrapper(args):
    dataset, week = args
    x, y_true, y_pred = run_level(dataset, week)
    return [week, x, y_pred]


def run_lstm_multiprocess(datasets, weeks):
    """
    Ejecuta run_exponential sobre cada dataset en paralelo usando procesos.
    - datasets: lista de DataFrames
    - weeks: entero
    """
    print(f"\n⚡ Ejecutando en paralelo con {os.cpu_count()} procesos disponibles ({len(datasets)} datasets)...")

    for i, dataset in enumerate(datasets):
        print(
            f"\n🚩 Procesando dataset {i + 1}/{len(datasets)}: {dataset['name'].iloc[0]} ({dataset['disease'].iloc[0]}, {dataset['classification'].iloc[0]})")
        args_list = [(dataset, i) for i in range(1, weeks + 1)]

        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            series = list(executor.map(run_level_wrapper, args_list))
            print(f"   🖼️ Graficando predicciones combinadas para {dataset['name'].iloc[0]}")
            graphing.graficar_predicciones(dataset, series, method="lstm")
    print("\n✅ Finalizó la ejecución de run_lstm.")

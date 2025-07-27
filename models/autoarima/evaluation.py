import pandas as pd

from evaluation.persist import save_as_csv
from models.autoarima.model import autoarima_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from evaluation import graphing, metrics
import concurrent.futures
import os
import warnings
from constants import LOSS_METRIC
from utils import get_plot_directory, get_results_directory

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
CONCURRENT = True

def run_level(dataset, week_i):
    # saving results
    disease = dataset['disease'].iloc[0].lower()
    level_name = dataset['name'].iloc[0].lower()
    classification = dataset['classification'].iloc[0].lower()
    plot_directory = get_plot_directory(disease, level_name, classification, 'autoarima')  # <--- Cambiado aquí
    results_directory = get_results_directory(disease, level_name, classification, 'autoarima')  # <--- Cambiado aquí
    filename = f"{dataset['name'].iloc[0]}_{dataset['classification'].iloc[0]}_{week_i}".lower()


    loss, x, y_true, y_pred = autoarima_model(dataset, week_i, return_predictions=True)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / np.mean(y_true)



    save_as_csv(pd.DataFrame({'Observed': y_true, 'Predicted': y_pred}), f'{filename}.csv',
                output_dir=results_directory)

    title = f"Modelo AutoARIMA ({dataset['disease'].iloc[0]})"
    descripcion = f'VP:{week_i} semanas'
    print(f"   📊 Generando gráficos de predicción y dispersión para {filename}")
    t_to_date = dict(zip(dataset['t'], dataset['date']))
    x_dates = [t_to_date.get(ti, pd.NaT) for ti in x]
    graphing.plot_observed_vs_predicted(x_dates, y_true, y_pred, f'plt_obs_pred_{filename}', output_dir=plot_directory,
                                        title=title, description=descripcion)
    graphing.plot_scatter(y_true, y_pred, f'plt_scatter_{filename}', 'mlp', title=title,  # <--- Cambiado aquí
                          description=descripcion, output_dir=plot_directory)
    print(f"   📝 Log de métricas del modelo para {filename}")
    metrics.log_model_metrics('AutoARIMA', disease, dataset['classification'].iloc[0],
                              dataset['name'].iloc[0], week_i, mae=mae, mape=mape, nrmse=nrmse, loss=loss, rmse=rmse,
                              hyperparams={})

    return x, y_true, y_pred


def run_level_wrapper(args):
    dataset, week = args

    x, y_true, y_pred = run_level(dataset, week)
    return [week, x, y_pred]


def run_autoarima(datasets, weeks):
    print(f"\n⚡ Ejecutando en paralelo con {os.cpu_count()} procesos disponibles ({len(datasets)} datasets)...")

    for i, dataset in enumerate(datasets):
        try:
            disease = dataset['disease'].iloc[0].lower()
            level = dataset['name'].iloc[0].lower()
            classification = dataset['classification'].iloc[0].lower()
            directory = f'outputs/plots/{LOSS_METRIC}/autoarima/{disease}/{level}/{classification}/nro_de_casos.png'
            if os.path.exists(directory):
                print(f'{dataset["id_proy"].iloc[0]} already calculated')
                continue
            print(
                f"\n🚩 Procesando dataset {i + 1}/{len(datasets)}: {dataset['name'].iloc[0]} ({dataset['disease'].iloc[0]}, {dataset['classification'].iloc[0]})")

            if CONCURRENT:
                args_list = [(dataset, i) for i in range(1, weeks + 1)]
                with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                    series = list(executor.map(run_level_wrapper, args_list))
                    print(f"   🖼️ Graficando predicciones combinadas para {dataset['name'].iloc[0]}")
                    graphing.graficar_predicciones(dataset, series, method="autoarima")
            else:
                series = []
                for i in range(1, weeks + 1):
                    series.append(run_level(dataset, i))
                graphing.graficar_predicciones(dataset, series, method="autoarima")
        except Exception as e:
            print(f'Error en dataset[{i}]:  ', e)

    print("\n✅ Finalizó la ejecución de autoarima...")

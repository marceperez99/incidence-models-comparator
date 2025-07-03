import pandas as pd

from evaluation.persist import save_as_csv
from models.autoarima.model import autoarima_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from evaluation import graphing, metrics
import concurrent.futures
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


LOSS = 'mape'

def run_level(dataset, week_i):
    loss, x, y_true, y_pred = autoarima_model(dataset, week_i, return_predictions=True)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / np.mean(y_true)

    # saving results
    disease = dataset['disease'].iloc[0].lower()
    level_name = dataset['name'].iloc[0].lower()
    classification = dataset['classification'].iloc[0].lower()
    filename = f"{dataset['name'].iloc[0]}_{dataset['classification'].iloc[0]}_{week_i}".lower()
    print(f"   💾 Guardando resultados en outputs/{LOSS}/predictions/autoarima/{disease}/{filename}.csv")
    save_as_csv(pd.DataFrame({'Observed': y_true, 'Predicted': y_pred}),
                f'{filename}.csv',
                output_dir=f'outputs/predictions/{LOSS}/autoarima/{disease}/{level_name}/{classification}')

    title = f"Modelo AutoARIMA ({dataset['disease'].iloc[0]})"
    descripcion = f'VP:{week_i} semanas'
    print(f"   📊 Generando gráficos de predicción y dispersión para {filename}")
    graphing.plot_observed_vs_predicted(y_true, y_pred, f'plt_obs_pred_{filename}',
                                        output_dir=f'outputs/plots/{LOSS}/autoarima/{disease}/{level_name}/{classification}',
                                        title=title, description=descripcion)
    graphing.plot_scatter(y_true, y_pred, f'plt_scatter_{filename}', 'AutoARIMA',
                          title=title, description=descripcion,
                          output_dir=f'outputs/plots/{LOSS}/autoarima/{disease}/{level_name}/{classification}')
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
        print(
            f"\n🚩 Procesando dataset {i + 1}/{len(datasets)}: {dataset['name'].iloc[0]} ({dataset['disease'].iloc[0]}, {dataset['classification'].iloc[0]})")
        args_list = [(dataset, i) for i in range(1, weeks + 1)]

        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            series = list(executor.map(run_level_wrapper, args_list))
            print(f"   🖼️ Graficando predicciones combinadas para {dataset['name'].iloc[0]}")
            graphing.graficar_predicciones(dataset, series, method="autoarima")
    print("\n✅ Finalizó la ejecución de autoarima...")
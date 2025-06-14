import pandas as pd
import functools
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from evaluation.persist import save_as_csv
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.contants import GENERATIONS, POPULATION, MUTATION_PROBABILITY
from utils import get_plot_directory, get_results_directory
from .model import subexponential_model
from .utils import get_initial_population
from evaluation import graphing, metrics
import concurrent.futures
import os


def subexponential_evaluator(dataset, weeks):
    @functools.cache
    def subexponential_evaluation(individual):
        if individual[0] <= 1: return float('inf')
        training_window = individual[0]
        loss = subexponential_model(dataset, training_window, weeks)
        return loss

    return subexponential_evaluation


def run_level(dataset, week):
    print(
        f"\n🔍 Ejecutando optimización genética para {dataset['name'].iloc[0]} ({dataset['disease'].iloc[0]}, {dataset['classification'].iloc[0]}) - Semana de predicción: {week}")
    genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, MUTATION_PROBABILITY,
                                     subexponential_evaluator(dataset, week),
                                     get_initial_population)
    individual, loss = genetic_agent.run()
    print(f"   ↳ Mejor individuo encontrado: ventana entrenamiento={individual[0]} | loss={loss:.4f}")

    training_window = individual[0]
    loss, x, y_true, y_pred = subexponential_model(dataset, training_window, week, return_predictions=True)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / np.mean(y_true)

    y_pred = np.exp(y_pred)
    y_true = np.exp(y_true)

    # saving results
    disease = dataset['disease'].iloc[0].lower()
    level_name = dataset['name'].iloc[0].lower()
    classification = dataset['classification'].iloc[0].lower()
    plot_directory = get_plot_directory(disease, level_name, classification, 'subexponential')
    results_directory = get_results_directory(disease, level_name, classification, 'subexponential')
    filename = f"{dataset['name'].iloc[0]}_{dataset['classification'].iloc[0]}_{week}".lower()
    print(f"   💾 Guardando resultados en {results_directory}")
    save_as_csv(pd.DataFrame({'Observed': y_true, 'Predicted': y_pred}), f'{filename}.csv',
                output_dir=results_directory)

    title = f"Modelo Subexponencial ({dataset['disease'].iloc[0]})"
    descripcion = f'VP:{week} semanas VE: {training_window} semanas'
    print(f"   📊 Generando gráficos de predicción y dispersión para {filename}")
    graphing.plot_observed_vs_predicted(y_true, y_pred, f'plt_obs_pred_{filename}', output_dir=plot_directory,
                                        title=title, description=descripcion)
    graphing.plot_scatter(y_true, y_pred, f'plt_scatter_{filename}', 'Subexponencial', title=title,
                          description=descripcion, output_dir=plot_directory)
    print(f"   📝 Log de métricas del modelo para {filename}")
    metrics.log_model_metrics('SubExponential', disease, dataset['classification'].iloc[0],
                              dataset['name'].iloc[0], week, mae=mae, mape=mape, nrmse=nrmse, loss=loss, rmse=rmse,
                              hyperparams={'training_window': training_window, 'prediction_window': week})

    return x, y_true, y_pred


def run_subexponential(datasets, weeks):
    for i, dataset in enumerate(datasets):
        print(
            f"\n🚩 Procesando dataset {i + 1}/{len(datasets)}: {dataset['name'].iloc[0]} ({dataset['disease'].iloc[0]}, {dataset['classification'].iloc[0]})")
        series = []
        for week_i in range(1, weeks + 1):
            x, y_true, y_pred = run_level(dataset, week_i)
            series.append([week_i, x, y_pred])

        print(f"   🖼️ Graficando predicciones combinadas para {dataset['name'].iloc[0]}")
        graphing.graficar_predicciones(dataset, series)
    print("\n✅ Finalizó la ejecución de run_subexponential.")


def run_level_wrapper(args):
    dataset, week = args
    x, y_true, y_pred = run_level(dataset, week)
    return [week, x, y_pred]


def run_subexponential_multiprocess(datasets, weeks):
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
            graphing.graficar_predicciones(dataset, series, method="subexponential")
    print("\n✅ Finalizó la ejecución de run_subexponential.")

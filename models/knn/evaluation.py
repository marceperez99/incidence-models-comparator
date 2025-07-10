import pandas as pd
import functools
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

from evaluation.persist import save_as_csv
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.contants import GENERATIONS, POPULATION, MUTATION_PROBABILITY
from utils import get_plot_directory, get_results_directory

from .model import knn_model
from .utils import get_initial_population
from evaluation import graphing, metrics
import concurrent.futures
import os


def knn_evaluator(dataset, weeks):
    @functools.cache
    def knn_evaluation(individual):
        training_window = individual[0]
        n_neighbors = individual[1]
        if training_window <= 1: return float('inf')
        if n_neighbors <= 1: return float('inf')

        loss = knn_model(dataset, training_window, weeks, n_neighbors)
        return loss

    return knn_evaluation


def run_level(dataset, week_i):
    genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, MUTATION_PROBABILITY,
                                     knn_evaluator(dataset, week_i),
                                     get_initial_population)
    individual, loss = genetic_agent.run()

    training_window = individual[0]
    n_neighbors = individual[1]

    loss, x, y_true, y_pred = knn_model(dataset, training_window, week_i, n_neighbors, return_predictions=True)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / np.mean(y_true)

    # saving results
    disease = dataset['disease'].iloc[0].lower()
    level_name = dataset['name'].iloc[0].lower()
    classification = dataset['classification'].iloc[0].lower()
    filename = f"{dataset['name'].iloc[0]}_{dataset['classification'].iloc[0]}_{week_i}".lower()
    save_as_csv(pd.DataFrame({'Observed': y_true, 'Predicted': y_pred}),
                f'{filename}.csv', output_dir=get_results_directory(disease, level_name, classification, 'knn'))

    title = f"Modelo KNN ({dataset['disease'].iloc[0]})"
    descripcion = f'VP:{week_i} semanas VE: {training_window} semanas, Neighbours: {n_neighbors}'
    plot_directory = get_plot_directory(disease, level_name, classification, 'knn')

    t_to_date = dict(zip(dataset['t'], dataset['date']))
    x_dates = [t_to_date.get(ti, pd.NaT) for ti in x]
    graphing.plot_observed_vs_predicted(x_dates, y_true, y_pred, f'plt_obs_pred_{filename}',
                                        output_dir=plot_directory, title=title, description=descripcion)

    graphing.plot_scatter(y_true, y_pred, f'plt_scatter_{filename}', 'KNN', title=title,
                          description=descripcion, output_dir=plot_directory)

    metrics.log_model_metrics('KNN', disease, dataset['classification'].iloc[0],
                              dataset['name'].iloc[0], week_i, mae=mae, mape=mape, nrmse=nrmse, loss=loss,
                              rmse=rmse,
                              hyperparams={
                                  'training_window': training_window,
                                  'prediction_window': week_i,
                                  'n_neighbors': n_neighbors
                              })
    return x, y_true, y_pred


def run_level_wrapper(args):
    dataset, week = args
    x, y_true, y_pred = run_level(dataset, week)
    return [week, x, y_pred]


def run_knn_multiprocess(datasets, weeks):
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
            graphing.graficar_predicciones(dataset, series, method="knn")
    print("\n✅ Finalizó la ejecución de run_knn.")

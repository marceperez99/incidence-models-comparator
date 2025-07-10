import pandas as pd

from evaluation.persist import save_as_csv
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.contants import GENERATIONS, POPULATION, MUTATION_PROBABILITY, NUMBER_OF_BITS
from models.svr.model import svr_model
from models.svr.utils import decode_c, decode_epsilon, get_initial_population
import functools
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from evaluation import graphing, metrics
import concurrent.futures
import os

from utils import get_plot_directory, get_results_directory


def svr_evaluator(dataset, weeks):
    @functools.cache
    def svr_evaluation(individual):
        training_window = individual[0]
        c = decode_c(individual[1], NUMBER_OF_BITS)
        epsilon = decode_epsilon(individual[2], NUMBER_OF_BITS)

        if training_window <= 1 or training_window >= len(dataset): return float('inf')
        if c == 0: return float('inf')

        loss = svr_model(dataset, training_window, weeks, c, epsilon)

        return loss

    return svr_evaluation


def run_level(dataset, week_i):
    print(
        f"\n🔍 Ejecutando optimización genética para {dataset['name'].iloc[0]} ({dataset['disease'].iloc[0]}, {dataset['classification'].iloc[0]}) - Semana de predicción: {week_i}")
    genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, MUTATION_PROBABILITY,
                                     svr_evaluator(dataset, week_i),
                                     get_initial_population)
    individual, loss = genetic_agent.run()
    print(f"   ↳ Mejor individuo encontrado: ventana entrenamiento={individual[0]} | loss={loss:.4f}")
    training_window = individual[0]
    c = decode_c(individual[1], NUMBER_OF_BITS)
    epsilon = decode_epsilon(individual[2], NUMBER_OF_BITS)

    loss, x, y_true, y_pred = svr_model(dataset, training_window, week_i, c, epsilon, return_predictions=True)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / np.mean(y_true)

    # saving results
    disease = dataset['disease'].iloc[0].lower()
    level_name = dataset['name'].iloc[0].lower()
    classification = dataset['classification'].iloc[0].lower()
    plot_directory = get_plot_directory(disease, level_name, classification, 'svr')
    results_directory = get_results_directory(disease, level_name, classification, 'svr')
    filename = f"{dataset['name'].iloc[0]}_{dataset['classification'].iloc[0]}_{week_i}".lower()
    save_as_csv(pd.DataFrame({'Observed': y_true, 'Predicted': y_pred}),
                f'{filename}.csv', output_dir=results_directory)

    title = f"Modelo SVR ({dataset['disease'].iloc[0]})"
    descripcion = f'VP:{week_i} semanas, VE: {training_window} semanas, C: {c}, Epsilon: {epsilon}'
    print(f"   📊 Generando gráficos de predicción y dispersión para {filename}")
    t_to_date = dict(zip(dataset['t'], dataset['date']))
    x_dates = [t_to_date.get(ti, pd.NaT) for ti in x]
    graphing.plot_observed_vs_predicted(x_dates, y_true, y_pred, f'plt_obs_pred_{filename}',
                                        output_dir=plot_directory, title=title, description=descripcion)

    graphing.plot_scatter(y_true, y_pred, f'plt_scatter_{filename}', 'SVR', title=title,
                          description=descripcion, output_dir=plot_directory)
    print(f"   📝 Log de métricas del modelo para {filename}")
    metrics.log_model_metrics('SVR', disease, classification, level_name, week_i, mae=mae, mape=mape,
                              nrmse=nrmse, loss=loss, rmse=rmse,
                              hyperparams={
                                  'training_window': training_window,
                                  'prediction_window': week_i,
                                  'c': c,
                                  'epsilon': epsilon
                              })
    return x, y_true, y_pred


def run_level_wrapper(args):
    dataset, week = args
    x, y_true, y_pred = run_level(dataset, week)
    return [week, x, y_pred]


def run_svr_multiprocess(datasets, weeks):
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
            graphing.graficar_predicciones(dataset, series, method='svr')
    print("\n✅ Finalizó la ejecución de run_svr.")

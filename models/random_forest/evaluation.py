import pandas as pd
import functools
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from evaluation.persist import save_as_csv
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.contants import GENERATIONS, POPULATION, MUTATION_PROBABILITY
from utils import get_plot_directory, get_results_directory
from .model import random_forest_model
from .utils import get_initial_population
from evaluation import graphing, metrics
import concurrent.futures
import os

def random_forest_evaluator(dataset, weeks):
    @functools.cache
    def random_forest_evaluation(individual):
        training_window = individual[0]
        n_estimators = individual[1]
        max_depth = individual[2]
        if training_window <= 1: return float('inf')
        if n_estimators <= 0: return float('inf')
        if max_depth <= 0: return float('inf')
        loss = random_forest_model(dataset, training_window, weeks, n_estimators, max_depth)
        return loss
    return random_forest_evaluation

def run_level(dataset, week_i):
    print(f"\n🔹 [RandomForest] Semana {week_i} - Optimización genética...")
    genetic_agent = GeneticAlgorithm(
        POPULATION, GENERATIONS, MUTATION_PROBABILITY,
        random_forest_evaluator(dataset, week_i),
        get_initial_population
    )
    individual, loss = genetic_agent.run()
    training_window = individual[0]
    n_estimators = individual[1]
    max_depth = individual[2]

    print(f"   ↳ Mejor individuo: ventana={training_window}, estimators={n_estimators}, max_depth={max_depth}")

    loss, x, y_true, y_pred = random_forest_model(
        dataset, training_window, week_i, n_estimators, max_depth, return_predictions=True
    )
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / np.mean(y_true)

    disease = dataset['disease'].iloc[0].lower()
    level_name = dataset['name'].iloc[0].lower()
    classification = dataset['classification'].iloc[0].lower()
    filename = f"{dataset['name'].iloc[0]}_{dataset['classification'].iloc[0]}_{week_i}".lower()

    print(f"   💾 Guardando resultados en {get_results_directory(disease, level_name, classification, 'random_forest')}/{filename}.csv")
    save_as_csv(
        pd.DataFrame({'Observed': y_true, 'Predicted': y_pred}),
        f'{filename}.csv',
        output_dir=get_results_directory(disease, level_name, classification, 'random_forest')
    )

    title = f"Modelo Random Forest ({dataset['disease'].iloc[0]})"
    descripcion = (
        f'VP:{week_i} semanas, VE: {training_window} semanas, '
        f'Estimators: {n_estimators}, Max Depth: {max_depth}'
    )
    plot_directory = get_plot_directory(disease, level_name, classification, 'random_forest')

    print(f"   📊 Graficando resultados en {plot_directory}")
    graphing.plot_observed_vs_predicted(
        y_true, y_pred, f'plt_obs_pred_{filename}',
        output_dir=plot_directory, title=title, description=descripcion
    )
    graphing.plot_scatter(
        y_true, y_pred, f'plt_scatter_{filename}', 'Random Forest', title=title,
        description=descripcion, output_dir=plot_directory
    )
    print(f"   📝 Log de métricas para semana {week_i}")
    metrics.log_model_metrics(
        'Random Forest', disease, dataset['classification'].iloc[0],
        dataset['name'].iloc[0], week_i, mae=mae, mape=mape, nrmse=nrmse, loss=loss,
        rmse=rmse,
        hyperparams={
            'training_window': training_window,
            'prediction_window': week_i,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
        }
    )
    return week_i, x, y_pred

def run_level_wrapper(args):
    dataset, week = args
    week_i, x, y_pred = run_level(dataset, week)
    return [week_i, x, y_pred]

def run_random_forest_multiprocess(datasets, weeks):
    print(f"\n⚡ Ejecutando en paralelo con {os.cpu_count()} procesos disponibles ({len(datasets)} datasets)...")
    for i, dataset in enumerate(datasets):
        print(
            f"\n🚩 Procesando dataset {i + 1}/{len(datasets)}: "
            f"{dataset['name'].iloc[0]} ({dataset['disease'].iloc[0]}, {dataset['classification'].iloc[0]})"
        )
        args_list = [(dataset, i) for i in range(1, weeks + 1)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            series = list(executor.map(run_level_wrapper, args_list))
            print(f"   🖼️ Graficando predicciones combinadas para {dataset['name'].iloc[0]}")
            graphing.graficar_predicciones(dataset, series, method="random_forest")
    print("\n✅ Finalizó la ejecución de run_random_forest.")


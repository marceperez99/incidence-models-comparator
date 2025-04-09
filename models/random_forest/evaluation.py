import pandas as pd
from evaluation.persist import save_as_csv
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.contants import GENERATIONS, POPULATION, MUTATION_PROBABILITY
from .model import random_forest_model
from .utils import  get_initial_population
import functools
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from evaluation import graphing, metrics


def random_forest_evaluator(dataset, weeks):
    @functools.cache
    def random_forest_evaluation(individual):
        training_window = individual[0]
        n_estimators = individual[1]
        max_depth = individual[2]

        if training_window <= 1: return float('inf')
        if n_estimators <= 0: return float('inf')

        loss = random_forest_model(dataset, training_window, weeks, n_estimators, max_depth)

        return loss

    return random_forest_evaluation


def run_random_forest(datasets, weeks):
    for dataset in datasets:

        for week_i in range(1, weeks + 1):
            genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, MUTATION_PROBABILITY,
                                             random_forest_evaluator(dataset, week_i),
                                             get_initial_population)
            individual, loss = genetic_agent.run()

            training_window = individual[0]
            n_estimators = individual[1]
            max_depth = individual[2]

            loss, y_true, y_pred = random_forest_model(dataset, training_window, weeks, n_estimators, max_depth, return_predictions=True)
            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            nrmse = rmse / np.mean(y_true)

            # saving results
            disease = dataset['disease'].iloc[0].lower()
            filename = f"{dataset['name'].iloc[0]}_{dataset['classification'].iloc[0]}_{week_i}".lower()
            save_as_csv(pd.DataFrame({'Observed': y_true, 'Predicted': y_pred}),
                        f'{filename}.csv', output_dir=f'outputs/predictions/random_forest/{disease}')

            title = f"Modelo Random Forest ({dataset['disease'].iloc[0]})"
            descripcion = f'VP:{week_i} semanas, VE: {training_window} semanas, Estimators: {n_estimators}, Max Depth: {max_depth}'
            graphing.plot_observed_vs_predicted(y_true, y_pred, f'plt_obs_pred_{filename}',
                                                output_dir=f'outputs/plots/random_forest/{disease}', title=title,
                                                description=descripcion)
            graphing.plot_scatter(y_true, y_pred, f'plt_scatter_{filename}', 'Random Forest', title=title,
                                  description=descripcion, output_dir=f'outputs/plots/random_forest/{disease}')
            metrics.log_model_metrics('Random Forest', disease, dataset['classification'].iloc[0],
                                      dataset['name'].iloc[0], week_i, mae=mae, mape=mape, nrmse=nrmse, loss=loss,
                                      rmse=rmse,
                                      hyperparams={
                                          'training_window': training_window,
                                          'prediction_window': week_i,
                                          'max_depth': max_depth,
                                          'n_estimators': n_estimators,
                                      })

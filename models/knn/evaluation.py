import pandas as pd
import functools
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

from evaluation.persist import save_as_csv
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.contants import GENERATIONS, POPULATION, MUTATION_PROBABILITY

from .model import knn_model
from .utils import get_initial_population
from evaluation import graphing, metrics


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


def run_knn(datasets, weeks):
    for dataset in datasets:

        for week_i in range(1, weeks + 1):
            genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, MUTATION_PROBABILITY,
                                             knn_evaluator(dataset, week_i),
                                             get_initial_population)
            individual, loss = genetic_agent.run()

            training_window = individual[0]
            n_neighbors = individual[1]

            loss, y_true, y_pred = knn_model(dataset, training_window, week_i, n_neighbors, return_predictions=True)
            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            nrmse = rmse / np.mean(y_true)

            y_pred = np.exp(y_pred)
            y_true = np.exp(y_true)
            # saving results
            disease = dataset['disease'].iloc[0].lower()
            filename = f"{dataset['name'].iloc[0]}_{dataset['classification'].iloc[0]}_{week_i}".lower()
            save_as_csv(pd.DataFrame({'Observed': y_true, 'Predicted': y_pred}),
                        f'{filename}.csv', output_dir=f'outputs/predictions/knn/{disease}')

            title = f"Modelo KNN ({dataset['disease'].iloc[0]})"
            descripcion = f'VP:{week_i} semanas VE: {training_window} semanas, '
            graphing.plot_observed_vs_predicted(y_true, y_pred, f'plt_obs_pred_{filename}',
                                                output_dir=f'outputs/plots/knn/{disease}', title=title,
                                                description=descripcion)
            graphing.plot_scatter(y_true, y_pred, f'plt_scatter_{filename}', 'KNN', title=title,
                                  description=descripcion, output_dir=f'outputs/plots/knn/{disease}')
            metrics.log_model_metrics('KNN', disease, dataset['classification'].iloc[0],
                                      dataset['name'].iloc[0], mae=mae, mape=mape, nrmse=nrmse, loss=loss, rmse=rmse,
                                      hyperparams={
                                          'training_window': training_window,
                                          'prediction_window': week_i,
                                          'n_neighbors': n_neighbors
                                      })

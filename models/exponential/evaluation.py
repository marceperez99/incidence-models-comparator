import pandas as pd
import functools
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np

from evaluation.persist import save_as_csv
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from genetic_algorithm.contants import GENERATIONS, POPULATION, MUTATION_PROBABILITY, NUMBER_OF_BITS

from .model import exponential_model
from .utils import get_initial_population
from evaluation import graphing


def exponential_evaluator(dataset, weeks):
    @functools.cache
    def exponential_evaluation(individual):
        if individual[0] <= 1: return float('inf')
        training_window = individual[0]

        loss = exponential_model(dataset, training_window, weeks)

        return loss

    return exponential_evaluation


def run_exponential(datasets, weeks):
    for dataset in datasets:

        for week_i in range(1, weeks + 1):
            genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, MUTATION_PROBABILITY,
                                             exponential_evaluator(dataset, week_i),
                                             get_initial_population)
            individual, loss = genetic_agent.run()

            training_window = individual[0]

            loss, y_true, y_pred = exponential_model(dataset, training_window, week_i, return_predictions=True)
            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            nrmse = rmse / np.mean(y_true)

            y_pred = np.exp(y_pred)
            y_true = np.exp(y_true)
            # saving results
            filename = f"exponential_{dataset['disease'].iloc[0]}_{dataset['level'].iloc[0]}_{dataset['classification'].iloc[0]}_{week_i}".lower()
            save_as_csv(pd.DataFrame({'Observed': y_true, 'Predicted': y_pred}),
                        f'predictions/{filename}.csv')

            title = f"Modelo Exponencial ({dataset['disease'].iloc[0]})"
            descripcion = f'VP:{week_i} semanas VE: {training_window} semanas'
            graphing.plot_observed_vs_predicted(y_true, y_pred, f'plt_obs_pred_{filename}',
                                                title=title, description=descripcion)
            graphing.plot_scatter(y_true, y_pred, f'plt_scatter_{filename}','Exponential', title=title, description=descripcion)

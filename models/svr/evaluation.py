import pandas as pd

from evaluation.persist import save_as_csv
from evolutionary_algorithm import GeneticAlgorithm
from genetic_algorithm.contants import GENERATIONS, POPULATION, MUTATION_PROBABILITY, NUMBER_OF_BITS
from models.svr.model import svr_model
from models.svr.utils import decode_c, decode_epsilon, get_initial_population
import functools
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np


def svr_evaluator(dataset, weeks):
    @functools.cache
    def svr_evaluation(individual):

        if individual[0] <= 1: return float('inf')
        if individual[1] == 0: return float('inf')

        training_window = individual[0]
        c = decode_c(individual[1], NUMBER_OF_BITS)
        epsilon = decode_epsilon(individual[2], NUMBER_OF_BITS)

        loss = svr_model(dataset, training_window, weeks, c, epsilon)

        return loss

    return svr_evaluation


def run_svr(datasets, weeks):
    for dataset in datasets:

        for week_i in range(1, weeks + 1):
            genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, MUTATION_PROBABILITY,
                                             svr_evaluator(dataset, week_i),
                                             get_initial_population)
            individual, loss = genetic_agent.run()

            training_window = individual[0]
            c = decode_c(individual[1], NUMBER_OF_BITS)
            epsilon = decode_epsilon(individual[2], NUMBER_OF_BITS)

            loss, y_true, y_pred = svr_model(dataset, training_window, weeks, c, epsilon, return_predictions=True)
            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            nrmse = rmse / np.mean(y_true)

            filename = f"svr_{dataset['disease'].iloc[0]}_{dataset['level'].iloc[0]}_{dataset['classification'].iloc[0]}_{week_i}.csv"
            save_as_csv(pd.DataFrame({'Observed': y_true, 'Predicted': y_pred}), f'predictions/{filename}')

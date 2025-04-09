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


def svr_evaluator(dataset, weeks):
    @functools.cache
    def svr_evaluation(individual):
        training_window = individual[0]
        c = decode_c(individual[1], NUMBER_OF_BITS)
        epsilon = decode_epsilon(individual[2], NUMBER_OF_BITS)

        if training_window <= 1: return float('inf')
        if c == 0: return float('inf')

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

            loss, y_true, y_pred = svr_model(dataset, training_window, week_i, c, epsilon, return_predictions=True)
            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            nrmse = rmse / np.mean(y_true)

            # saving results
            disease = dataset['disease'].iloc[0].lower()
            filename = f"{dataset['name'].iloc[0]}_{dataset['classification'].iloc[0]}_{week_i}".lower()
            save_as_csv(pd.DataFrame({'Observed': y_true, 'Predicted': y_pred}),
                        f'{filename}.csv', output_dir=f'outputs/predictions/svr/{disease}')

            title = f"Modelo SVR ({dataset['disease'].iloc[0]})"
            descripcion = f'VP:{week_i} semanas, VE: {training_window} semanas, C: {c}, Epsilon: {epsilon}'
            graphing.plot_observed_vs_predicted(y_true, y_pred, f'plt_obs_pred_{filename}',
                                                output_dir=f'outputs/plots/svr/{disease}', title=title,
                                                description=descripcion)
            graphing.plot_scatter(y_true, y_pred, f'plt_scatter_{filename}', 'SVR', title=title,
                                  description=descripcion, output_dir=f'outputs/plots/svr/{disease}')
            metrics.log_model_metrics('SVR', disease, dataset['classification'].iloc[0],
                                      dataset['name'].iloc[0], week_i, mae=mae, mape=mape, nrmse=nrmse, loss=loss,
                                      rmse=rmse,
                                      hyperparams={
                                          'training_window': training_window,
                                          'prediction_window': week_i,
                                          'c': c,
                                          'epsilon': epsilon
                                      })

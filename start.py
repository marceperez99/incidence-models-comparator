import sys

from models.artificial_neural_network import search_best_ann_architecture
from models.autoarima import autoarima_model
from genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from models.knn import knn_model, knn_evaluator
from models.lstm import lstm_model

from models.random_forest import random_forest_evaluator
from models.SVR import svr_evaluator

from population import get_initial_population, DATASET_NAME, get_initial_population_knn
from population import get_initial_population_random_forest
from population import get_initial_population_svr
from population import dataset
import utils

POPULATION = 100
GENERATIONS = 1000
args = sys.argv

match args[1]:



    case 'random_forest':
        genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, random_forest_evaluator(dataset, 4),
                                         get_initial_population_random_forest)
        loss, params = genetic_agent.run()
        print(loss, params)
    case 'svr':
        genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, svr_evaluator(dataset, 4),
                                         get_initial_population_svr)
        params, loss = genetic_agent.run()

    case 'ann':
        architectures = [
            [16, 8],
            [32, 16],
            [32, 16, 8],
            [64, 32],
            [64, 32, 16],
            [32, 32],
            [16],  # muy simple, buen baseline
            [128, 64, 32],  # más profunda, solo si no hay sobreajuste
        ]

        best_model, best_architecture, best_loss = search_best_ann_architecture(
            dataset,
            lagged_number_of_weeks=4,
            prediction_window=1,
            architectures=architectures
        )

        print(best_model, best_architecture, best_loss)

    case 'arima':
        loss = autoarima_model(dataset, 4)
        utils.log_experiment({
            'prediction_window': 4
        }, loss, 'AutoArima',
            './data/results.csv', DATASET_NAME)
        loss = autoarima_model(dataset, 3)
        utils.log_experiment({
            'prediction_window': 3
        }, loss, 'AutoArima',
            './data/results.csv', DATASET_NAME)
        loss = autoarima_model(dataset, 2)
        utils.log_experiment({
            'prediction_window': 2
        }, loss, 'AutoArima',
            './data/results.csv', DATASET_NAME)
        loss = autoarima_model(dataset, 1)
        utils.log_experiment({
            'prediction_window': 1
        }, loss, 'AutoArima',
            './data/results.csv', DATASET_NAME)

    case 'lstm':
        loss, model = lstm_model(dataset, 5, 1,True)
        print(loss)
    case _:
        print('Invalid model')
        exit()

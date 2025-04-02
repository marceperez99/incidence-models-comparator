import sys
from models.autoarima import autoarima_model
from evolutionary_algorithm import GeneticAlgorithm
from models.subexponential import subexponential_evaluator, subexponential_model
from models.exponential import exponential_evaluator, exponential_model
from models.random_forest import random_forest_evaluator
from models.SVR import svr_evaluator
from models.subexponential_amort import subexp_amort_evaluator, subexp_amort_model

from population import get_initial_population, DATASET_NAME

from population import get_initial_population_random_forest
from population import get_initial_population_svr
from population import dataset
import utils

POPULATION = 100
GENERATIONS = 1000
args = sys.argv

match args[1]:
    case 'exponential':
        genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, exponential_evaluator(dataset),
                                         get_initial_population)
        best_individual, error = genetic_agent.run()
        utils.log_experiment({
            'training_window': best_individual[0],
            'prediction_window': best_individual[1]
        }, error, 'Exponential',
            './data/results.csv', DATASET_NAME)
        print(best_individual, error)
        exponential_model(dataset, best_individual[0], best_individual[1], plot=True)

    case 'subexponential':
        genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, subexponential_evaluator(dataset),
                                         get_initial_population)
        best_individual, error = genetic_agent.run()
        utils.log_experiment({
            'training_window': best_individual[0],
            'prediction_window': best_individual[1]
        }, error, 'SubExponential',
            './data/results.csv', DATASET_NAME)
        print(best_individual, error)
        subexponential_model(dataset, best_individual[0], best_individual[1], plot=True)
    case 'subexponential-amortized':
        genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, subexp_amort_evaluator(dataset),
                                         get_initial_population)
        best_individual, error = genetic_agent.run()
        utils.log_experiment({
            'training_window': best_individual[0],
            'prediction_window': best_individual[1]
        }, error, 'SubExponential Amortized',
            './data/results.csv', DATASET_NAME)
        print(best_individual, error)
        subexp_amort_model(dataset, best_individual[0], best_individual[1], plot=True)
    case 'random_forest':
        genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, random_forest_evaluator(dataset),
                                         get_initial_population_random_forest)
    case 'svr':
        genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, svr_evaluator(dataset),
                                         get_initial_population_svr)
    case 'arima':
        loss = autoarima_model(dataset, 4)
        exit()
    case _:
        print('Invalid model')
        exit()

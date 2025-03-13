import sys
from models.autoarima import autoarima_model
from evolutionary_algorithm import GeneticAlgorithm
from models.subexponential import subexponential_evaluator
from models.exponential import exponential_evaluator
from models.random_forest import random_forest_evaluator
from models.SVR import svr_evaluator

from population import get_initial_population
from population import get_initial_population_ann
from population import get_initial_population_random_forest
from population import get_initial_population_svr
from population import dataset

POPULATION = 100
GENERATIONS = 50
args = sys.argv

match args[1]:
    case 'exponential':
        genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, exponential_evaluator(dataset),
                                         get_initial_population)
    case 'subexponential':
        genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, subexponential_evaluator(dataset),
                                         get_initial_population)
    case 'random_forest':
        genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, random_forest_evaluator(dataset),
                                         get_initial_population_random_forest)
    case 'svr':
        genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, svr_evaluator(dataset),
                                         get_initial_population_svr)
    case 'arima':
        loss = autoarima_model(dataset, 4)
    case _:
        print('Invalid model')
        exit()

best_individual, error = genetic_agent.run()
print(best_individual, error)

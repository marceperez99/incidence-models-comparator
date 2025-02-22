from evolutionary_algorithm import GeneticAlgorithm
from models.exponential import  exponential_evaluator
import numpy as np
import sys

from models.subexponential import subexponential_evaluator
from utils import get_dataset

POPULATION = 100
GENERATIONS = 50
dataset = get_dataset('./case_data.csv')
args = sys.argv

def get_initial_population(size):
    return [(np.random.randint(low=2, high=dataset.shape[0] // 2), np.random.randint(low=2, high=6)) for _ in
            range(size - 1)]

def get_initial_population_ann(size):
    return [(np.random.randint(low=1, high=10), np.random.randint(low=1, high=4)) for _ in range(size - 1)]


# Exponential model
if args[1] == 'exponential':

    genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, exponential_evaluator(dataset), get_initial_population)
    best_individual, error = genetic_agent.run()
    print(best_individual, error)
elif args[1] == 'subexponential':
    genetic_agent = GeneticAlgorithm(POPULATION, GENERATIONS, 0.1, subexponential_evaluator(dataset), get_initial_population)
    best_individual, error = genetic_agent.run()
    print(best_individual, error)
else:
    print('Invalid model')

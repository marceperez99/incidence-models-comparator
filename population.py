import numpy as np
from utils import get_dataset

ENFERMEDAD = 'Dengue'
# ENFERMEDAD = 'Chikungunya'
CLASIFICACION =  'Confirmado'
DEPARTAMENTO = 'Central'

DATASET_NAME = f'{ENFERMEDAD.lower()}_{CLASIFICACION.lower()}_{DEPARTAMENTO.lower()}.csv'

dataset = get_dataset(f'./data/{DATASET_NAME}')

def get_initial_population(size):
    return [(np.random.randint(low=2, high=dataset.shape[0] // 2), np.random.randint(low=2, high=6)) for _ in
            range(size - 1)]


def get_initial_population_ann(size):
    return [(np.random.randint(low=1, high=10), np.random.randint(low=1, high=4)) for _ in range(size - 1)]


def get_initial_population_random_forest(size):
    return [
        (
            np.random.randint(low=2, high=dataset.shape[0] // 2),
            np.random.randint(low=2, high=6),
            np.random.choice([50, 100, 200, 300, 500]),
            np.random.choice([5, 10, 20, 30])
        ) for _ in range(size)
    ]
def get_initial_population_svr(size):
    return [
        (
            np.random.randint(low=2, high=dataset.shape[0] // 2),  # Training window
            np.random.randint(low=2, high=6),  # Prediction window
            np.random.randint(low=0, high=10),
            np.random.randint(low=0, high=10)
            #np.random.uniform(low=0.1, high=10.0),  # C parameter (Regularization)
            #np.random.uniform(low=0.01, high=1.0)  # Epsilon (Tolerance margin)
        ) for _ in range(size)
    ]

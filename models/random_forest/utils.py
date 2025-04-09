import numpy as np


def get_initial_population(size):
    return [
        (
            np.random.randint(low=2, high=20),  # Training window
            np.random.choice([50, 100, 200, 300, 500]),  # N estimator
            np.random.choice([5, 10, 20, 30])  # Max depth
        ) for _ in range(size)
    ]

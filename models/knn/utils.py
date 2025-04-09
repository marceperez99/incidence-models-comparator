import numpy as np


def get_initial_population(size):
    return [
        (
            np.random.randint(low=2, high=10),  # Training window
            np.random.randint(low=1, high=10),
        ) for _ in range(size)
    ]

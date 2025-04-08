import numpy as np


def get_initial_population(size):
    return [
        (
            np.random.randint(low=2, high=10),  # Training window
            np.random.randint(low=0, high=10),  # C
            np.random.randint(low=0, high=10)   # Epsilon
        ) for _ in range(size)
    ]


def decode_c(c_int, number_of_bits, min_c=0.01, max_c=100):
    max_int = 2 ** number_of_bits - 1
    return min_c + (c_int / max_int) * (max_c - min_c)


def decode_epsilon(epsilon, number_of_bits):
    return epsilon / (2 ** number_of_bits - 1)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import functools
import matplotlib.pyplot as plt
from enum import Enum

import utils
from genetic_algorithm.genetic_algorithm import NUMBER_OF_BITS
from population import ENFERMEDAD


class SVRParameter(Enum):
    TRAINING_WINDOW = 0
    # PREDICTION_WINDOW = 1
    C = 1
    EPSILON = 2


def svr_model(dataset, training_window, prediction_window, C, epsilon, plot=False):

    dataset = dataset.copy()

    dataset['target'] = dataset['i_cases']

    predicted_values = []
    observed_values = []

    dataset = dataset.sort_values(by=['t'])
    dataset = dataset.set_index('t')  # al principio


    for i in range(len(dataset) - training_window - prediction_window + 1):

        section = dataset.iloc[i:i + training_window]

        X_train = section.index.values.reshape(-1, 1)

        y_train = section['target'].values

        svr = SVR(kernel='linear', C=float(C), epsilon=epsilon)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        svr.fit(X_train_scaled, y_train)


        test_x = (section.index.max() + np.arange(0, prediction_window * 7, 7)).reshape(-1, 1)
        test_x = test_x.ravel()
        test_x_scaled = scaler.transform(test_x.reshape(-1, 1))
        test_y = svr.predict(test_x_scaled)

        filtered_dataset = dataset.loc[test_x] if set(test_x).issubset(dataset.index) else None

        if filtered_dataset.shape[0] == len(test_x):
            predicted_values.append(max(0, test_y[-1]))
            observed_values.append(filtered_dataset['target'].values[-1])

    if plot:
        plt.figure(figsize=(8, 5))
        plt.title(f'Modelo SRV / {ENFERMEDAD} (VE: {training_window}, VP: {prediction_window}, C: {round(C, 4)}, epsilon: {round(epsilon, 4)})')

        plt.plot(observed_values, label="Observed")
        plt.plot(predicted_values, label="Predicted")
        plt.legend()
        plt.show()

    # Compute and return the loss function (e.g., MAE)
    return utils.loss_function(predicted_values, observed_values)

def decode_c(c_int, number_of_bits, min_c=0.01, max_c=100):
    max_int = 2 ** number_of_bits - 1
    return min_c + (c_int / max_int) * (max_c - min_c)

def decode_epsilon(epsilon, number_of_bits):
    return epsilon / (2**number_of_bits - 1)

def svr_evaluator(dataset, weeks):
    @functools.cache
    def svr_evaluation(individual):

        if individual[SVRParameter.TRAINING_WINDOW.value] <= 1:
            return float('inf')
        # if individual[SVRParameter.PREDICTION_WINDOW.value] == 0:
        #     return float('inf')
        # if individual[SVRParameter.PREDICTION_WINDOW.value] > 5:
        #     return float('inf')
        if individual[SVRParameter.C.value] == 0:
            return float('inf')

        loss = svr_model(
            dataset,
            individual[SVRParameter.TRAINING_WINDOW.value],
            weeks, # individual[SVRParameter.PREDICTION_WINDOW.value],
            decode_c(individual[SVRParameter.C.value], NUMBER_OF_BITS),
            decode_epsilon(individual[SVRParameter.EPSILON.value])
        )

        return loss

    return svr_evaluation

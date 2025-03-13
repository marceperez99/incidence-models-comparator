from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import functools
import matplotlib.pyplot as plt
from enum import Enum

import utils


class SVRParameter(Enum):
    TRAINING_WINDOW = 0
    PREDICTION_WINDOW = 1
    C = 2
    EPSILON = 3


def svr_model(dataset, training_window, prediction_window, C, epsilon, plot=False):
    dataset = dataset.copy()
    dataset['target'] = dataset['i_cases']

    predicted_values = []
    observed_values = []

    dataset = dataset.sort_values(by=['t'])

    for i in range(len(dataset) - training_window - prediction_window + 1):
        section = dataset.iloc[i:i + training_window]

        X_train = section[['t']].values
        y_train = section['target'].values

        svr = SVR(kernel='linear', C=float(C), epsilon=epsilon)
        svr.fit(X_train, y_train)

        test_x = (section['t'].max(skipna=True) + np.arange(0, prediction_window * 7, 7)).reshape(-1, 1)
        test_x = test_x.ravel()

        test_y = svr.predict(test_x.reshape(-1, 1))

        filtered_dataset = dataset[dataset['t'].isin(test_x)]
        if filtered_dataset.shape[0] == len(test_x):
            predicted_values.append(test_y[-1])
            observed_values.append(filtered_dataset['target'].values[-1])

    if plot:
        plt.figure(figsize=(8, 5))
        plt.title(f'SVR - Training: {training_window}, Prediction: {prediction_window}')
        plt.plot(observed_values, label="Observed", marker='o')
        plt.plot(predicted_values, label="Predicted", marker='x')
        plt.legend()
        plt.show()

    # Compute and return the loss function (e.g., MAE)
    return utils.loss_function(predicted_values, observed_values)


def svr_evaluator(dataset):
    @functools.cache
    def svr_evaluation(individual):
        if individual[SVRParameter.TRAINING_WINDOW.value] <= 1:
            return float('inf')
        if individual[SVRParameter.PREDICTION_WINDOW.value] == 0:
            return float('inf')
        if individual[SVRParameter.PREDICTION_WINDOW.value] > 5:
            return float('inf')

        loss = svr_model(
            dataset,
            individual[SVRParameter.TRAINING_WINDOW.value],
            individual[SVRParameter.PREDICTION_WINDOW.value],
            individual[SVRParameter.C.value],
            individual[SVRParameter.EPSILON.value]
        )
        utils.log_experiment({
            'training_window': individual[SVRParameter.TRAINING_WINDOW.value],
            'prediction_window': individual[SVRParameter.PREDICTION_WINDOW.value],
            'C': individual[SVRParameter.C.value],
            'epsilon': individual[SVRParameter.EPSILON.value]
        }, loss, 'SVR', './results.csv')

        return loss

    return svr_evaluation

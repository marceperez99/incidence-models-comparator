import numpy as np
import pandas as pd
import functools
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from enum import Enum


import utils


class RandomForestParameter(Enum):
    TRAINING_WINDOW = 0
    PREDICTION_WINDOW = 1
    N_ESTIMATORS = 2
    MAX_DEPTH = 3



def random_forest_model(dataset, training_window, prediction_window, n_estimators, max_depth, plot=False):
    dataset = dataset.copy()
    dataset['target'] = dataset['i_cases']

    predicted_values = []
    observed_values = []

    dataset = dataset.sort_values(by=['t'])

    for i in range(len(dataset) - training_window - prediction_window + 1):  # Fix loop range
        section = dataset.iloc[i:i + training_window]

        X_train = section[['t']].values
        y_train = section['target'].values

        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)

        #
        test_x = (section['t'].max(skipna=True) + np.arange(0, prediction_window * 7, 7)).reshape(-1, 1)
        test_x = test_x.ravel()


        test_y = rf.predict(test_x.reshape(-1, 1))

        filtered_dataset = dataset[dataset['t'].isin(test_x)]
        if filtered_dataset.shape[0] == len(test_x):
            predicted_values.append(test_y[-1])
            observed_values.append(filtered_dataset['target'].values[-1])

    if plot:
        plt.figure(figsize=(8, 5))
        plt.title(f'Random Forest - Training: {training_window}, Prediction: {prediction_window}')
        plt.plot(observed_values, label="Observed", marker='o')
        plt.plot(predicted_values, label="Predicted", marker='x')
        plt.legend()
        plt.show()

    # Compute and return the loss function (e.g., MAE)
    return utils.loss_function(predicted_values, observed_values)



def random_forest_evaluator(dataset):
    @functools.cache
    def random_forest_evaluation(individual):
        if individual[RandomForestParameter.TRAINING_WINDOW.value] <= 1:
            return float('inf')
        if individual[RandomForestParameter.PREDICTION_WINDOW.value] == 0:
            return float('inf')
        if individual[RandomForestParameter.PREDICTION_WINDOW.value] > 5:
            return float('inf')
        loss = random_forest_model(
            dataset,
            individual[RandomForestParameter.TRAINING_WINDOW.value],
            individual[RandomForestParameter.PREDICTION_WINDOW.value],
            individual[RandomForestParameter.N_ESTIMATORS.value],
            individual[RandomForestParameter.MAX_DEPTH.value]
        )
        utils.log_experiment({
            'training_window': individual[RandomForestParameter.TRAINING_WINDOW.value],
            'prediction_window': individual[RandomForestParameter.PREDICTION_WINDOW.value],
            'n_estimators': individual[RandomForestParameter.N_ESTIMATORS.value],
            'max_depth': individual[RandomForestParameter.MAX_DEPTH.value]
        }, loss, 'RandomForest',
            './results.csv')
        return loss

    return random_forest_evaluation

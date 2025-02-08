import random
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import functools
import matplotlib.pyplot as plt

def exponential_model(dataset, training_window, prediction_window, plot=False):
    dataset = dataset.copy()
    dataset['target'] = dataset['i_cases']

    predicted_values = []
    observed_values = []
    # Iterate over the data with a sliding window
    for i in range(len(dataset) - training_window + 1):
        # Take a section of `training_window` rows
        section = dataset.iloc[i:i + training_window]

        # For example, print or process each section individually
        x = section['t'].values.reshape(-1, 1)
        y = np.log(section['target'])
        # training of model
        exponential = LinearRegression()
        exponential.fit(x, y)
        test_x = (section['t'].max(skipna=True) + np.arange(0, prediction_window * 7, 7))
        test_y = exponential.predict(test_x.reshape(-1, 1))

        filtered_dataset = dataset[dataset['t'].isin(test_x)]
        if filtered_dataset.shape[0] == test_x.shape[0]:
            predicted_values.append(test_y[-1])
            observed_values.append(np.log(filtered_dataset['target'].to_numpy()[-1]))

    if plot:
        plt.figure(figsize=(8, 5))
        plt.title(f'Training: {training_window}, Prediction: {prediction_window}')
        plt.plot(np.exp(observed_values), label="Observed")
        plt.plot(np.exp(predicted_values), label="Predicted")
        plt.show()

    return mean_absolute_percentage_error(predicted_values, observed_values)

def exponential_evaluator(dataset):

    @functools.cache
    def exponential_evaluation(individual):
        if individual[0] <= 1: return float('inf')
        if individual[1] == 0 or individual[1] > 5: return float('inf')
        return exponential_model(dataset, individual[0], individual[1])* individual[0] / individual[1]

    return exponential_evaluation
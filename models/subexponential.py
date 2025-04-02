import numpy as np
from sklearn.linear_model import LinearRegression
import functools
import matplotlib.pyplot as plt
import utils
from population import ENFERMEDAD


def subexponential_model(dataset, training_window, prediction_window, plot=False):
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
        exponential.fit(np.log(x), y)
        test_x = (section['t'].max(skipna=True) + np.arange(0, prediction_window * 7, 7))
        test_y = exponential.predict(np.log(test_x.reshape(-1, 1)))

        filtered_dataset = dataset[dataset['t'].isin(test_x)]
        if filtered_dataset.shape[0] == test_x.shape[0]:
            predicted_values.append(test_y[-1])
            observed_values.append(np.log(filtered_dataset['target'].to_numpy()[-1]))

    if plot:
        plt.figure(figsize=(8, 5))
        plt.title(f'Modelo Subexponencial / {ENFERMEDAD} (VE: {training_window}, VP: {prediction_window})')
        plt.xlabel('Nro de semana epidemiologica')
        plt.ylabel('Nro de casos')
        plt.plot(np.exp(observed_values), label="Observaciones")
        plt.plot(np.exp(predicted_values), label="Predicciones")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return utils.loss_function(predicted_values, observed_values)

def subexponential_evaluator(dataset):

    @functools.cache
    def exponential_evaluation(individual):
        if individual[0] <= 1: return float('inf')
        if individual[1] == 0 or individual[1] > 5: return float('inf')

        loss = subexponential_model(dataset, individual[0], individual[1])
        utils.log_experiment({
            'training_window': individual[0],
            'prediction_window': individual[1]
        }, loss, 'Sub-Exponential',
            './results.csv')
        return loss

    return exponential_evaluation
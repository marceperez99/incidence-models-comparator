import numpy as np
from sklearn.linear_model import LinearRegression
import functools
import matplotlib.pyplot as plt
import utils
from population import ENFERMEDAD


def subexp_amort_model(dataset, training_window, prediction_window, plot=False):
    dataset = dataset.copy()
    dataset['target'] = dataset['i_cases']

    predicted_values = []
    observed_values = []

    for i in range(len(dataset) - training_window + 1):
        section = dataset.iloc[i:i + training_window]

        # Preparamos las variables para regresión lineal
        t = section['t'].values
        y = np.log(section['target'].values + 1e-8)  # Evita log(0)

        # Creamos los features: log(t) y t
        X = np.column_stack((np.log(t + 1e-8), t))

        model = LinearRegression()
        model.fit(X, y)


        # Realizamos la predicción
        test_t = section['t'].max(skipna=True) + np.arange(0, prediction_window * 7, 7)
        X_test = np.column_stack((np.log(test_t + 1e-8), test_t))
        y_pred = model.predict(X_test)

        filtered_dataset = dataset[dataset['t'].isin(test_t)]
        if filtered_dataset.shape[0] == test_t.shape[0]:
            predicted_values.append(y_pred[-1])
            observed_values.append(np.log(filtered_dataset['target'].to_numpy()[-1] + 1e-8))

    if plot:
        plt.figure(figsize=(8, 5))
        plt.title(f'Modelo Subexponencial amortizado / {ENFERMEDAD} (VE: {training_window}, VP: {prediction_window})')
        plt.xlabel('Nro de semana epidemiologica')
        plt.ylabel('Nro de casos')
        plt.plot(np.exp(observed_values), label="Observaciones")
        plt.plot(np.exp(predicted_values), label="Predicciones")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return utils.loss_function(predicted_values, observed_values)

def subexp_amort_evaluator(dataset):

    @functools.cache
    def amort_evaluation(individual):
        if individual[0] <= 1: return float('inf')
        if individual[1] == 0 or individual[1] > 5: return float('inf')

        loss = subexp_amort_model(dataset, individual[0], individual[1])
        return loss

    return amort_evaluation

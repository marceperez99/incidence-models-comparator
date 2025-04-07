from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt
import utils
from enum import Enum
import functools

class KNNParameter(Enum):
    TRAINING_WINDOW = 0
    NEIGHBOURS = 1

def knn_model(dataset, training_window, prediction_window, n_neighbors=5, plot=False):
    dataset = dataset.copy()
    dataset['target'] = dataset['i_cases']
    dataset = dataset.sort_values(by='t')
    dataset = dataset.set_index('t')  # para búsquedas rápidas
    predicted_values = []
    observed_values = []

    for i in range(len(dataset) - training_window - prediction_window + 1):
        section = dataset.iloc[i:i + training_window]

        X_train = section.index.values.reshape(-1, 1)
        t_mean = X_train.mean()
        X_train_centered = X_train - t_mean
        y_train = section['target'].values

        n_neighbors_safe = min(n_neighbors, len(X_train_centered))
        knn = KNeighborsRegressor(n_neighbors=n_neighbors_safe)
        knn.fit(X_train_centered, y_train)

        # Generar fechas futuras centradas
        future_t = section.index.max() + np.arange(0, prediction_window * 7, 7)
        test_x = (future_t - t_mean).reshape(-1, 1)

        test_y = knn.predict(test_x)
        test_y = [max(0, y) for y in test_y]  # truncar negativos

        try:
            real_y = dataset.loc[future_t]['target'].values
        except KeyError:
            continue

        if len(real_y) == len(test_y):
            predicted_values.append(test_y[-1])
            observed_values.append(real_y[-1])

    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(observed_values, label='Observed')
        plt.plot(predicted_values, label='Predicted')
        plt.title(f'KNN Regressor (k={n_neighbors}) - Ventana: {training_window} | Predicción: {prediction_window}')
        plt.xlabel('Muestras')
        plt.ylabel('Número de casos')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return utils.loss_function(predicted_values, observed_values)


def knn_evaluator(dataset, weeks):
    @functools.cache
    def knn_evaluation(individual):

        if individual[KNNParameter.TRAINING_WINDOW.value] <= 1 or individual[KNNParameter.TRAINING_WINDOW.value] > 50:
            return float('inf')
        if individual[KNNParameter.NEIGHBOURS.value] <= 0:
            return float('inf')

        loss = knn_model(
            dataset,
            individual[KNNParameter.TRAINING_WINDOW.value],
            weeks,
            individual[KNNParameter.NEIGHBOURS.value],
        )

        return loss

    return knn_evaluation

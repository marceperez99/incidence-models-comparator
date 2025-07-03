from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from metrics import loss_function


def knn_model(dataset, training_window, prediction_window, n_neighbors, return_predictions=False):
    dataset = dataset.copy()
    dataset['target'] = dataset['i_cases']
    dataset = dataset.sort_values(by='t')
    dataset = dataset.set_index('t')  # para búsquedas rápidas
    predicted_values = []
    observed_values = []
    dates = []

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
            dates.append(future_t[-1])

    if return_predictions:
        return loss_function.loss_function(observed_values, predicted_values), dates, observed_values, predicted_values

    return loss_function.loss_function(observed_values, predicted_values)


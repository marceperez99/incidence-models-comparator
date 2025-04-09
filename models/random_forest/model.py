from metrics import loss_function
import numpy as np
from sklearn.ensemble import RandomForestRegressor



def random_forest_model(dataset, training_window, prediction_window, n_estimators, max_depth, return_predictions=False):
    dataset = dataset.copy()
    dataset['target'] = dataset['i_cases']
    dataset = dataset.sort_values(by='t')
    dataset = dataset.set_index('t')  # mejora para acceso rápido
    predicted_values = []
    observed_values = []
    print(
        f'training_window={training_window}, prediction_window={prediction_window}, n_estimators={n_estimators}, max_depth={max_depth}')
    for i in range(len(dataset) - training_window - prediction_window + 1):  # Fix loop range

        section = dataset.iloc[i:i + training_window]

        # X_train centrado para mejorar estabilidad
        X_train = section.index.values.reshape(-1, 1)
        t_mean = X_train.mean()
        X_train_centered = X_train - t_mean
        y_train = section['target'].values

        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train_centered, y_train)

        # test_x: fechas a futuro centradas igual que X_train
        future_t = section.index.max() + np.arange(0, prediction_window * 7, 7)
        test_x = (future_t - t_mean).reshape(-1, 1)

        test_y = rf.predict(test_x)
        test_y = [max(0, y) for y in test_y]  # truncar valores negativos a 0

        try:
            real_y = dataset.loc[future_t]['target'].values
        except KeyError:
            continue  # si faltan datos reales, salta la iteración

        if len(real_y) == len(test_y):
            predicted_values.append(test_y[-1])
            observed_values.append(real_y[-1])

    # Compute and return the loss function (e.g., MAE)
    if return_predictions:
        return loss_function.loss_function(predicted_values, observed_values), observed_values, predicted_values

    return loss_function.loss_function(predicted_values, observed_values)

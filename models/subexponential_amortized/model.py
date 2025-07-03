from metrics import loss_function
import numpy as np
from sklearn.linear_model import LinearRegression


def subexponential_amort_model(dataset, training_window, prediction_window, return_predictions=False):
    dataset = dataset.copy()
    dataset['target'] = dataset['i_cases']

    predicted_values = []
    observed_values = []
    dates = []
    # Iterate over the data with a sliding window
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
            dates.append(test_t[-1])
            observed_values.append(np.log(filtered_dataset['target'].to_numpy()[-1] + 1e-8))

    if return_predictions:
        return loss_function.loss_function(observed_values, predicted_values), dates, observed_values, predicted_values

    return loss_function.loss_function(observed_values, predicted_values)

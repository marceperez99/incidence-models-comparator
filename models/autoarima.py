import numpy as np
from pmdarima.arima import auto_arima
import utils
import matplotlib.pyplot as plt

from population import ENFERMEDAD


def autoarima_model(dataset, prediction_window):
    # split into train and test sets
    X = dataset['i_cases'].values
    size = 10
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]

    predictions = list()
    for t in range(len(test)):
        # print(history)
        obs = test[t]
        history.append(obs)
        arima_model = auto_arima(np.array(history))
        prediction = arima_model.predict(n_periods=prediction_window)

        predictions.append(prediction[-1])

    # evaluate forecasts
    predictions = predictions[:-prediction_window]
    test = test[prediction_window:]


    predictions = [0 if prediction < 0 else prediction for prediction in predictions]
    plt.figure(figsize=(8, 5))
    plt.title(f'Modelo AutoARIMA / {ENFERMEDAD} (VP: {prediction_window})')
    plt.xlabel('Nro de semana epidemiologica')
    plt.ylabel('Nro de casos')
    plt.plot(test, label="Observaciones")
    plt.plot(predictions, label="Predicciones")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return utils.loss_function(predictions, test)

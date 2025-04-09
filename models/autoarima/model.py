from metrics import loss_function
import numpy as np
from pmdarima.arima import auto_arima


def autoarima_model(dataset, prediction_window, return_predictions=False):
    # split into train and test sets
    X = dataset['i_cases'].values
    size = 10
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]

    predictions = list()
    for t in range(len(test)):

        obs = test[t]
        history.append(obs)
        arima_model = auto_arima(np.array(history))
        prediction = arima_model.predict(n_periods=prediction_window)

        predictions.append(prediction[-1])

    # evaluate forecasts
    predictions = predictions[:-prediction_window]
    test = test[prediction_window:]

    predictions = [0 if prediction < 0 else prediction for prediction in predictions]

    if return_predictions:
        return loss_function.loss_function(predictions, test), test, predictions
    return loss_function.loss_function(predictions, test)

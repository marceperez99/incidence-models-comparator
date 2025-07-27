from metrics import loss_function
import numpy as np
from pmdarima.arima import auto_arima

def autoarima_model(dataset, prediction_window, return_predictions=False):
    # split into train and test sets
    X = dataset['t'].values
    Y = dataset['i_cases'].values
    size = 10
    train, test = Y[0:size], Y[size:len(Y)]

    print(f"[INFO] Train size: {len(train)}, Test size: {len(test)}")

    if np.isnan(train).any():
        print("[ERROR] NaNs found in training data")

    if np.isnan(test).any():
        print("[ERROR] NaNs found in test data")

    history = [x for x in train]
    predictions = []

    for t in range(len(test)):

        obs = test[t]

        if np.isnan(obs):
            print(f"[ERROR] NaN found in observation at test index {t}")
            continue

        history.append(obs)

        try:
            if np.isnan(history).any():
                print(f"[ERROR] NaN in history at iteration {t}")
                continue

            arima_model = auto_arima(np.array(history), error_action='ignore', suppress_warnings=True)
            prediction = arima_model.predict(n_periods=prediction_window)

            if np.isnan(prediction).any():
                print(f"[ERROR] NaN in prediction at iteration {t}")
                continue

            predictions.append(prediction[-1])
        except Exception as e:
            print(f"[ERROR] Failed at iteration {t}: {e}")
            continue

    if len(predictions) < prediction_window:
        print(f"[ERROR] Not enough predictions: expected at least {prediction_window}, got {len(predictions)}")
        return None

    predictions = predictions[:-prediction_window]
    test = test[prediction_window:]
    test_dates = X[size:len(Y)][prediction_window:]

    predictions = [0 if prediction < 0 else prediction for prediction in predictions]

    if return_predictions:
        return loss_function.loss_function(test, predictions), test_dates, test, predictions
    return loss_function.loss_function(test, predictions)

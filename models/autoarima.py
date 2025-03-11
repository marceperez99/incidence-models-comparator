import numpy as np
from pmdarima.arima import auto_arima
import utils

def autoarima_model(dataset, prediction_window):
    dataset = dataset.copy()

    for id_proy in dataset['id_proy'].unique():
        subset = dataset[dataset['id_proy'] == id_proy]


        # split into train and test sets
        X = subset['i_cases'].values
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
        # plt.figure(figsize=(8, 5))
        # plt.title(f'Training')
        # plt.plot(predictions, label="Observed")
        # plt.plot(test, label="Predicted")
        # plt.show()

        return utils.loss_function(predictions, test)

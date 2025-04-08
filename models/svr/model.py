from sklearn.preprocessing import StandardScaler
from  metrics import loss_function
import numpy as np
from sklearn.svm import SVR

def svr_model(dataset, training_window, prediction_window, c, epsilon, return_predictions=False):

    dataset = dataset.copy()
    dataset['target'] = dataset['i_cases']

    predicted_values = []
    observed_values = []

    dataset = dataset.sort_values(by=['t'])
    dataset = dataset.set_index('t')  # al principio


    for i in range(len(dataset) - training_window - prediction_window + 1):

        section = dataset.iloc[i:i + training_window]

        X_train = section.index.values.reshape(-1, 1)

        y_train = section['target'].values

        svr = SVR(kernel='linear', C=float(c), epsilon=epsilon)
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(X_train)
        svr.fit(x_train_scaled, y_train)


        test_x = (section.index.max() + np.arange(0, prediction_window * 7, 7)).reshape(-1, 1)
        test_x = test_x.ravel()
        test_x_scaled = scaler.transform(test_x.reshape(-1, 1))
        test_y = svr.predict(test_x_scaled)

        filtered_dataset = dataset.loc[test_x] if set(test_x).issubset(dataset.index) else None

        if filtered_dataset.shape[0] == len(test_x):
            predicted_values.append(max(0, test_y[-1]))
            observed_values.append(filtered_dataset['target'].values[-1])

    # Compute and return the loss function (e.g., MAE)
    if return_predictions:
        return loss_function.loss_function(predicted_values, observed_values), observed_values, predicted_values

    return loss_function.loss_function(predicted_values, observed_values)


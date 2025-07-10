from  metrics import loss_function
import numpy as np
from sklearn.linear_model import LinearRegression


def exponential_model(dataset, training_window, prediction_window, return_predictions=False):

    dataset = dataset.copy()
    dataset['target'] = dataset['i_cases']

    dates = []
    predicted_values = []
    observed_values = []
    # Iterate over the data with a sliding window
    for i in range(len(dataset) - training_window + 1):
        # Take a section of `training_window` rows
        section = dataset.iloc[i:i + training_window]

        # For example, print or process each section individually
        x = section['t'].values.reshape(-1, 1)
        y = np.log(section['target'])
        # training of model
        exponential = LinearRegression()
        exponential.fit(x, y)
        test_x = (section['t'].max(skipna=True) + np.arange(0, prediction_window * 7, 7))
        test_y = exponential.predict(test_x.reshape(-1, 1))

        filtered_dataset = dataset[dataset['t'].isin(test_x)]
        if filtered_dataset.shape[0] == test_x.shape[0]:
            dates.append(test_x[-1])
            predicted_values.append(test_y[-1])
            observed_values.append(np.log(filtered_dataset['target'].to_numpy()[-1]))

    if not len(observed_values):
        return float('inf')

    # Compute and return the loss function (e.g., MAE)
    if return_predictions:
        return loss_function.loss_function(observed_values, predicted_values), dates, observed_values, predicted_values

    return loss_function.loss_function(observed_values, predicted_values)

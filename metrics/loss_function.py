from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from constants import LOSS_METRIC

def loss_function(y_true, y_pred):
    if LOSS_METRIC == 'mape':
        return mean_absolute_percentage_error(y_true, y_pred)
    if LOSS_METRIC == 'mae':
        return mean_absolute_error(y_true, y_pred)

    return None
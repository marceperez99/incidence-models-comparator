import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import csv
import datetime
import os

def get_dataset(path):
    data_csv = pd.read_csv(path)

    df_0 = data_csv[(data_csv['disease'] != "DENGUE,CHIKUNGUNYA") & (data_csv['disease'] != "ARBOVIROSIS")]
    # Update the 'i_cases' column in the DataFrame 'df_0' where the value is 0 with a random exponential value between 1e-9 and 1.
    df_0.loc[df_0['i_cases'] == 0, 'i_cases'] = np.exp(np.random.uniform(np.log(1e-9), np.log(1)))

    # Group by specified columns and apply mutations
    df_0 = df_0.groupby(
        ['name', 'level', 'disease', 'classification'],
        group_keys=False
    ).apply(lambda group: (
        group.assign(id_proy=group['name'] + '-' + group['disease'] + '-' + group['classification'],
                     i_cases=group['i_cases'] + np.exp(np.random.uniform(np.log(1e-9), np.log(1), size=len(group))),
                     csum=group['i_cases'].cumsum())
    )
            ).reset_index(drop=True)
    # create columns of time
    df_0['date'] = pd.to_datetime(df_0['date'])
    df_0['t'] = (df_0['date'] - df_0['date'].min() + (1 * np.timedelta64(1, 'D'))) / np.timedelta64(1, 'D')
    return df_0


def loss_function(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / np.mean(y_true)
    return (mae + mape + rmse + nrmse) / 4


def log_experiment(parameters: dict, loss: float, algorithm: str, filename="experiment_log.csv", dataset: str = ''):
    # Define the header
    headers = ["timestamp", "dataset", "algorithm", "loss", "parameters"]

    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert parameters dict to string for storage
    parameters_str = str(parameters)

    # Check if file exists to write headers
    file_exists = os.path.isfile(filename)

    # Append data to the CSV file
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write headers only if the file is new
        if not file_exists:
            writer.writerow(headers)

        # Write the new experiment log
        writer.writerow([timestamp, dataset, algorithm, loss, parameters_str])

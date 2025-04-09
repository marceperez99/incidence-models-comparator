import pandas as pd

from evaluation.persist import save_as_csv
from models.autoarima.model import autoarima_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from evaluation import graphing, metrics


def run_autoarima(datasets, weeks):
    for dataset in datasets:

        for week_i in range(1, weeks + 1):
            loss, y_true, y_pred = autoarima_model(dataset, week_i, return_predictions=True)
            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            nrmse = rmse / np.mean(y_true)

            # saving results
            disease = dataset['disease'].iloc[0].lower()
            filename = f"{dataset['name'].iloc[0]}_{dataset['classification'].iloc[0]}_{week_i}".lower()
            save_as_csv(pd.DataFrame({'Observed': y_true, 'Predicted': y_pred}),
                        f'{filename}.csv', output_dir=f'outputs/predictions/autoarima/{disease}')

            title = f"Modelo Autoarima ({dataset['disease'].iloc[0]})"
            descripcion = f'VP:{week_i} semanas'
            graphing.plot_observed_vs_predicted(y_true, y_pred, f'plt_obs_pred_{filename}',
                                                output_dir=f'outputs/plots/autoarima/{disease}', title=title,
                                                description=descripcion)
            graphing.plot_scatter(y_true, y_pred, f'plt_scatter_{filename}', 'Autoarima', title=title,
                                  description=descripcion, output_dir=f'outputs/plots/autoarima/{disease}')
            metrics.log_model_metrics('Autoarima', disease, dataset['classification'].iloc[0],
                                      dataset['name'].iloc[0], week_i, mae=mae, mape=mape, nrmse=nrmse, loss=loss,
                                      rmse=rmse,
                                      hyperparams={
                                          'prediction_window': week_i,
                                      })
